import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.Flux2SelfFlow import (
    Flux2SelfFlowEMA,
    Flux2SelfFlowProjector,
    flux2_flow_sigma,
    flux2_interpolate_token_view,
    flux2_self_flow_forward,
    flux2_stratified_token_indices,
    flux2_structural_alignment_loss,
    flux2_token_weight_to_spatial,
)
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode


class _TimeGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, dim, bias=False)

    def forward(self, timestep, guidance):
        value = timestep[:, None]
        if guidance is not None:
            value = value + guidance[:, None]
        return self.linear(value)


class _Modulation(nn.Module):
    def __init__(self, dim, sets):
        super().__init__()
        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 3 * sets, bias=False)

    def forward(self, embedding):
        return self.linear(self.act_fn(embedding))


def _split_modulation(modulation, sets):
    if modulation.ndim == 2:
        modulation = modulation.unsqueeze(1)
    chunks = torch.chunk(modulation, 3 * sets, dim=-1)
    return tuple(chunks[index * 3:(index + 1) * 3] for index in range(sets))


class _DoubleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.image_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb_mod_img,
            temb_mod_txt,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
    ):
        del image_rotary_emb, joint_attention_kwargs
        image_attn, image_mlp = _split_modulation(temb_mod_img, 2)
        text_attn, text_mlp = _split_modulation(temb_mod_txt, 2)

        def update(value, norm, mods):
            for shift, scale, gate in mods:
                value = value + torch.tanh((1 + scale) * norm(value) + shift) * torch.sigmoid(gate)
            return value

        return (
            update(encoder_hidden_states, self.text_norm, (text_attn, text_mlp)),
            update(hidden_states, self.image_norm, (image_attn, image_mlp)),
        )


class _SingleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb_mod,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
    ):
        del image_rotary_emb, joint_attention_kwargs
        if encoder_hidden_states is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        shift, scale, gate = _split_modulation(temb_mod, 1)[0]
        return hidden_states + torch.tanh((1 + scale) * self.norm(hidden_states) + shift) * torch.sigmoid(gate)


class _PositionEmbedding(nn.Module):
    def forward(self, ids):
        values = ids[:, 0].to(dtype=torch.float32)[:, None]
        return values, values


class _OutputNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, embedding):
        projected = self.linear(self.silu(embedding).to(hidden_states.dtype))
        scale, shift = torch.chunk(projected, 2, dim=1)
        return self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]


class _FakeFlux2(nn.Module):
    def __init__(self, input_dim=4, context_dim=5, hidden_dim=8, double_layers=2, single_layers=4):
        super().__init__()
        self.time_guidance_embed = _TimeGuidance(hidden_dim)
        self.double_stream_modulation_img = _Modulation(hidden_dim, 2)
        self.double_stream_modulation_txt = _Modulation(hidden_dim, 2)
        self.single_stream_modulation = _Modulation(hidden_dim, 1)
        self.x_embedder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.context_embedder = nn.Linear(context_dim, hidden_dim, bias=False)
        self.pos_embed = _PositionEmbedding()
        self.transformer_blocks = nn.ModuleList([_DoubleBlock(hidden_dim) for _ in range(double_layers)])
        self.single_transformer_blocks = nn.ModuleList([_SingleBlock(hidden_dim) for _ in range(single_layers)])
        self.norm_out = _OutputNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, input_dim, bias=False)

    def native_forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance=None):
        text_length = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        embedding = self.time_guidance_embed(timestep, guidance)
        image_mod = self.double_stream_modulation_img(embedding)
        text_mod = self.double_stream_modulation_txt(embedding)
        single_mod = self.single_stream_modulation(embedding)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        image_rope = self.pos_embed(img_ids[0] if img_ids.ndim == 3 else img_ids)
        text_rope = self.pos_embed(txt_ids[0] if txt_ids.ndim == 3 else txt_ids)
        rope = (
            torch.cat([text_rope[0], image_rope[0]], dim=0),
            torch.cat([text_rope[1], image_rope[1]], dim=0),
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states, encoder_hidden_states, image_mod, text_mod, rope, None
            )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, None, single_mod, rope, None)
        hidden_states = hidden_states[:, text_length:]
        return self.proj_out(self.norm_out(hidden_states, embedding))


class _DPOTestModel:
    def __init__(self):
        self.policy = nn.Parameter(torch.tensor(0.25))
        self.optimizer = torch.optim.SGD([self.policy], lr=0.1)


class _DPOTestSetup(BaseModelSetup):
    def __init__(self):
        super().__init__(torch.device("cpu"), torch.device("cpu"), False)
        self.reference_flags = []
        self.streamed_backward_count = 0

    def create_parameters(self, model, config):
        return None

    def setup_optimizations(self, model, config):
        pass

    def setup_model(self, model, config):
        pass

    def setup_train_device(self, model, config):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        del config, train_progress, deterministic
        is_reference = self._dpo_reference_prediction()
        self.reference_flags.append(is_reference)
        latent_score = batch["latent_image"].flatten(1).mean(dim=1)
        if is_reference:
            logp = latent_score * 0.5
        else:
            # Retain a zero-valued policy dependency so every DPO objective is
            # differentiable while its gradient contribution remains zero.
            logp = latent_score + model.policy * 0.0
        return {
            "logp": logp,
            "timestep": torch.zeros(logp.shape[0], dtype=torch.long),
            "auxiliary": (model.policy - 1.0).pow(2),
            "supervised": model.policy.pow(2),
        }

    def calculate_loss(self, model, batch, data, config):
        del model, batch, config
        return data["supervised"]

    def after_optimizer_step(self, model, config, train_progress):
        pass

    def rlhf_logp_per_sample(self, model, batch, data, config):
        del model, batch, config
        return data["logp"]

    def rlhf_policy_auxiliary_loss(self, model, batch, data, config):
        del model, batch, config
        return data["auxiliary"]

    def after_streamed_dpo_branch_backward(self, model, config, train_progress):
        del model, config, train_progress
        self.streamed_backward_count += 1

    @contextmanager
    def reference_model(self, model, config, reference_mode=None, reference_key=None):
        del model, config, reference_mode, reference_key
        yield

    def _dpo_pair_identity(self, batch, index):
        del batch
        return f"pair-{index}"

    def _write_dpo_pair_csv_log(self, **kwargs):
        pass

    def _write_dpo_bad_pair_csv_log(self, **kwargs):
        pass


class _DPOTestSetupNoAuxiliary(_DPOTestSetup):
    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        del config, train_progress, deterministic
        is_reference = self._dpo_reference_prediction()
        self.reference_flags.append(is_reference)
        latent_score = batch["latent_image"].flatten(1).mean(dim=1)
        logp = latent_score * (0.5 if is_reference else model.policy)
        return {
            "logp": logp,
            "timestep": torch.zeros(logp.shape[0], dtype=torch.long),
            "supervised": model.policy.pow(2),
        }

    def rlhf_policy_auxiliary_loss(self, model, batch, data, config):
        del model, batch, data, config
        return None


class _DPOTestSetupBranchAuxiliary(_DPOTestSetup):
    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        output = super().predict(
            model,
            batch,
            config,
            train_progress,
            deterministic=deterministic,
        )
        latent_score = batch["latent_image"].flatten(1).mean(dim=1)
        output["auxiliary_per_sample"] = (model.policy - latent_score).pow(2)
        return output

    def rlhf_policy_auxiliary_loss(self, model, batch, data, config):
        del model, batch, config
        return data["auxiliary_per_sample"].mean()


def _dpo_test_config(objective):
    optimizer_kind = SimpleNamespace(supports_fused_back_pass=lambda: False)
    return SimpleNamespace(
        rlhf_dpo_objective=objective,
        rlhf_dpo_beta=2.0,
        rlhf_dpo_label_smoothing=0.0,
        rlhf_dpo_ipo_tau=10.0,
        rlhf_supervised_mix=0.0,
        rlhf_dpo_chosen_reward_anchor=False,
        rlhf_dpo_hard_pair_curriculum=False,
        optimizer=SimpleNamespace(
            optimizer=optimizer_kind,
            fused_back_pass=False,
        ),
    )


def _dpo_test_batch():
    return {
        "latent_image": torch.tensor([[2.0], [1.0]]),
        "latent_image_rejected": torch.tensor([[-1.0], [-2.0]]),
    }


class Flux2SelfFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.transformer = _FakeFlux2()
        self.image = torch.randn(2, 6, 4)
        self.text = torch.randn(2, 3, 5)
        self.timestep = torch.tensor([0.2, 0.7])
        self.image_ids = torch.zeros(2, 6, 4)
        self.text_ids = torch.zeros(2, 3, 4)

    def test_homogeneous_token_timestep_matches_native_forward(self):
        native = self.transformer.native_forward(
            self.image, self.text, self.timestep, self.image_ids, self.text_ids
        )
        token_timestep = self.timestep[:, None].expand(-1, self.image.shape[1])
        self_flow = flux2_self_flow_forward(
            transformer=self.transformer,
            hidden_states=self.image,
            encoder_hidden_states=self.text,
            image_timestep=token_timestep,
            text_timestep=self.timestep,
            img_ids=self.image_ids,
            txt_ids=self.text_ids,
            capture_layer=1,
        )
        torch.testing.assert_close(self_flow.sample, native, rtol=1e-5, atol=1e-5)

    def test_pinned_diffusers_flux2_homogeneous_regression(self):
        from diffusers import Flux2Transformer2DModel

        torch.manual_seed(7)
        transformer = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=12,
            timestep_guidance_channels=16,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        ).eval()
        image = torch.randn(2, 6, 8)
        text = torch.randn(2, 3, 12)
        timestep = torch.tensor([0.2, 0.7])
        image_ids = torch.zeros(2, 6, 4)
        text_ids = torch.zeros(2, 3, 4)

        with torch.no_grad():
            native = transformer(
                hidden_states=image,
                encoder_hidden_states=text,
                timestep=timestep,
                img_ids=image_ids,
                txt_ids=text_ids,
            ).sample
            token_conditioned = flux2_self_flow_forward(
                transformer=transformer,
                hidden_states=image,
                encoder_hidden_states=text,
                image_timestep=timestep[:, None].expand(-1, image.shape[1]),
                text_timestep=timestep,
                img_ids=image_ids,
                txt_ids=text_ids,
                capture_layer=0,
            ).sample

        torch.testing.assert_close(token_conditioned, native, rtol=1e-5, atol=1e-6)

    def test_mixed_timestep_and_teacher_early_exit_shapes(self):
        token_timestep = self.timestep[:, None].expand(-1, self.image.shape[1]).clone()
        token_timestep[:, ::2] = torch.tensor([0.8, 0.1])[:, None]
        student = flux2_self_flow_forward(
            self.transformer, self.image, self.text, token_timestep, self.timestep,
            self.image_ids, self.text_ids, capture_layer=0,
        )
        teacher = flux2_self_flow_forward(
            self.transformer, self.image, self.text, self.timestep, self.timestep,
            self.image_ids, self.text_ids, stop_at_layer=2,
        )
        self.assertEqual(student.sample.shape, self.image.shape)
        self.assertEqual(student.feature.shape[:2], self.image.shape[:2])
        self.assertIsNone(teacher.sample)
        self.assertEqual(teacher.feature.shape, student.feature.shape)

    def test_prediction_only_forward_skips_feature_capture(self):
        prediction = flux2_self_flow_forward(
            self.transformer,
            self.image,
            self.text,
            self.timestep,
            self.timestep,
            self.image_ids,
            self.text_ids,
        )
        self.assertEqual(prediction.sample.shape, self.image.shape)
        self.assertIsNone(prediction.feature)

    def test_token_interpolation_and_spatial_weight(self):
        clean = torch.zeros(1, 4, 2)
        noise = torch.ones_like(clean)
        timestep = torch.tensor([[0, 249, 499, 999]])
        interpolated = flux2_interpolate_token_view(clean, noise, timestep, 1000)
        expected = flux2_flow_sigma(timestep, 1000).unsqueeze(-1).expand_as(clean)
        torch.testing.assert_close(interpolated, expected)

        spatial = flux2_token_weight_to_spatial(expected[..., 0], 2, 2)
        self.assertEqual(spatial.shape, (1, 1, 4, 4))
        torch.testing.assert_close(
            spatial[:, :, 0:2, 0:2],
            torch.full_like(spatial[:, :, 0:2, 0:2], expected[0, 0, 0]),
        )

    def test_cpu_ema_swap_update_resume_and_accumulation_isolation(self):
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        identity = id(parameter)
        ema = Flux2SelfFlowEMA([parameter], decay=0.5)

        for _ in range(3):
            with ema.teacher_parameters():
                self.assertEqual(ema.optimization_steps, 0)
        self.assertEqual(ema.optimization_steps, 0)

        with torch.no_grad():
            parameter.add_(2.0)
        ema.update_after_optimizer_step()
        torch.testing.assert_close(ema.ema_parameters[0], torch.tensor([2.0, 3.0]))

        with ema.teacher_parameters():
            self.assertEqual(id(parameter), identity)
            torch.testing.assert_close(parameter, torch.tensor([2.0, 3.0]))
        self.assertEqual(id(parameter), identity)
        torch.testing.assert_close(parameter, torch.tensor([3.0, 4.0]))
        self.assertEqual(ema.optimization_steps, 1)

        restored = Flux2SelfFlowEMA([parameter], decay=0.1, state_dict=ema.state_dict())
        torch.testing.assert_close(restored.ema_parameters[0], ema.ema_parameters[0])
        self.assertEqual(restored.optimization_steps, 1)
        self.assertEqual(restored.decay, 0.5)

    def test_projector_is_trainable_but_teacher_target_is_detached(self):
        projector = Flux2SelfFlowProjector(8)
        student = torch.randn(2, 4, 8, requires_grad=True)
        with torch.inference_mode():
            inference_teacher = torch.randn(2, 4, 8)
        teacher = inference_teacher.detach().clone()
        loss = 1.0 - torch.nn.functional.cosine_similarity(projector(student), teacher, dim=-1).mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in projector.parameters()))
        self.assertFalse(teacher.requires_grad)
        self.assertFalse(teacher.is_inference())

    def test_structural_alignment_is_sampled_deterministic_and_teacher_detached(self):
        teacher = torch.randn(2, 32, 8, requires_grad=True)
        student = teacher.detach().clone().requires_grad_(True)

        generator = torch.Generator(device="cpu").manual_seed(91)
        identical_loss = flux2_structural_alignment_loss(
            student,
            teacher,
            sample_count=8,
            generator=generator,
        )
        torch.testing.assert_close(identical_loss, torch.zeros_like(identical_loss), atol=1e-7, rtol=0)
        identical_loss.sum().backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

        changed_student = student.detach().clone()
        changed_student[:, 8:16] = torch.roll(changed_student[:, 8:16], shifts=1, dims=-1)
        first = flux2_structural_alignment_loss(
            changed_student,
            teacher,
            sample_count=8,
            generator=torch.Generator(device="cpu").manual_seed(91),
        )
        second = flux2_structural_alignment_loss(
            changed_student,
            teacher,
            sample_count=8,
            generator=torch.Generator(device="cpu").manual_seed(91),
        )
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.all(first > 0))

        indices = flux2_stratified_token_indices(
            num_tokens=64,
            sample_count=8,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(7),
        )
        self.assertEqual(indices.unique().numel(), 8)
        for index, value in enumerate(indices.tolist()):
            self.assertGreaterEqual(value, index * 8)
            self.assertLess(value, (index + 1) * 8)

    def test_flux2_training_tab_exposes_self_flow_toggle(self):
        view_path = Path("modules/ui/BaseTrainingTabView.py")
        if not view_path.exists():
            self.skipTest("Targeted DPO source bundle does not include the unchanged Training-tab file.")
        view_source = view_path.read_text(encoding="utf-8")
        config_source = Path("modules/util/config/TrainConfig.py").read_text(encoding="utf-8")
        self.assertIn("self.__create_self_flow_frame(column_1, 3, ui_state)", view_source)
        self.assertIn('self.components.switch(frame, 0, 1, ui_state, "self_flow_enabled")', view_source)
        self.assertIn('self.components.switch(frame, 3, 1, ui_state, "self_flow_structural_enabled")', view_source)
        self.assertIn('(\"self_flow_enabled\", False, bool, False)', config_source)
        self.assertIn('(\"self_flow_structural_enabled\", False, bool, False)', config_source)
        self.assertIn('(\"self_flow_structural_weight\", 0.25, float, False)', config_source)
        self.assertIn('(\"self_flow_structural_tokens\", 256, int, False)', config_source)

    def test_self_flow_dpo_batched_and_streamed_all_objectives(self):
        for streamed in (False, True):
            for objective in (
                DPOObjective.SIGMOID,
                DPOObjective.IPO,
                DPOObjective.ANCHORED_REJECT,
            ):
                with self.subTest(streamed=streamed, objective=objective):
                    model = _DPOTestModel()
                    setup = _DPOTestSetup()
                    loss = setup.calculate_dpo_loss(
                        model,
                        _dpo_test_batch(),
                        _dpo_test_config(objective),
                        SimpleNamespace(global_step=0),
                        objective=objective,
                        reference_mode=DPORefMode.NEW_ADAPTER,
                        streamed=streamed,
                    )
                    loss.backward()

                    # DPO logp has a deliberately zero policy gradient. Every
                    # objective receives the policy-only auxiliary (-1.5), and
                    # Anchored Reject additionally receives one ordinary
                    # chosen supervised gradient (+0.5).
                    expected_gradient = (
                        -1.0
                        if objective == DPOObjective.ANCHORED_REJECT
                        else -1.5
                    )
                    torch.testing.assert_close(
                        model.policy.grad,
                        torch.tensor(expected_gradient),
                    )
                    self.assertAlmostEqual(
                        setup.get_last_dpo_metrics()["policy_auxiliary_loss"],
                        0.5625,
                    )
                    if streamed:
                        self.assertEqual(setup.reference_flags[:2], [True, True])
                        self.assertTrue(all(flag is False for flag in setup.reference_flags[2:]))
                        self.assertEqual(setup.streamed_backward_count, 2)
                    else:
                        self.assertEqual(setup.reference_flags, [True, False])

    def test_anchored_reject_uses_one_full_chosen_supervised_loss(self):
        for streamed in (False, True):
            for configured_mix in (0.0, 0.25, 2.0):
                with self.subTest(streamed=streamed, configured_mix=configured_mix):
                    model = _DPOTestModel()
                    setup = _DPOTestSetup()
                    config = _dpo_test_config(DPOObjective.ANCHORED_REJECT)
                    config.rlhf_supervised_mix = configured_mix
                    # Legacy chosen-target settings must have no effect.
                    config.rlhf_dpo_anchored_chosen_target = 100.0
                    config.rlhf_dpo_anchored_chosen_weight = 100.0

                    loss = setup.calculate_dpo_loss(
                        model,
                        _dpo_test_batch(),
                        config,
                        SimpleNamespace(global_step=0),
                        objective=DPOObjective.ANCHORED_REJECT,
                        reference_mode=DPORefMode.NEW_ADAPTER,
                        streamed=streamed,
                    )
                    loss.backward()

                    metrics = setup.get_last_dpo_metrics()
                    self.assertAlmostEqual(metrics["chosen_supervised_weight"], 1.0)
                    self.assertAlmostEqual(metrics["chosen_supervised_loss"], 0.0625)
                    self.assertAlmostEqual(loss.detach().item(), 0.625)
                    torch.testing.assert_close(model.policy.grad, torch.tensor(-1.0))

    def test_streamed_dpo_without_policy_auxiliary_matches_batched(self):
        for objective in (
            DPOObjective.SIGMOID,
            DPOObjective.IPO,
            DPOObjective.ANCHORED_REJECT,
        ):
            results = []
            for streamed in (False, True):
                model = _DPOTestModel()
                setup = _DPOTestSetupNoAuxiliary()
                loss = setup.calculate_dpo_loss(
                    model,
                    _dpo_test_batch(),
                    _dpo_test_config(objective),
                    SimpleNamespace(global_step=0),
                    objective=objective,
                    reference_mode=DPORefMode.NEW_ADAPTER,
                    streamed=streamed,
                )
                loss.backward()
                results.append((loss.detach(), model.policy.grad.detach()))

            with self.subTest(objective=objective):
                torch.testing.assert_close(results[0][0], results[1][0])
                torch.testing.assert_close(results[0][1], results[1][1])

    def test_dpo_policy_auxiliary_is_chosen_only(self):
        for streamed in (False, True):
            with self.subTest(streamed=streamed):
                model = _DPOTestModel()
                setup = _DPOTestSetupBranchAuxiliary()
                loss = setup.calculate_dpo_loss(
                    model,
                    _dpo_test_batch(),
                    _dpo_test_config(DPOObjective.SIGMOID),
                    SimpleNamespace(global_step=0),
                    objective=DPOObjective.SIGMOID,
                    reference_mode=DPORefMode.NEW_ADAPTER,
                    streamed=streamed,
                )
                loss.backward()

                # Chosen scores are [2, 1]. At policy=0.25 their auxiliary is
                # 1.8125 with gradient -2.5. Including rejected scores [-1,
                # -2] would produce a different value and gradient.
                self.assertAlmostEqual(
                    setup.get_last_dpo_metrics()["policy_auxiliary_loss"],
                    1.8125,
                )
                torch.testing.assert_close(model.policy.grad, torch.tensor(-2.5))

    def test_self_flow_dpo_pair_lock_and_reference_isolation_are_wired(self):
        flux_source = Path("modules/modelSetup/BaseFlux2Setup.py").read_text(encoding="utf-8")
        base_source = Path("modules/modelSetup/BaseModelSetup.py").read_text(encoding="utf-8")
        lora_source = Path("modules/modelSetup/Flux2LoRASetup.py").read_text(encoding="utf-8")

        self.assertIn("token_mask = self._apply_dpo_paired_rng(token_mask)", flux_source)
        self.assertIn("if not dpo_reference_forward:", flux_source)
        self.assertIn("capture_layer=None if dpo_reference_forward", flux_source)
        self.assertGreaterEqual(base_source.count("self._dpo_reference_predict_context()"), 2)
        self.assertIn("grad_auxiliary", base_source)
        self.assertIn("rejected_auxiliary_loss", base_source)
        self.assertIn("if data.get('self_flow_dpo_policy', False):", flux_source)
        self.assertIn("self_flow_structural_loss_per_sample", flux_source)
        self.assertIn("config.self_flow_structural_weight * structural_loss", flux_source)
        self.assertNotIn("Self-Flow cannot be combined with DPO/RLHF", lora_source)

        mask = torch.tensor([
            [True, False, True],
            [False, True, False],
            [False, False, False],
            [True, True, True],
        ])
        holder = SimpleNamespace(_dpo_paired_half=2)
        locked_mask = ModelSetupNoiseMixin._apply_dpo_paired_rng(holder, mask)
        torch.testing.assert_close(locked_mask[2:], locked_mask[:2])

    def test_dpo_csv_compacts_per_token_timesteps(self):
        timesteps = torch.tensor([
            [733, 733, 856, 733],
            [349, 789, 349, 349],
        ])
        self.assertEqual(
            BaseModelSetup._dpo_csv_timestep_value(timesteps, 0),
            "n=4;values=733:3|856:1",
        )
        self.assertEqual(
            BaseModelSetup._dpo_csv_timestep_value(timesteps, 1),
            "n=4;values=349:3|789:1",
        )
        self.assertEqual(
            BaseModelSetup._dpo_csv_timestep_value(torch.tensor([475]), 0),
            475,
        )


if __name__ == "__main__":
    unittest.main()
