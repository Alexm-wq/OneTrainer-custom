"""Streamed DPO ported from preview 7e189924 (B-sized replay/backward)."""
import copy
from contextlib import contextmanager

import modules.util.multi_gpu_util as multi
from modules.util.compile_util import init_compile

import torch
from torch import Tensor


class ModelSetupStreamedDPOMixin:
    def _create_dpo_stream_batches(self, batch: dict) -> tuple[dict, dict, int]:
        """Build B-sized chosen/rejected batches without allocating a 2B tensor.

        The key mapping intentionally mirrors _create_dpo_batched_batch. Shared
        conditioning and metadata are reused; paired rejected values are exposed
        under the ordinary key names expected by predict().
        """
        latent_image = batch.get("latent_image")
        rejected_latent = batch.get("latent_image_rejected")
        if not isinstance(latent_image, torch.Tensor) or latent_image.ndim == 0:
            raise RuntimeError(
                "Streamed DPO batch must contain a batched latent_image tensor"
            )
        if not isinstance(rejected_latent, torch.Tensor):
            raise RuntimeError(
                "Streamed DPO batch must contain latent_image_rejected as a Tensor"
            )
        if rejected_latent.shape != latent_image.shape:
            raise RuntimeError(
                "Streamed DPO latent shape mismatch: "
                f"latent_image {tuple(latent_image.shape)} != "
                f"latent_image_rejected {tuple(rejected_latent.shape)}"
            )

        chosen_b = int(latent_image.shape[0])
        if chosen_b <= 0:
            raise RuntimeError("Streamed DPO batch is empty")

        rejected_key_map = {
            "latent_image": "latent_image_rejected",
            "image": "image_rejected",
            "image_path": "image_path_rejected",
            "chosen_image_path": "rejected_image_path",
            "chosen_source_path": "rejected_source_path",
        }
        chosen: dict = {}
        rejected: dict = {}

        for key, value in batch.items():
            if key.endswith("_rejected") or key.startswith("rejected_"):
                continue

            rejected_key = rejected_key_map.get(key)
            if rejected_key is None and key.startswith("chosen_"):
                candidate = "rejected_" + key[len("chosen_"):]
                if candidate in batch:
                    rejected_key = candidate

            rejected_value = (
                batch[rejected_key]
                if rejected_key is not None and rejected_key in batch
                else value
            )

            if rejected_key is not None and rejected_key in batch:
                if isinstance(value, torch.Tensor):
                    if not isinstance(rejected_value, torch.Tensor):
                        raise TypeError(
                            f"DPO batch key '{key}' is Tensor but rejected key "
                            f"'{rejected_key}' is {type(rejected_value).__name__}"
                        )
                    if (
                        value.ndim == 0
                        or rejected_value.ndim == 0
                        or int(value.shape[0]) != chosen_b
                        or int(rejected_value.shape[0]) != chosen_b
                    ):
                        raise RuntimeError(
                            f"DPO paired tensor keys '{key}'/'{rejected_key}' "
                            f"must both have batch size {chosen_b}"
                        )
                elif isinstance(value, (list, tuple)):
                    if not isinstance(rejected_value, (list, tuple)):
                        raise TypeError(
                            f"DPO batch key '{key}' is a sequence but rejected "
                            f"key '{rejected_key}' is "
                            f"{type(rejected_value).__name__}"
                        )
                    if len(value) != chosen_b or len(rejected_value) != chosen_b:
                        raise RuntimeError(
                            f"DPO paired sequence keys '{key}'/'{rejected_key}' "
                            f"must both have length {chosen_b}"
                        )

            chosen[key] = value
            rejected[key] = rejected_value

        return chosen, rejected, chosen_b

    @contextmanager
    def _dpo_stream_predict_context(self):
        """Mark a B-sized streamed forward as paired DPO.

        The context lives inside the checkpointed callable so backward
        recomputation gets exactly the same conditioning behavior as its
        original forward. ContextVar keeps independent autograd worker contexts
        from racing on shared setup state.
        """
        token = self._dpo_stream_active.set(True)
        try:
            yield
        finally:
            self._dpo_stream_active.reset(token)

    def _dpo_conditioning_locked(self) -> bool:
        return bool(
            self._dpo_paired_half is not None
            or self._dpo_stream_active.get()
        )


    def _streamed_dpo_scores(self, model, batch, config, train_progress):
        chosen, rejected, count = self._create_dpo_stream_batches(batch)
        # Backward replay must use the same progress-based noise/timestep seed.
        progress = copy.copy(train_progress)

        def score(branch):
            output = self.predict(model, branch, config, progress)
            logp = self.rlhf_logp_per_sample(model, branch, output, config)
            logp = self._validate_rlhf_logp_per_sample(logp, count, "streamed", paired=False)
            return output, logp

        with torch.no_grad(), self.reference_model(model, config), self._dpo_stream_predict_context():
            output, ref_chosen = score(chosen)
            del output
            output, ref_rejected = score(rejected)
            del output

        include_chosen = config.rlhf_chosen_weight > 0 and not self._dpo_is_validation
        parameters = ()
        direct_backward = multi.world_size() <= 1
        if torch.is_grad_enabled():
            optimizer = getattr(model, "optimizer", None)
            groups = getattr(optimizer, "param_groups", None)
            if not isinstance(groups, list):
                raise RuntimeError("Streamed DPO requires initialized optimizer parameter groups")
            parameters = tuple(dict.fromkeys(
                parameter for group in groups for parameter in group["params"]
                if parameter.requires_grad
            ))
            if not parameters:
                raise RuntimeError("Streamed DPO requires trainable optimizer parameters")
            if config.optimizer.optimizer.supports_fused_back_pass() and config.optimizer.fused_back_pass:
                raise ValueError("Streamed DPO requires Fused Back Pass disabled; ordinary gradient accumulation is supported")
            if config.multi_gpu and config.fused_gradient_reduce:
                raise ValueError("Streamed DPO requires Fused Gradient Reduce disabled; gradients are reduced after all branches")
            if not direct_backward and any(
                callable(getattr(value, "offload_activated", None)) and value.offload_activated()
                for value in vars(model).values()
            ):
                raise ValueError("Streamed DPO with layer/activation offloading currently supports one GPU")

        def policy_branch(branch, include_supervised):
            timestep_holder = {}

            def evaluate(latent):
                replay = dict(branch)
                replay["latent_image"] = latent
                with self._dpo_stream_predict_context():
                    output, logp = score(replay)
                    timestep = output.get("timestep")
                    timestep_holder["value"] = timestep.detach() if isinstance(timestep, Tensor) else None
                    supervised = self.calculate_loss(model, replay, output, config) if include_supervised else logp.new_empty(0)
                    del output
                return logp, supervised

            class StreamedDPOBranch(torch.autograd.Function):
                @staticmethod
                def forward(ctx, latent, *params):
                    # Value-only forwards retain no transformer activation graph.
                    ctx.save_for_backward(latent)
                    ctx.cpu_rng = torch.get_rng_state()
                    device = torch.device(self.train_device)
                    ctx.cuda_device = (
                        device.index if device.index is not None else torch.cuda.current_device()
                    ) if device.type == "cuda" else None
                    ctx.cuda_rng = torch.cuda.get_rng_state(ctx.cuda_device) if ctx.cuda_device is not None else None
                    ctx.autocast_device = device.type
                    ctx.autocast_enabled = torch.is_autocast_enabled(device.type)
                    ctx.autocast_dtype = torch.get_autocast_dtype(device.type)
                    return evaluate(latent)

                @staticmethod
                def backward(ctx, grad_logp, grad_supervised):
                    # The replay forward runs in the autograd thread and may compile new shapes.
                    init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
                    (latent,) = ctx.saved_tensors
                    devices = [] if ctx.cuda_device is None else [ctx.cuda_device]
                    with (
                        torch.random.fork_rng(devices=devices),
                        torch.enable_grad(),
                        torch.autocast(ctx.autocast_device, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype),
                        self.preference_dropout_disabled(model),
                        self._dpo_stream_predict_context(),
                    ):
                        torch.set_rng_state(ctx.cpu_rng)
                        if ctx.cuda_rng is not None:
                            torch.cuda.set_rng_state(ctx.cuda_rng, ctx.cuda_device)
                        logp, supervised = evaluate(latent)
                        outputs, gradients = [], []
                        if grad_logp is not None and logp.requires_grad:
                            outputs.append(logp)
                            gradients.append(grad_logp)
                        if include_supervised and grad_supervised is not None and supervised.requires_grad:
                            outputs.append(supervised)
                            gradients.append(grad_supervised)
                        if outputs and direct_backward:
                            # Matches preview: nested backward supports OT's reentrant
                            # offload checkpoints and adds straight into .grad buffers.
                            # Incoming derivatives already contain AMP/mixture scaling.
                            torch.autograd.backward(outputs, gradients)
                            parameter_gradients = (None,) * len(parameters)
                        elif outputs:
                            parameter_gradients = torch.autograd.grad(
                                outputs, parameters, grad_outputs=gradients, allow_unused=True,
                            )
                        else:
                            parameter_gradients = (None,) * len(parameters)
                    return (None, *parameter_gradients)

            if torch.is_grad_enabled():
                logp, supervised = StreamedDPOBranch.apply(branch["latent_image"], *parameters)
            else:
                logp, supervised = evaluate(branch["latent_image"])
            return logp, supervised, timestep_holder.get("value")

        policy_chosen, chosen_loss, timestep = policy_branch(chosen, include_chosen)
        policy_rejected, _, _ = policy_branch(rejected, False)
        return (count, ref_chosen, ref_rejected, policy_chosen, policy_rejected,
                chosen_loss if include_chosen else None, timestep)
