"""Identity reward on CPU with tiny random models; no ArcFace or detector weights are downloaded."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.util import create

# isort: split
from modules.modelSetup.BaseFlux2Setup import BaseFlux2Setup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.identity_reward import (
    ARCFACE_TEMPLATE,
    KNOWN_HASHES,
    YUNET_URL,
    IResNet,
    align_face,
    alignment_matrix,
    identity_losses,
    latent_face_box,
    load_arcface,
    resolve_model_file,
    similarity_transform,
)
from modules.util.TrainProgress import TrainProgress
from modules.util.ui.validation import validate_path

import torch
from torch.nn import functional as F

import cv2
import numpy as np
import test_moss_pairing
from test_moss_image import tiny_model, train_config


def tiny_arcface(path):
    torch.manual_seed(3)
    net = IResNet([1, 1, 1, 1], num_features=16)
    torch.save({"module." + key: value for key, value in net.state_dict().items()}, path)
    return load_arcface(str(path))


def landmarks_at(x, y, scale):
    return torch.tensor(ARCFACE_TEMPLATE) * scale + torch.tensor([x, y])


class AlignmentTest(unittest.TestCase):
    def test_similarity_recovers_transform_and_is_least_squares(self):
        rng = np.random.default_rng(0)
        scale, angle, shift = 1.7, 0.3, np.array([12.0, -7.0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        target = np.array(ARCFACE_TEMPLATE)
        source = (target - shift) @ np.linalg.inv(scale * rotation).T
        matrix = similarity_transform(torch.tensor(source), torch.tensor(target))
        np.testing.assert_allclose(matrix.numpy(), np.c_[scale * rotation, shift], atol=1e-9)

        noisy = torch.tensor(source + rng.normal(0, 3, source.shape))
        matrix = similarity_transform(noisy, torch.tensor(target))
        parameters = torch.tensor([matrix[0, 0], matrix[1, 0], matrix[0, 2], matrix[1, 2]], requires_grad=True)
        a, b, tx, ty = parameters
        mapped = noisy @ torch.stack([torch.stack([a, -b]), torch.stack([b, a])]).T + torch.stack([tx, ty])
        (residual,) = torch.autograd.grad((mapped - torch.tensor(target)).square().sum(), parameters)
        self.assertLess(residual.abs().max().item(), 1e-8)

    def test_warp_matches_insightface_norm_crop(self):
        rng = np.random.default_rng(1)
        image = cv2.GaussianBlur(rng.uniform(0, 255, (300, 260, 3)).astype(np.float32), (0, 0), 2)
        tensor = torch.tensor(image).permute(2, 0, 1)[None] / 127.5 - 1
        for landmarks in (landmarks_at(70, 90, 1.3), landmarks_at(-10, 90, 1.3)):
            matrix = alignment_matrix(landmarks.double() + torch.tensor(rng.normal(0, 2, (5, 2))))
            expected = cv2.warpAffine(image, matrix.numpy(), (112, 112), borderValue=0.0)
            # A crop of the image with its offset must give the same face.
            box = latent_face_box(matrix, 300 // 8, 260 // 8)
            origin = (8 * box[2], 8 * box[0])
            actual = align_face(tensor[:, :, origin[1]:, origin[0]:], matrix, origin)
            actual = ((actual[0].permute(1, 2, 0) + 1) * 127.5).numpy()
            # cv2 interpolates with 5-bit fixed-point weights.
            self.assertLess(np.abs(actual - expected).max(), 3.0)
            self.assertLess(np.abs(actual - expected).mean(), 0.1)

    def test_large_faces_are_box_filtered_before_sampling(self):
        pixels = torch.zeros(1, 3, 900, 900)
        pixels[..., ::2, :] = 1  # Line pattern a point sampler would alias.
        face = align_face(pixels * 2 - 1, alignment_matrix(landmarks_at(0, 0, 8)), (0, 0))
        torch.testing.assert_close(face[..., 4:100, 4:100], torch.zeros(1, 3, 96, 96), atol=0.05, rtol=0)


class ArcFaceLoadTest(unittest.TestCase):
    def test_loads_arcface_torch_layout_and_depth(self):
        keys = set(IResNet([2, 2, 2, 2]).state_dict())
        for key in ("conv1.weight", "bn1.running_var", "prelu.weight", "layer1.0.downsample.0.weight",
                    "layer1.0.downsample.1.running_mean", "layer4.1.bn3.weight", "bn2.weight",
                    "fc.weight", "features.running_mean"):
            self.assertIn(key, keys)
        with tempfile.TemporaryDirectory() as directory:
            net = IResNet([2, 3, 1, 2])
            path = Path(directory, "backbone.pth")
            torch.save(net.state_dict(), path)
            loaded = load_arcface(str(path))
            self.assertEqual([len(loaded.get_submodule(f"layer{i}")) for i in range(1, 5)], [2, 3, 1, 2])
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            torch.save({"fc.weight": torch.zeros(4, 4)}, path)
            with self.assertRaises(ValueError):
                load_arcface(str(path))


def varint(value):
    value &= (1 << 64) - 1
    out = bytearray()
    while True:
        out.append((value & 0x7F) | (0x80 if value > 0x7F else 0))
        value >>= 7
        if not value:
            return bytes(out)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    value = value.encode() if isinstance(value, str) else value
    return varint(number << 3 | 2) + varint(len(value)) + value


def onnx_tensor(name, array):
    array = np.asarray(array)
    dtype = {np.dtype(np.float32): 1, np.dtype(np.int64): 7}[array.dtype]
    return b"".join(field(1, int(d)) for d in array.shape) + field(2, dtype) + field(8, name) + field(9, array.tobytes())


def onnx_node(op_type, inputs, outputs, **attributes):
    data = b"".join(field(1, name) for name in inputs) + b"".join(field(2, name) for name in outputs) + field(4, op_type)
    for name, value in attributes.items():
        if isinstance(value, float):
            attribute = field(1, name) + varint(2 << 3 | 5) + np.float32(value).tobytes() + field(20, 1)
        elif isinstance(value, int):
            attribute = field(1, name) + field(3, value) + field(20, 2)
        elif isinstance(value, np.ndarray):
            attribute = field(1, name) + field(5, onnx_tensor("", value)) + field(20, 4)
        else:  # Packed ints.
            attribute = field(1, name) + field(8, b"".join(varint(v) for v in value)) + field(20, 7)
        data += field(5, attribute)
    return data


class OnnxArcFaceTest(unittest.TestCase):
    def test_runs_insightface_style_onnx_graph_with_gradients(self):
        """Conv/BN/PReLU/Add/Flatten/Gemm as insightface exports them, plus a self-normalizing input."""
        rng = np.random.default_rng(0)
        weights = {
            "w": rng.normal(0, 0.3, (4, 3, 3, 3)).astype(np.float32), "b": rng.normal(0, 0.1, 4).astype(np.float32),
            "bn_s": rng.uniform(0.5, 1.5, 4).astype(np.float32), "bn_b": rng.normal(0, 0.1, 4).astype(np.float32),
            "bn_m": rng.normal(0, 0.1, 4).astype(np.float32), "bn_v": rng.uniform(0.5, 1.5, 4).astype(np.float32),
            "slope": rng.uniform(0, 0.3, (4, 1, 1)).astype(np.float32),
            "fc_w": rng.normal(0, 0.1, (5, 4 * 56 * 56)).astype(np.float32), "fc_b": rng.normal(0, 0.1, 5).astype(np.float32),
        }
        nodes = [
            onnx_node("Constant", [], ["mean"], value=np.full((1,), 127.5, np.float32)),
            onnx_node("Sub", ["data", "mean"], ["centered"]),
            onnx_node("Constant", [], ["scale"], value=np.full((1,), 1 / 127.5, np.float32)),
            onnx_node("Mul", ["centered", "scale"], ["x"]),
            onnx_node("Conv", ["x", "w", "b"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2], group=1),
            onnx_node("BatchNormalization", ["conv", "bn_s", "bn_b", "bn_m", "bn_v"], ["bn"], epsilon=1e-5),
            onnx_node("PRelu", ["bn", "slope"], ["act"]),
            onnx_node("Add", ["act", "conv"], ["sum"]),
            onnx_node("Flatten", ["sum"], ["flat"], axis=1),
            onnx_node("Gemm", ["flat", "fc_w", "fc_b"], ["out"], alpha=1.0, beta=1.0, transB=1),
        ]
        graph = b"".join(field(1, node) for node in nodes) + field(2, "tiny")
        graph += b"".join(field(5, onnx_tensor(name, value)) for name, value in weights.items())
        graph += field(11, field(1, "data")) + field(12, field(1, "out"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "arcface.onnx")
            path.write_bytes(field(1, 6) + field(7, graph))
            net = load_arcface(str(path))
        self.assertTrue(net.raw_pixels)  # The graph normalizes 0..255 pixels itself.
        w = {name: torch.from_numpy(value) for name, value in weights.items()}
        faces = (torch.rand(2, 3, 112, 112) * 2 - 1).requires_grad_(True)
        conv = F.conv2d(faces, w["w"], w["b"], stride=2, padding=1)
        bn = F.batch_norm(conv, w["bn_m"], w["bn_v"], w["bn_s"], w["bn_b"], eps=1e-5)
        expected = (torch.where(bn >= 0, bn, bn * w["slope"]) + conv).flatten(1) @ w["fc_w"].T + w["fc_b"]
        actual = net(faces)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        (gradient,) = torch.autograd.grad(actual.sum(), faces)
        (reference,) = torch.autograd.grad(expected.sum(), faces)
        torch.testing.assert_close(gradient, reference, atol=1e-5, rtol=1e-4)
        self.assertFalse(any(p.requires_grad for p in net.parameters()))


class ModelFileTest(unittest.TestCase):
    def test_local_files_hugging_face_links_and_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory, "backbone.pth")
            local.touch()
            self.assertEqual(resolve_model_file(str(local)), str(local))
        with patch("huggingface_hub.hf_hub_download", return_value="/cache/backbone.pth") as download:
            link = "https://huggingface.co/someone/arcface/resolve/main/r100/backbone.pth?download=true"
            self.assertEqual(resolve_model_file(link), "/cache/backbone.pth")
            download.assert_called_once_with("someone/arcface", "r100/backbone.pth", revision="main", repo_type=None)
            resolve_model_file("https://huggingface.co/datasets/someone/faces/blob/abc123/backbone.pth")
            download.assert_called_with("someone/faces", "backbone.pth", revision="abc123", repo_type="dataset")
        with patch("pooch.retrieve", return_value="/cache/yunet.onnx") as retrieve:
            self.assertEqual(resolve_model_file(YUNET_URL), "/cache/yunet.onnx")
            args, kwargs = retrieve.call_args
            self.assertEqual(args[:2], (YUNET_URL, KNOWN_HASHES[YUNET_URL]))
            self.assertTrue(kwargs["fname"].endswith("-face_detection_yunet_2023mar.onnx"))
        with patch("pooch.retrieve", return_value="/cache/yunet.onnx") as retrieve:
            resolve_model_file(YUNET_URL.replace("/raw/", "/blob/"))
            self.assertEqual(retrieve.call_args.args[:2], (YUNET_URL, KNOWN_HASHES[YUNET_URL]))
        with self.assertRaises(FileNotFoundError):
            resolve_model_file("missing/backbone.pth")
        self.assertEqual(TrainConfig.default_values().identity_reward_detector_path, YUNET_URL)
        arcface = TrainConfig.default_values().identity_reward_arcface_path
        with patch("huggingface_hub.hf_hub_download", return_value="/cache/arcface.onnx") as download:
            resolve_model_file(arcface)
        download.assert_called_once_with("FoivosPar/Arc2Face", "arcface.onnx", revision="main", repo_type=None)

    def test_model_fields_accept_download_links(self):
        self.assertIsNone(validate_path(YUNET_URL, allow_urls=True))
        self.assertIsNone(validate_path(TrainConfig.default_values().identity_reward_arcface_path, allow_urls=True))
        # Other path fields keep rejecting arbitrary links, and a link must name a file.
        self.assertIsNotNone(validate_path(YUNET_URL))
        self.assertIsNotNone(validate_path("https://github.com/opencv/opencv_zoo", allow_urls=True))


class StreamedLossTest(unittest.TestCase):
    def test_streamed_gradient_equals_direct_gradient(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            net = tiny_arcface(Path(directory, "backbone.pth"))
        weight = torch.randn(3, 4, 1, 1)

        def decode(latent):
            return F.interpolate(F.conv2d(latent, weight), scale_factor=8, mode="bilinear").tanh()

        parameter = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = torch.randn(2, 4, 8, 8)
        landmarks = torch.stack([landmarks_at(8, 4, 0.45), torch.full((5, 2), float("nan"))])
        losses, valid = identity_losses(parameter * 1.5, target, landmarks, decode, net, 512)
        self.assertEqual(valid.tolist(), [True, False])
        (streamed,) = torch.autograd.grad(losses.sum(), parameter)

        matrix = alignment_matrix(landmarks[0].double())
        y0, y1, x0, x1 = latent_face_box(matrix, 8, 8)
        prediction = parameter * 1.5

        def embedding(latent):
            return F.normalize(net(align_face(decode(latent).clamp(-1, 1), matrix, (8 * x0, 8 * y0))), dim=-1)

        direct = 1 - (embedding(prediction[:1, :, y0:y1, x0:x1]) * embedding(target[:1, :, y0:y1, x0:x1]).detach()).sum()
        (expected,) = torch.autograd.grad(direct, parameter)
        torch.testing.assert_close(losses[0], direct.detach())
        self.assertEqual(losses[1].item(), 0)
        self.assertGreater(streamed.abs().sum().item(), 0)
        torch.testing.assert_close(streamed, expected, atol=1e-6, rtol=1e-4)

    def test_faces_outside_crop_budget_are_skipped(self):
        landmarks = landmarks_at(8, 4, 0.45)[None]
        losses, valid = identity_losses(
            torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), landmarks,
            lambda latent: torch.zeros(1, 3, *(8 * s for s in latent.shape[-2:])), lambda faces: faces.flatten(1), 16,
        )
        self.assertFalse(valid.any())


class MOSSIdentityRewardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = tiny_model()
        self.config = train_config()
        self.config.identity_reward = True
        self.config.identity_reward_weight = 0.7
        self.config.identity_reward_interval = 1
        self.config.identity_reward_trigger = "zx, Ava Stone"
        self.config.identity_reward_arcface_path = str(Path(self.directory.name, "backbone.pth"))
        self.config.identity_reward_detector_path = "yunet.onnx"
        tiny_arcface(self.config.identity_reward_arcface_path)
        self.setup = create.get_model_setup_class(ModelType.MOSS_IMAGE, TrainingMethod.LORA)(
            torch.device("cpu"), torch.device("cpu"), False,
        )

    def batch(self, rows=3):
        return {
            "latent_image": torch.randn(rows, 32, 8, 8),
            "tokens": torch.tensor([[1, 2, 0, 0]] * rows),
            "tokens_mask": torch.tensor([[1, 1, 0, 0]] * rows),
            "text_encoder_hidden_state": torch.randn(rows, 4, 8),
            "loss_weight": torch.tensor([1.0, 0.5, 2.0][:rows]),
            "concept_type": [ConceptType.STANDARD] * rows,
            "prompt": ["photo of zx", "AVA STONE smiling", "zxy at night"][:rows],
            "face_landmarks": torch.stack([landmarks_at(8, 4, 0.45)] * rows),
        }

    def rows(self, batch, sigma=(0.1, 0.2, 0.3), dropped=(False, False, False), step=0):
        return self.setup._identity_reward_rows(
            batch, self.config, TrainProgress(global_step=step), torch.tensor(sigma), list(dropped),
        )

    def test_only_supervised_rows_naming_the_character_qualify(self):
        batch = self.batch()
        self.assertEqual(self.rows(batch), [0, 1])  # "zxy" is not the trigger "zx".
        self.config.identity_reward_trigger = ""
        self.assertEqual(self.rows(batch), [0, 1, 2])
        self.assertEqual(self.rows(batch, sigma=(0.1, 0.6, 0.3)), [0, 2])
        self.assertEqual(self.rows(batch, dropped=(True, False, False)), [1, 2])
        batch["concept_type"][1] = ConceptType.PRIOR_PREDICTION
        batch["face_landmarks"][2] = float("nan")
        self.assertEqual(self.rows(batch), [0])
        self.config.identity_reward_interval = 4
        self.assertIsNone(self.rows(batch, step=6))
        self.assertEqual(self.rows(batch, step=8), [0])
        with torch.no_grad():
            self.assertIsNone(self.rows(batch))
        with self.setup._dpo_stream_predict_context():
            self.assertIsNone(self.rows(batch))
        del batch["face_landmarks"]
        self.assertIsNone(self.rows(batch))

    def test_predict_skips_dropped_captions_and_validation(self):
        self.config.identity_reward_trigger = ""
        batch = self.batch()
        with patch.object(self.setup, "_get_timestep_discrete", return_value=torch.tensor([100, 200, 300])):
            self.assertEqual(self.setup.predict(self.model, batch, self.config, TrainProgress())["identity_rows"], [0, 1, 2])
            self.assertNotIn("identity_rows", self.setup.predict(self.model, batch, self.config, TrainProgress(), deterministic=True))
            self.config.text_encoder.dropout_probability = 1.0
            self.model.empty_text[4] = (torch.zeros(1, 4, 8), torch.tensor([[1, 0, 0, 0]]))
            self.assertNotIn("identity_rows", self.setup.predict(self.model, batch, self.config, TrainProgress()))

    def test_reward_backpropagates_the_x0_face_loss_into_the_lora(self):
        self.model.transformer.requires_grad_(False)
        from modules.module.LoRAModule import LoRAModuleWrapper
        self.model.transformer_lora = LoRAModuleWrapper(self.model.transformer, "transformer", self.config, ["layers."])
        self.model.transformer_lora.hook_to_module()
        parameters = list(self.model.transformer_lora.parameters())
        for parameter in parameters:
            parameter.data.normal_(0, 0.1)
        self.setup.setup_optimizations(self.model, self.config)
        self.model.vae.bn.running_mean.normal_()
        self.model.vae.bn.running_var.uniform_(0.5, 2)
        batch = self.batch()
        calls = []
        forward = self.model.transformer.forward

        def record(*args, **kwargs):
            result = forward(*args, **kwargs)
            calls.append((kwargs["hidden_states"], kwargs["timestep"], result.sample))
            return result

        with patch.object(self.setup, "_get_timestep_discrete", return_value=torch.tensor([100, 700, 300])), \
                patch.object(self.model.transformer, "forward", side_effect=record):
            output = self.setup.predict(self.model, batch, self.config, TrainProgress())
        self.assertEqual(output["identity_rows"], [0])  # Row 1 is too noisy, row 2 lacks the trigger.
        loss = self.setup.calculate_loss(self.model, batch, output, self.config)
        streamed = torch.autograd.grad(loss, parameters, retain_graph=True)

        # Reference: x0 = x_t - sigma * v in the model's latent space, decoded with a full graph.
        noisy, timestep, velocity = calls[0]
        sigma = (timestep / 1000).view(-1, 1, 1, 1)
        x0 = self.model.unpatchify_latents(self.model.unscale_latents(noisy - sigma * velocity))
        matrix = alignment_matrix(batch["face_landmarks"][0].double())
        y0, y1, x0_, x1 = latent_face_box(matrix, 8, 8)

        def embedding(latent):
            pixels = self.model.vae.decode(latent, return_dict=False)[0].clamp(-1, 1)
            return F.normalize(self.setup._arcface(align_face(pixels, matrix, (8 * x0_, 8 * y0))), dim=-1)

        face = 1 - (embedding(x0[:1, :, y0:y1, x0_:x1]) * embedding(batch["latent_image"][:1, :, y0:y1, x0_:x1])).sum()
        supervised = BaseFlux2Setup.calculate_loss(self.setup, self.model, batch, output, self.config)
        direct = supervised + 0.7 * face / 3
        expected = torch.autograd.grad(direct, parameters)
        torch.testing.assert_close(loss, direct)
        self.assertGreater(face.item(), 0)
        for actual, reference in zip(streamed, expected, strict=True):
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-4)
        tensorboard = SimpleNamespace(scalars={}, add_scalar=lambda name, value, step: tensorboard.scalars.update({name: value}))
        with patch("modules.modelSetup.BaseModelSetup.BaseModelSetup.report_to_tensorboard"):
            self.model.train_progress = TrainProgress()
            self.setup.report_to_tensorboard(self.model, self.config, None, tensorboard)
        self.assertAlmostEqual(tensorboard.scalars["loss/identity"], face.item(), places=5)
        self.assertEqual(tensorboard.scalars["identity/faces"], 1)


class StubDetector:
    """Encodes each training image's color in its landmarks; sample 0 of every concept has no face."""

    def __init__(self, path):
        self.path = path

    def __call__(self, pixels):
        red = int(pixels[0, 0, 0])
        return None if red == 40 else np.full((5, 2), red, dtype=np.float32)


class FaceLandmarkPairingTest(test_moss_pairing.MOSSPairingTest):
    """Landmarks stay with their own image's latent and caption through bucketing and both caches."""

    def configure(self, config):
        config.identity_reward = True
        config.identity_reward_detector_path = str(Path(self.temp.name, "yunet.onnx"))
        Path(config.identity_reward_detector_path).touch()
        detector = patch("modules.dataLoader.MOSSImageBaseDataLoader.FaceLandmarkDetector", StubDetector)
        detector.start()
        self.addCleanup(detector.stop)

    def test_latent_and_text_conditioning_belong_to_the_same_image(self):
        super().test_latent_and_text_conditioning_belong_to_the_same_image()
        # Enabling the reward re-caches images with landmarks, but keeps the text cache.
        caches = sorted(path.name for path in self.root.glob("cache-True-True/moss-layout-*/*"))
        self.assertEqual(caches, ["image-faces-yunet", "text"])

    def check_row(self, batch, row):
        identity = super().check_row(batch, row)
        red = test_moss_pairing.color(*identity)[0]
        landmarks = batch["face_landmarks"][row]
        if red == 40:
            self.assertTrue(landmarks.isnan().all())
        else:
            torch.testing.assert_close(landmarks, torch.full((5, 2), float(red)))
        return identity


if __name__ == "__main__":
    unittest.main()
