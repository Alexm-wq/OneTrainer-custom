"""Face identity reward: ArcFace on aligned faces decoded from x0 predictions.

The IResNet matches insightface's arcface_torch layout, so its backbone.pth files load as they are;
insightface's ONNX exports run through a small torch ONNX interpreter, so gradients reach the face.
Landmarks come from OpenCV's YuNet detector, whose five points share ArcFace's template order.
"""
import hashlib
import os
import re
import threading
from urllib.parse import urlparse

from modules.util.onnx_torch import OnnxModule

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import numpy as np

ARCFACE_SIZE = 112
ARCFACE_TEMPLATE = (
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041),
)
# Latent pixels decoded around the aligned face, so the decoder sees some context.
LATENT_MARGIN = 2
# Smaller faces give ArcFace too few pixels to judge identity.
MIN_EYE_DISTANCE = 20.0
# YuNet for OpenCV 4.x, pinned to an opencv_zoo commit.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/48b10fa505aeb51fbd127fc60a5c660040e823b7/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# ArcFace R100 trained on WebFace42M, as Arc2Face distributes it.
ARCFACE_URL = "https://huggingface.co/FoivosPar/Arc2Face/resolve/main/arcface.onnx"
KNOWN_HASHES = {YUNET_URL: "sha256:8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"}
HF_FILE_URL = re.compile(
    r"https://huggingface\.co/(?:(datasets|spaces)/)?([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/([^?#]+)(?:[?#].*)?"
)


def resolve_model_file(location: str) -> str:
    """A local file, a Hugging Face file link or another download URL, as a local path.

    Hugging Face files land in its usual cache; other downloads in external/models/identity.
    """
    if os.path.isfile(location):
        return location
    match = HF_FILE_URL.fullmatch(location)
    if match:
        import huggingface_hub
        repo_type, repo_id, revision, filename = match.groups()
        return huggingface_hub.hf_hub_download(
            repo_id, filename, revision=revision, repo_type=repo_type[:-1] if repo_type else None,
        )
    if location.startswith(("https://", "http://")):
        import pooch
        parsed = urlparse(location)
        if parsed.netloc == "github.com" and "/blob/" in parsed.path:
            # A link copied from GitHub's file view; /raw/ serves the file itself.
            location = location.replace("/blob/", "/raw/", 1)
        # The URL hash keeps different files with the same name apart.
        name = hashlib.sha256(location.encode()).hexdigest()[:8] + "-" + os.path.basename(urlparse(location).path)
        return pooch.retrieve(
            location, KNOWN_HASHES.get(location), fname=name,
            path=os.path.join("external", "models", "identity"), progressbar=True,
        )
    raise FileNotFoundError(f"Model file not found: {location!r}")


class IBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes, eps=1e-5),
            )

    def forward(self, x):
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        return out + (x if self.downsample is None else self.downsample(x))


class IResNet(nn.Module):
    def __init__(self, layers, num_features=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        inplanes = 64
        for index, (planes, count) in enumerate(zip((64, 128, 256, 512), layers, strict=True)):
            blocks = [IBasicBlock(inplanes, planes, 2)] + [IBasicBlock(planes, planes, 1) for _ in range(count - 1)]
            setattr(self, f"layer{index + 1}", nn.Sequential(*blocks))
            inplanes = planes
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)

    def forward(self, x):
        """RGB faces in [-1, 1], aligned to the 112x112 ArcFace template."""
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return self.features(self.fc(torch.flatten(x, 1)))


class OnnxArcFace(nn.Module):
    """An insightface ArcFace ONNX model (e.g. antelopev2, buffalo_l, Arc2Face), taking IResNet's input."""

    def __init__(self, path: str):
        super().__init__()
        self.net = OnnxModule(path)
        # As insightface: graphs that start by normalizing themselves (MXNet exports) get 0..255 pixels.
        first = [node[0] for node in self.net.nodes[:8]]
        self.raw_pixels = "Sub" in first and "Mul" in first

    def forward(self, x):
        return self.net((x + 1) * 127.5 if self.raw_pixels else x)


def load_arcface(path: str) -> nn.Module:
    """Load an insightface ArcFace model: an arcface_torch backbone (.pth/.safetensors) or an ONNX export."""
    if path.endswith(".onnx"):
        return OnnxArcFace(path).eval().requires_grad_(False)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    layers = [
        1 + max((int(key.split(".")[1]) for key in state if key.startswith(f"layer{index}.")), default=-1)
        for index in range(1, 5)
    ]
    if 0 in layers or "fc.weight" not in state:
        raise ValueError(f"{path} is not an insightface arcface_torch IResNet backbone")
    net = IResNet(layers, state["fc.weight"].shape[0])
    missing, unexpected = net.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise ValueError(f"{path} does not match the IResNet layout: missing {missing}, unexpected {unexpected}")
    return net.eval().requires_grad_(False)


def similarity_transform(source: Tensor, target: Tensor) -> Tensor:
    """Least-squares similarity (Umeyama) mapping source points onto target points, as a 2x3 matrix."""
    source, target = source.double(), target.double()
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, s, vh = torch.linalg.svd(covariance)
    d = torch.ones(2, dtype=torch.float64)
    if torch.det(covariance) < 0:
        d[1] = -1
    rotation = u @ torch.diag(d) @ vh
    scale = (s * d).sum() / source_centered.square().sum(1).mean()
    translation = target_mean - scale * rotation @ source_mean
    return torch.cat([scale * rotation, translation[:, None]], dim=1)


def alignment_matrix(landmarks: Tensor) -> Tensor:
    """2x3 matrix from image pixels to the 112x112 ArcFace crop, as insightface's norm_crop uses it."""
    return similarity_transform(landmarks, torch.tensor(ARCFACE_TEMPLATE, dtype=torch.float64))


def latent_face_box(matrix: Tensor, latent_height: int, latent_width: int, scale: int = 8):
    """Latent rows/columns (y0, y1, x0, x1) whose decode covers the aligned crop, or None if it misses the image."""
    linear, offset = matrix[:, :2], matrix[:, 2]
    corners = torch.tensor([[0, 0], [ARCFACE_SIZE, 0], [0, ARCFACE_SIZE], [ARCFACE_SIZE, ARCFACE_SIZE]], dtype=torch.float64)
    image = (corners - offset) @ torch.linalg.inv(linear).T
    x0 = max(int(image[:, 0].min() // scale) - LATENT_MARGIN, 0)
    y0 = max(int(image[:, 1].min() // scale) - LATENT_MARGIN, 0)
    x1 = min(int(image[:, 0].max() // scale) + 1 + LATENT_MARGIN, latent_width)
    y1 = min(int(image[:, 1].max() // scale) + 1 + LATENT_MARGIN, latent_height)
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def align_face(pixels: Tensor, matrix: Tensor, origin: tuple[int, int]) -> Tensor:
    """Warp an RGB crop in [-1, 1], whose top-left image pixel is origin=(x, y), to the ArcFace template.

    Bilinear like cv2.warpAffine with black borders. Faces much larger than the template are box
    filtered first, so the gradient reaches every pixel instead of a sparse sampling grid.
    """
    linear, offset = matrix[:, :2], matrix[:, 2]
    factor = max(int(1.0 / float(torch.linalg.det(linear).abs().sqrt()) + 1e-6), 1)
    if factor > 1:
        pixels = F.avg_pool2d(pixels, factor, factor)
    height, width = pixels.shape[-2:]
    inverse = torch.linalg.inv(linear)
    grid = torch.stack(torch.meshgrid(
        torch.arange(ARCFACE_SIZE, dtype=torch.float64), torch.arange(ARCFACE_SIZE, dtype=torch.float64), indexing="xy",
    ), dim=-1)
    image = (grid - offset) @ inverse.T - torch.tensor(origin, dtype=torch.float64)
    # Pooled pixel j averages input pixels j*factor .. j*factor+factor-1.
    image = (image - (factor - 1) / 2) / factor
    normalized = torch.stack([(2 * image[..., 0] + 1) / width - 1, (2 * image[..., 1] + 1) / height - 1], dim=-1)
    normalized = normalized[None].to(device=pixels.device, dtype=pixels.dtype).expand(pixels.shape[0], -1, -1, -1)
    # Zero padding of the shifted image is black (-1), as insightface pads with 0 before normalizing.
    return F.grid_sample(pixels + 1, normalized, mode="bilinear", padding_mode="zeros", align_corners=False) - 1


class FaceLandmarkDetector:
    """YuNet five-point landmarks of the largest face, one OpenCV detector per thread."""

    def __init__(self, model_path: str, score_threshold: float = 0.8):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.local = threading.local()

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        """RGB uint8 HxWx3 image to a (5, 2) float array in pixels, or None."""
        import cv2
        detector = getattr(self.local, "detector", None)
        if detector is None:
            detector = self.local.detector = cv2.FaceDetectorYN.create(
                self.model_path, "", (image.shape[1], image.shape[0]), self.score_threshold,
            )
        detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = detector.detect(np.ascontiguousarray(image[..., ::-1]))
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda row: row[2] * row[3])
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)
        if np.linalg.norm(landmarks[1] - landmarks[0]) < MIN_EYE_DISTANCE:
            return None
        return landmarks


def identity_losses(
        prediction: Tensor, target: Tensor, landmarks: Tensor, decode, embed, max_crop: int,
) -> tuple[Tensor, Tensor]:
    """Per-row ArcFace loss 1 - cos(predicted face, target face) with gradients for the prediction.

    prediction and target are unscaled VAE latents. Each face is decoded and embedded on its own, and
    its decoder/ArcFace graph is freed before the next one: the result carries the face gradient as a
    surrogate on prediction, so only the latents stay attached to the caller's graph.
    Returns the losses and a mask of rows that had a usable face.
    """
    losses = []
    valid = torch.zeros(prediction.shape[0], dtype=torch.bool)
    for row in range(prediction.shape[0]):
        losses.append(prediction.new_zeros((), dtype=torch.float32))
        points = landmarks[row].detach().cpu().double()
        if not torch.isfinite(points).all():
            continue
        matrix = alignment_matrix(points)
        box = latent_face_box(matrix, *prediction.shape[-2:])
        if box is None or 8 * max(box[1] - box[0], box[3] - box[2]) > max_crop:
            continue
        y0, y1, x0, x1 = box
        origin = (8 * x0, 8 * y0)

        def face_embedding(latent, matrix=matrix, origin=origin):
            pixels = decode(latent).float().clamp(-1, 1)
            return F.normalize(embed(align_face(pixels, matrix, origin)).float(), dim=-1)

        with torch.no_grad():
            reference = face_embedding(target[row:row + 1, :, y0:y1, x0:x1].float())
        crop = prediction[row:row + 1, :, y0:y1, x0:x1]
        leaf = crop.detach().float().requires_grad_(crop.requires_grad)
        with torch.enable_grad():
            loss = 1 - (face_embedding(leaf) * reference).sum()
            gradient = torch.autograd.grad(loss, leaf)[0] if leaf.requires_grad else None
        if gradient is not None:
            surrogate = (crop.float() * gradient).sum()
            loss = surrogate - surrogate.detach() + loss.detach()
        losses[row] = loss
        valid[row] = True
    return torch.stack(losses), valid
