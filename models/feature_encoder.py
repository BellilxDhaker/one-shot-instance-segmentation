"""Step 1 - spatial visual features from a frozen vision foundation model.

Model choice
------------
`facebook/dinov2-base` (DINOv2 ViT-B/14) through `transformers`.

* spatial quality: patch tokens are trained without a text/image alignment loss,
  so they stay spatially consistent and support patch-level correspondence;
* availability: the checkpoint is served by the Hugging Face hub and loads with
  `transformers` alone - no external research repository is cloned or imported;
* Colab compatibility: ViT-B fits comfortably in a T4 GPU at 518x518 input;
* compute: ~86M parameters, one forward pass per image, no fine-tuning;
* suitability for one-shot matching: the same encoder is applied to the
  reference and the target image, so the two feature grids live in one metric
  space and cosine similarity is directly meaningful.

Only the last transformer block is used (`layer=-1`), the standard choice for
DINOv2 dense prediction.

Interface
---------
    encoder = FeatureEncoder(device="cuda")
    features = encoder.extract(image)          # -> FeatureMap
    features.tokens                            # (C, h, w) L2-normalised
    features.grid_positions()                  # (h, w, 2) normalised xy

The image is resized (never cropped) to a multiple of the patch size, so the
token grid covers the whole image and its geometry stays known exactly. That is
what lets the reference mask be pooled onto the same grid without guessing.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_ENCODER_ID = "facebook/dinov2-base"
DEFAULT_MAX_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageInput = Union[str, "os.PathLike[str]", np.ndarray, "Any"]

_ENCODER_CACHE: Dict[Tuple[Any, ...], "FeatureEncoder"] = {}


def to_pil_image(image: ImageInput):
    """Accept a path, an RGB/RGBA array or a PIL image and return a PIL image."""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 1.0) * 255.0
            array = array.astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L").convert("RGB")
        return Image.fromarray(array[..., :3], mode="RGB")
    return image.convert("RGB")


@dataclass
class FeatureMap:
    """Dense features for one image, kept on their patch grid.

    Attributes:
        tokens: (C, h, w) patch tokens, channel-last features folded into a map
            and L2-normalised when `normalised` is True.
        image_size: (H, W) size of the source image in pixels.
        input_size: (H_in, W_in) size the encoder actually received.
        patch_size: side of one square token in input pixels.
        model_id: checkpoint identifier, recorded for provenance.
        layer: transformer block the tokens come from.
        normalised: whether the tokens are L2-normalised.
    """

    tokens: torch.Tensor
    image_size: Tuple[int, int]
    input_size: Tuple[int, int]
    patch_size: int
    model_id: str
    layer: int = -1
    normalised: bool = True

    @property
    def grid_size(self) -> Tuple[int, int]:
        return int(self.tokens.shape[1]), int(self.tokens.shape[2])

    @property
    def num_patches(self) -> int:
        return int(self.tokens.shape[1] * self.tokens.shape[2])

    @property
    def channels(self) -> int:
        return int(self.tokens.shape[0])

    def to(self, device: Union[str, torch.device]) -> "FeatureMap":
        return FeatureMap(
            tokens=self.tokens.to(device),
            image_size=self.image_size,
            input_size=self.input_size,
            patch_size=self.patch_size,
            model_id=self.model_id,
            layer=self.layer,
            normalised=self.normalised,
        )

    def tokens_hwc(self) -> torch.Tensor:
        """(h, w, C) view, the layout used for patch-wise comparisons."""
        return self.tokens.permute(1, 2, 0).contiguous()

    def grid_positions(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """(h, w, 2) centre of every patch as normalised image coordinates in [0, 1]."""
        grid_h, grid_w = self.grid_size
        ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) / grid_h
        xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) / grid_w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        positions = torch.stack([grid_x, grid_y], dim=-1)
        return positions.to(device or self.tokens.device)

    def pixel_scale(self) -> Tuple[float, float]:
        """Pixels per patch, as (height, width)."""
        return (self.input_size[0] / self.grid_size[0], self.input_size[1] / self.grid_size[1])

    def upsample(self, size: Optional[Tuple[int, int]] = None, mode: str = "bilinear") -> torch.Tensor:
        """Resample tokens to pixel resolution, returning (H, W, C)."""
        target = tuple(size) if size is not None else self.image_size
        resized = F.interpolate(
            self.tokens.unsqueeze(0), size=target, mode=mode, align_corners=False
        )
        return resized.squeeze(0).permute(1, 2, 0).contiguous()

    def describe(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "image_size": list(self.image_size),
            "input_size": list(self.input_size),
            "grid_size": list(self.grid_size),
            "num_patches": self.num_patches,
            "channels": self.channels,
            "patch_size": self.patch_size,
            "normalised": self.normalised,
        }


class FeatureEncoder:
    """Frozen dense feature extractor.

    Args:
        model_id: Hugging Face identifier of the vision backbone.
        device: torch device; `None` picks CUDA when available.
        dtype: dtype used for inference.
        max_size: side length (in patches' input pixels) the longest image side is
            resized to when `image_size` is None.
        image_size: fixed square input; `None` keeps the aspect ratio and rounds
            both sides to a multiple of the patch size.
        layer: transformer block to read (`-1` = last).
        normalise: L2-normalise tokens so cosine similarity is a dot product.
        cache_enabled: keep features of recent images in memory.
        cache_size: number of entries kept.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_ENCODER_ID,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        max_size: int = DEFAULT_MAX_SIZE,
        image_size: Optional[int] = None,
        layer: int = -1,
        normalise: bool = True,
        cache_enabled: bool = True,
        cache_size: int = 8,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device) if device is not None else _default_device()
        self.dtype = dtype
        self.max_size = int(max_size)
        self.image_size = image_size
        self.layer = int(layer)
        self.normalise = bool(normalise)
        self.cache_enabled = bool(cache_enabled)
        self.cache_size = int(cache_size)
        self.cache_dir = cache_dir
        self._model: Optional[torch.nn.Module] = None
        self._cache: "Dict[Tuple[Any, ...], FeatureMap]" = {}
        self._cache_order: List[Tuple[Any, ...]] = []

    def load(self) -> "FeatureEncoder":
        """Download (or read from `cache_dir`) and load the frozen backbone."""
        if self._model is not None:
            return self
        from transformers import AutoModel

        kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir
        model = AutoModel.from_pretrained(self.model_id, **kwargs)
        model.eval().to(device=self.device, dtype=self.dtype)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        return self

    def unload(self) -> None:
        self._model = None
        self._cache.clear()
        self._cache_order.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def patch_size(self) -> int:
        self.load()
        return int(self._model.config.patch_size)

    def describe(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "layer": self.layer,
            "max_size": self.max_size,
            "image_size": self.image_size,
            "patch_size": self.patch_size if self.is_loaded else None,
            "loaded": self.is_loaded,
        }

    def extract(self, image: ImageInput) -> FeatureMap:
        """Return the dense feature map of one image."""
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[ImageInput]) -> List[FeatureMap]:
        """Extract several images; identical inputs hit the in-memory cache.

        Images that ended up with the same input size are stacked into one
        forward pass, so a batch of equally shaped images stays efficient while
        mixed aspect ratios remain correct.
        """
        pil_images = [to_pil_image(image) for image in images]
        results: List[Optional[FeatureMap]] = [None] * len(pil_images)

        pending: List[int] = []
        keys: Dict[int, Tuple[Any, ...]] = {}
        for index, pil_image in enumerate(pil_images):
            key = self._cache_key(pil_image)
            keys[index] = key
            cached = self._cache.get(key) if self.cache_enabled else None
            if cached is not None:
                results[index] = cached
            else:
                pending.append(index)

        if pending:
            self.load()
            groups: Dict[Tuple[int, int], List[int]] = {}
            for index in pending:
                groups.setdefault(self._target_size(pil_images[index].height, pil_images[index].width),
                                  []).append(index)
            for input_size, indices in groups.items():
                batch = torch.stack([self._to_tensor(pil_images[index]) for index in indices])
                with torch.inference_mode():
                    output = self._model(
                        pixel_values=batch.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=self.layer != -1,
                    )
                hidden = output.last_hidden_state if self.layer == -1 else output.hidden_states[self.layer]
                for slot, index in enumerate(indices):
                    features = self._to_feature_map(hidden[slot], pil_images[index], input_size)
                    results[index] = features
                    if self.cache_enabled:
                        self._store(keys[index], features)

        return [result for result in results if result is not None]

    def _to_feature_map(self, hidden: torch.Tensor, image, input_size: Tuple[int, int]) -> FeatureMap:
        grid_h = int(input_size[0]) // self.patch_size
        grid_w = int(input_size[1]) // self.patch_size
        row = hidden.reshape(-1, hidden.shape[-1]).float()
        num_patches = grid_h * grid_w
        if row.shape[0] < num_patches:
            raise RuntimeError(
                f"{self.model_id} returned {row.shape[0]} tokens, fewer than the {num_patches} "
                f"patches implied by an input of {input_size} with patch size {self.patch_size}"
            )
        patch_tokens = row[-num_patches:]
        tokens = patch_tokens.reshape(grid_h, grid_w, -1).permute(2, 0, 1).contiguous()
        if self.normalise:
            tokens = F.normalize(tokens, dim=0)
        return FeatureMap(
            tokens=tokens,
            image_size=(image.height, image.width),
            input_size=(int(input_size[0]), int(input_size[1])),
            patch_size=self.patch_size,
            model_id=self.model_id,
            layer=self.layer,
            normalised=self.normalise,
        )

    def _to_tensor(self, image) -> torch.Tensor:
        return self._preprocess(image)

    def _preprocess(self, image) -> torch.Tensor:
        import torchvision.transforms.functional as TF

        width, height = image.width, image.height
        target = self._target_size(height, width)
        resized = TF.resize(image, list(target), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    def _target_size(self, height: int, width: int) -> Tuple[int, int]:
        patch = self.patch_size
        if self.image_size:
            if int(self.image_size) % patch:
                raise ValueError(f"image_size={self.image_size} must be a multiple of patch_size={patch}")
            return int(self.image_size), int(self.image_size)
        scale = self.max_size / max(height, width)
        new_height = max(patch, int(round(height * scale / patch)) * patch)
        new_width = max(patch, int(round(width * scale / patch)) * patch)
        return new_height, new_width

    def _cache_key(self, image) -> Tuple[Any, ...]:
        digest = hashlib.sha1(np.asarray(image).tobytes()).hexdigest()
        return (self.model_id, self.layer, self.max_size, self.image_size, digest)

    def _store(self, key: Tuple[Any, ...], features: FeatureMap) -> None:
        self._cache[key] = features
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_size:
            self._cache.pop(self._cache_order.pop(0), None)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_default_encoder(device: Optional[Union[str, torch.device]] = None, **kwargs: Any) -> FeatureEncoder:
    """Process-wide encoder, reused across calls so the weights load once."""
    device = torch.device(device) if device is not None else _default_device()
    key = (str(device), DEFAULT_ENCODER_ID, kwargs.get("image_size"), kwargs.get("max_size"))
    encoder = _ENCODER_CACHE.get(key)
    if encoder is None:
        encoder = FeatureEncoder(device=device, **kwargs)
        _ENCODER_CACHE[key] = encoder
    return encoder