from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch


QWEN_VISION_FACTOR = 28
QWEN_MIN_PIXELS = 56 * 56
QWEN_MAX_PIXELS = 14 * 14 * 4 * 1280


@dataclass(slots=True)
class FrameSelectionResult:
    frames: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatchSelectionResult:
    selected_indices: torch.Tensor | None = None
    selected_features: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _qwen_smart_resize(
    height: int,
    width: int,
    *,
    factor: int = QWEN_VISION_FACTOR,
    min_pixels: int = QWEN_MIN_PIXELS,
    max_pixels: int = QWEN_MAX_PIXELS,
) -> tuple[int, int]:
    if factor <= 0:
        return height, width
    if min(height, width) <= 0:
        raise ValueError(f"Invalid frame size: height={height}, width={width}")

    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}"
        )

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    resized_height = max(factor, resized_height)
    resized_width = max(factor, resized_width)

    pixel_count = resized_height * resized_width
    if pixel_count > max_pixels:
        beta = float(np.sqrt((height * width) / max_pixels))
        resized_height = max(factor, int(np.floor(height / beta / factor)) * factor)
        resized_width = max(factor, int(np.floor(width / beta / factor)) * factor)
    elif pixel_count < min_pixels:
        beta = float(np.sqrt(min_pixels / (height * width)))
        resized_height = max(factor, int(np.ceil(height * beta / factor)) * factor)
        resized_width = max(factor, int(np.ceil(width * beta / factor)) * factor)

    return resized_height, resized_width


def _resize_frame(
    frame_rgb: np.ndarray,
    *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> np.ndarray:
    resized_frame = frame_rgb
    if max_side is not None:
        height, width = resized_frame.shape[:2]
        scale = min(max_side / max(height, width), 1.0)
        if scale < 1.0:
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            resized_frame = cv2.resize(
                resized_frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

    if ensure_qwen_compatibility:
        height, width = resized_frame.shape[:2]
        resized_height, resized_width = _qwen_smart_resize(
            height,
            width,
            factor=qwen_factor,
        )
        if (resized_height, resized_width) != (height, width):
            resized_frame = cv2.resize(
                resized_frame,
                (resized_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

    return resized_frame


def uniform_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Failed to read total frame count.")

    if num_frames <= 0:
        cap.release()
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")

    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()

    target_set = set(indices)
    frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(
                frame_rgb,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            frames.append(frame_rgb)
            sampled_indices.append(frame_idx)

        frame_idx += 1

    cap.release()

    if len(frames) != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} sampled frames, but got {len(frames)}."
        )

    base_height, base_width = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)

    video_np = np.stack(normalized_frames, axis=0)
    metadata = {
        "video_path": video_path,
        "sampled_indices": sampled_indices,
        "num_frames": len(normalized_frames),
        "total_frames": total_frames,
        "fps": fps if fps > 0 else None,
        "frame_shape": list(video_np.shape[1:]),
        "ensure_qwen_compatibility": ensure_qwen_compatibility,
        "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
    }
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata=metadata,
    )


def identity_patch_selection(
    video_features: torch.Tensor,
    **_: Any,
) -> PatchSelectionResult:
    selected_indices = torch.arange(
        video_features.shape[0],
        device=video_features.device,
        dtype=torch.long,
    )
    return PatchSelectionResult(selected_indices=selected_indices)
