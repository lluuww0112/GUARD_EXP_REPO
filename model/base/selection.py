from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
import torch


QWEN_VISION_FACTOR = 28
QWEN_MIN_PIXELS = 56 * 56
QWEN_MAX_PIXELS = 14 * 14 * 4 * 1280


def _inspect_video_capture(video_path: str) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    return cap, total_frames, fps


def _transcode_video_for_opencv(video_path: str) -> Path | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="guard_opencv_decode_"))
    output_path = temp_dir / f"{Path(video_path).stem}_opencv_h264.mp4"
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    return output_path if output_path.exists() else None


def _open_video_for_sampling(
    video_path: str,
) -> tuple[cv2.VideoCapture, int, float, Path | None]:
    cap, total_frames, fps = _inspect_video_capture(video_path)
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, None

    cap.release()
    transcoded_path = _transcode_video_for_opencv(video_path)
    if transcoded_path is None:
        raise RuntimeError(
            "Failed to decode video with OpenCV. "
            "This often happens in Colab when the source video uses AV1. "
            "Install ffmpeg with AV1 software decoding support or transcode the video "
            "to H.264 before running frame selection."
        )

    cap, total_frames, fps = _inspect_video_capture(str(transcoded_path))
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, transcoded_path

    cap.release()
    shutil.rmtree(transcoded_path.parent, ignore_errors=True)
    raise RuntimeError(
        "Failed to decode video even after ffmpeg transcoding fallback. "
        f"source={video_path}"
    )


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


def _decode_target_frames(
    cap: cv2.VideoCapture,
    indices: list[int],
    *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> tuple[list[np.ndarray], list[int]]:
    frames: list[np.ndarray] = []
    sampled_indices: list[int] = []

    for frame_idx in indices:
        seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        if not seek_ok:
            continue
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(
            _resize_frame(
                frame_rgb,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
        )
        sampled_indices.append(int(frame_idx))

    return frames, sampled_indices


def _decode_target_frames_sequentially(
    cap: cv2.VideoCapture,
    indices: list[int],
    *,
    num_frames: int,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> tuple[list[np.ndarray], list[int], int, bool, bool]:
    target_set = set(indices)
    frames: list[np.ndarray] = []
    prefix_frames: list[np.ndarray] = []
    sampled_indices: list[int] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        should_keep_prefix = frame_idx < num_frames
        should_sample = frame_idx in target_set
        if should_keep_prefix or should_sample:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(
                frame_rgb,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            if should_keep_prefix:
                # Keep the prefix so we can fall back to "all decoded frames"
                # when the actual video is shorter than the requested sample count.
                prefix_frames.append(frame_rgb)
            if should_sample:
                frames.append(frame_rgb)
                sampled_indices.append(frame_idx)

        frame_idx += 1

    actual_total_frames = frame_idx
    used_short_video_fallback = False
    used_index_mismatch_fallback = False
    if len(frames) != len(indices):
        recoverable_frame_count = min(actual_total_frames, num_frames)
        if recoverable_frame_count <= 0 or len(prefix_frames) != recoverable_frame_count:
            raise RuntimeError(
                f"Expected {len(indices)} sampled frames, but got {len(frames)}."
            )
        frames = prefix_frames
        sampled_indices = list(range(recoverable_frame_count))
        used_short_video_fallback = actual_total_frames < num_frames
        used_index_mismatch_fallback = not used_short_video_fallback

    return (
        frames,
        sampled_indices,
        actual_total_frames,
        used_short_video_fallback,
        used_index_mismatch_fallback,
    )


def uniform_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")

    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)

    use_all_frames = total_frames <= num_frames
    if use_all_frames:
        indices = list(range(total_frames))
    elif num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()

    try:
        frames, sampled_indices = _decode_target_frames(
            cap,
            indices,
            max_side=max_side,
            ensure_qwen_compatibility=ensure_qwen_compatibility,
            qwen_factor=qwen_factor,
        )
        actual_total_frames = total_frames
        used_short_video_fallback = False
        used_index_mismatch_fallback = False
        sampling_strategy = "indexed_seek"
        if len(frames) != len(indices):
            cap.release()
            fallback_video_path = str(transcoded_path) if transcoded_path is not None else video_path
            cap, total_frames, fps = _inspect_video_capture(fallback_video_path)
            if not cap.isOpened() or total_frames <= 0:
                raise RuntimeError(
                    "Failed to reopen video for sequential fallback. "
                    f"source={video_path}"
                )
            (
                frames,
                sampled_indices,
                actual_total_frames,
                used_short_video_fallback,
                used_index_mismatch_fallback,
            ) = _decode_target_frames_sequentially(
                cap,
                indices,
                num_frames=num_frames,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            sampling_strategy = "sequential_fallback"
    finally:
        cap.release()
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

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
        "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
        "sampled_indices": sampled_indices,
        "num_frames": len(normalized_frames),
        "total_frames": total_frames,
        "decoded_total_frames": actual_total_frames,
        "fps": fps if fps > 0 else None,
        "frame_shape": list(video_np.shape[1:]),
        "sampling_strategy": sampling_strategy,
        "ensure_qwen_compatibility": ensure_qwen_compatibility,
        "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
        "used_short_video_fallback": used_short_video_fallback,
        "used_index_mismatch_fallback": used_index_mismatch_fallback,
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
