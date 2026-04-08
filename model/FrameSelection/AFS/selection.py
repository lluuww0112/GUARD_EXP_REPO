from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from model.base.selection import (
    FrameSelectionResult,
    _open_video_for_sampling,
    _resize_frame,
)


def _normalize_probabilities(scores: list[float]) -> np.ndarray:
    probabilities = np.asarray(scores, dtype=np.float64)
    probabilities = np.clip(probabilities, a_min=0.0, a_max=None)

    total = float(probabilities.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(scores), 1.0 / max(len(scores), 1), dtype=np.float64)
    return probabilities / total


def _build_uniform_candidate_indices(
    total_frames: int,
    *,
    num_frames: int,
    candidate_frames: int | None,
    candidate_multiplier: int,
) -> np.ndarray:
    if total_frames <= 0:
        return np.empty(0, dtype=int)

    effective_candidate_frames = candidate_frames
    if effective_candidate_frames is None:
        effective_candidate_frames = max(num_frames, num_frames * candidate_multiplier)

    effective_candidate_frames = max(num_frames, int(effective_candidate_frames))
    effective_candidate_frames = min(total_frames, effective_candidate_frames)
    if effective_candidate_frames == total_frames:
        return np.arange(total_frames, dtype=int)

    return np.linspace(
        0,
        total_frames - 1,
        effective_candidate_frames,
    ).round().astype(int)


def _decode_frames_at_indices(
    cap: cv2.VideoCapture,
    frame_indices: np.ndarray,
    *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> tuple[list[np.ndarray], list[int]]:
    decoded_frames: list[np.ndarray] = []
    decoded_indices: list[int] = []

    for frame_idx in frame_indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = _resize_frame(
            frame_rgb,
            max_side=max_side,
            ensure_qwen_compatibility=ensure_qwen_compatibility,
            qwen_factor=qwen_factor,
        )
        decoded_frames.append(frame_rgb)
        decoded_indices.append(int(frame_idx))

    return decoded_frames, decoded_indices


def _prepare_gray_frame(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_a = cv2.GaussianBlur(frame_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(frame_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(frame_a * frame_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(frame_b * frame_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(frame_a * frame_b, (11, 11), 1.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    denominator = np.where(denominator == 0.0, 1e-12, denominator)

    ssim_map = numerator / denominator
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _frame_dissimilarity(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    *,
    metric: str,
    flow_kwargs: dict[str, Any],
) -> float:
    gray_a = _prepare_gray_frame(frame_a)
    gray_b = _prepare_gray_frame(frame_b)

    if metric == "ssvd":
        return 1.0 - _compute_ssim(gray_a, gray_b)

    if metric == "ofvd":
        flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, **flow_kwargs)
        magnitude = np.linalg.norm(flow, axis=2)
        return float(magnitude.mean())

    raise ValueError(
        f"Unsupported AFS metric: {metric}. Available: ['ofvd', 'ssvd']"
    )


def adaptive_frame_sampling(
    video_path: str,
    num_frames: int = 8,
    metric: str = "ofvd",
    candidate_frames: int | None = None,
    candidate_multiplier: int = 4,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = 28,
    random_seed: int | None = 42,
    farneback_pyr_scale: float = 0.5,
    farneback_levels: int = 3,
    farneback_winsize: int = 15,
    farneback_iterations: int = 3,
    farneback_poly_n: int = 5,
    farneback_poly_sigma: float = 1.2,
    farneback_flags: int = 0,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if candidate_multiplier <= 0:
        raise ValueError(
            f"`candidate_multiplier` must be positive, got {candidate_multiplier}."
        )

    metric = metric.strip().lower()
    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)

    candidate_indices = _build_uniform_candidate_indices(
        total_frames,
        num_frames=num_frames,
        candidate_frames=candidate_frames,
        candidate_multiplier=candidate_multiplier,
    )

    frames: list[np.ndarray] = []
    decoded_candidate_indices: list[int] = []
    try:
        frames, decoded_candidate_indices = _decode_frames_at_indices(
            cap,
            candidate_indices,
            max_side=max_side,
            ensure_qwen_compatibility=ensure_qwen_compatibility,
            qwen_factor=qwen_factor,
        )
    finally:
        cap.release()
        if transcoded_path is not None:
            import shutil

            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    if not frames:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    if num_frames >= len(frames):
        sampled_indices = list(range(len(frames)))
        video_np = np.stack(frames, axis=0)
        metadata = {
            "video_path": video_path,
            "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
            "sampled_indices": decoded_candidate_indices,
            "num_frames": len(sampled_indices),
            "total_frames": total_frames,
            "fps": fps if fps > 0 else None,
            "frame_shape": list(video_np.shape[1:]),
            "ensure_qwen_compatibility": ensure_qwen_compatibility,
            "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
            "sampling_method": "afs",
            "afs_metric": metric,
            "candidate_frames": len(decoded_candidate_indices),
            "candidate_indices": decoded_candidate_indices,
            "fvi_scores": [1.0 for _ in sampled_indices],
            "fvi_probabilities": [1.0 / len(sampled_indices) for _ in sampled_indices],
        }
        return FrameSelectionResult(
            frames=torch.from_numpy(video_np),
            metadata=metadata,
        )

    flow_kwargs = {
        "pyr_scale": farneback_pyr_scale,
        "levels": farneback_levels,
        "winsize": farneback_winsize,
        "iterations": farneback_iterations,
        "poly_n": farneback_poly_n,
        "poly_sigma": farneback_poly_sigma,
        "flags": farneback_flags,
    }

    fvi_scores: list[float] = []
    for idx in range(len(frames)):
        next_idx = min(idx + 1, len(frames) - 1)
        dissimilarity = _frame_dissimilarity(
            frames[idx],
            frames[next_idx],
            metric=metric,
            flow_kwargs=flow_kwargs,
        )
        fvi_scores.append(max(dissimilarity, 0.0))

    fvi_probabilities = _normalize_probabilities(fvi_scores)
    rng = np.random.default_rng(random_seed)
    sampled_indices = np.sort(
        rng.choice(
            len(frames),
            size=num_frames,
            replace=False,
            p=fvi_probabilities,
        )
    ).astype(int)
    sampled_video_indices = [decoded_candidate_indices[int(idx)] for idx in sampled_indices]

    sampled_frames = [frames[int(idx)] for idx in sampled_indices]
    base_height, base_width = sampled_frames[0].shape[:2]
    normalized_frames = []
    for frame in sampled_frames:
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
        "sampled_indices": sampled_video_indices,
        "num_frames": len(normalized_frames),
        "total_frames": total_frames,
        "fps": fps if fps > 0 else None,
        "frame_shape": list(video_np.shape[1:]),
        "ensure_qwen_compatibility": ensure_qwen_compatibility,
        "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
        "sampling_method": "afs",
        "afs_metric": metric,
        "random_seed": random_seed,
        "candidate_frames": len(decoded_candidate_indices),
        "candidate_indices": decoded_candidate_indices,
        "fvi_scores": [float(score) for score in fvi_scores],
        "fvi_probabilities": fvi_probabilities.tolist(),
    }
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata=metadata,
    )
