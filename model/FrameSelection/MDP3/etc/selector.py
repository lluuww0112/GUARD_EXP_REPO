from __future__ import annotations

from dataclasses import dataclass
import shutil
from typing import Any

import cv2
import numpy as np
import torch

from model.base.selection import (
    FrameSelectionResult,
    _open_video_for_sampling,
    _resize_frame,
)


@dataclass(slots=True)
class _DecodedFrame:
    index: int
    frame: np.ndarray


def _decode_video_frames(
    video_path: str,
    *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> tuple[list[_DecodedFrame], dict[str, Any]]:
    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    decoded_frames: list[_DecodedFrame] = []

    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_frame(
                frame_rgb,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            decoded_frames.append(_DecodedFrame(index=frame_idx, frame=frame_rgb))
            frame_idx += 1
    finally:
        cap.release()
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    metadata = {
        "video_path": video_path,
        "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
        "total_frames": total_frames,
        "fps": fps if fps > 0 else None,
        "ensure_qwen_compatibility": ensure_qwen_compatibility,
        "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
    }
    return decoded_frames, metadata


def _extract_frame_features(frame: np.ndarray, bins_per_channel: int) -> np.ndarray:
    frame_f32 = frame.astype(np.float32) / 255.0
    mean_rgb = frame_f32.mean(axis=(0, 1))
    std_rgb = frame_f32.std(axis=(0, 1))

    hist_features = []
    for channel in range(frame_f32.shape[2]):
        hist, _ = np.histogram(
            frame_f32[..., channel],
            bins=bins_per_channel,
            range=(0.0, 1.0),
        )
        hist = hist.astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        hist_features.append(hist)

    return np.concatenate([mean_rgb, std_rgb, *hist_features], axis=0)


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return features / norms


def _build_quality_scores(features: np.ndarray) -> np.ndarray:
    center = features.mean(axis=0, keepdims=True)
    diversity_from_center = np.linalg.norm(features - center, axis=1)
    motion_proxy = np.zeros(features.shape[0], dtype=np.float32)
    if features.shape[0] > 1:
        diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)
        motion_proxy[1:] = diffs
        motion_proxy[0] = diffs[0]

    quality = 0.5 * diversity_from_center + 0.5 * motion_proxy
    if np.allclose(quality.max(), quality.min()):
        return np.ones_like(quality, dtype=np.float32)
    quality = (quality - quality.min()) / (quality.max() - quality.min())
    return quality.astype(np.float32)


def _build_similarity_matrix(features: np.ndarray) -> np.ndarray:
    normalized = _l2_normalize(features)
    similarity = normalized @ normalized.T
    return np.clip(similarity, -1.0, 1.0).astype(np.float32)


def _select_indices_with_dynamic_programming(
    *,
    quality_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    num_frames: int,
    temporal_window: int,
    temporal_reward: float,
    diversity_penalty: float,
    min_frame_distance: int,
) -> list[int]:
    total_frames = quality_scores.shape[0]
    dp = np.full((num_frames, total_frames), -np.inf, dtype=np.float32)
    backtrack = np.full((num_frames, total_frames), -1, dtype=np.int32)

    dp[0] = quality_scores
    for step in range(1, num_frames):
        for current in range(total_frames):
            prev_candidates = np.arange(0, current, dtype=np.int32)
            if prev_candidates.size == 0:
                continue

            if min_frame_distance > 0:
                prev_candidates = prev_candidates[
                    current - prev_candidates >= min_frame_distance
                ]
                if prev_candidates.size == 0:
                    continue

            temporal_gap = current - prev_candidates
            temporal_bonus = temporal_reward * np.exp(
                -np.maximum(temporal_gap - 1, 0) / max(float(temporal_window), 1.0)
            )
            candidate_scores = (
                dp[step - 1, prev_candidates]
                + quality_scores[current]
                + temporal_bonus
                - diversity_penalty * similarity_matrix[prev_candidates, current]
            )
            best_offset = int(np.argmax(candidate_scores))
            best_prev = int(prev_candidates[best_offset])
            dp[step, current] = candidate_scores[best_offset]
            backtrack[step, current] = best_prev

    last_index = int(np.argmax(dp[num_frames - 1]))
    if not np.isfinite(dp[num_frames - 1, last_index]):
        raise RuntimeError(
            "Failed to find a valid MDP3 frame subset. Try reducing `num_frames` or `min_frame_distance`."
        )

    selected = [last_index]
    for step in range(num_frames - 1, 0, -1):
        last_index = int(backtrack[step, last_index])
        if last_index < 0:
            raise RuntimeError("MDP3 backtracking failed due to an invalid predecessor state.")
        selected.append(last_index)

    selected.reverse()
    return selected


def mdp3_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 224,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = 28,
    bins_per_channel: int = 8,
    temporal_window: int = 12,
    temporal_reward: float = 0.15,
    diversity_penalty: float = 0.35,
    min_frame_distance: int = 1,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if bins_per_channel <= 0:
        raise ValueError(
            f"`bins_per_channel` must be positive, got {bins_per_channel}."
        )
    if temporal_window <= 0:
        raise ValueError(f"`temporal_window` must be positive, got {temporal_window}.")
    if min_frame_distance < 0:
        raise ValueError(
            f"`min_frame_distance` must be non-negative, got {min_frame_distance}."
        )
    if min_frame_distance <= 1:
        minimum_required_frames = num_frames
    else:
        minimum_required_frames = 1 + (num_frames - 1) * min_frame_distance

    decoded_frames, base_metadata = _decode_video_frames(
        video_path,
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )
    if not decoded_frames:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")

    total_decoded = len(decoded_frames)
    if num_frames > total_decoded:
        raise ValueError(
            f"Requested {num_frames} frames, but only {total_decoded} frames are available."
        )
    if minimum_required_frames > total_decoded:
        raise ValueError(
            "The current `min_frame_distance` is too large for the requested frame budget: "
            f"need at least {minimum_required_frames} decoded frames, but only {total_decoded} are available."
        )

    feature_matrix = np.stack(
        [
            _extract_frame_features(item.frame, bins_per_channel=bins_per_channel)
            for item in decoded_frames
        ],
        axis=0,
    )
    quality_scores = _build_quality_scores(feature_matrix)
    similarity_matrix = _build_similarity_matrix(feature_matrix)

    selected_positions = _select_indices_with_dynamic_programming(
        quality_scores=quality_scores,
        similarity_matrix=similarity_matrix,
        num_frames=num_frames,
        temporal_window=temporal_window,
        temporal_reward=temporal_reward,
        diversity_penalty=diversity_penalty,
        min_frame_distance=min_frame_distance,
    )
    selected_items = [decoded_frames[position] for position in selected_positions]

    base_height, base_width = selected_items[0].frame.shape[:2]
    normalized_frames = []
    for item in selected_items:
        frame = item.frame
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)

    video_np = np.stack(normalized_frames, axis=0)
    sampled_indices = [item.index for item in selected_items]
    metadata = {
        **base_metadata,
        "sampled_indices": sampled_indices,
        "num_frames": len(sampled_indices),
        "frame_shape": list(video_np.shape[1:]),
        "selector": "mdp3_sampling",
        "selection_strategy": {
            "bins_per_channel": bins_per_channel,
            "temporal_window": temporal_window,
            "temporal_reward": temporal_reward,
            "diversity_penalty": diversity_penalty,
            "min_frame_distance": min_frame_distance,
        },
        "quality_scores": quality_scores[sampled_indices].tolist(),
    }
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata=metadata,
    )
