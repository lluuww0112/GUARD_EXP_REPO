from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from model.base.selection import (
    FrameSelectionResult,
    QWEN_VISION_FACTOR,
    _open_video_for_sampling,
    _resize_frame,
    uniform_sampling,
)


EmbeddingFunction = Callable[[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor]]
CANDIDATE_SAMPLING_STRATEGIES = {"global_uniform", "segment_uniform"}
AUXILIARY_REDUNDANCY_METRICS = {"null", "sfvd", "ssvd", "ofvd"}


def _resolve_query_text(
    *,
    query: str,
    query_file: str | None,
) -> str:
    query_text = query.strip()
    if not query_text and query_file:
        query_path = Path(query_file).expanduser()
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        query_text = query_path.read_text(encoding="utf-8").strip()
    if not query_text:
        raise ValueError("`query` or `query_file` must provide a non-empty query.")
    return query_text


def _resolve_candidate_count(
    *,
    total_frames: int,
    fps: float | None,
    duration_seconds: float | None,
    min_candidate_frames: int,
    max_candidate_frames: int,
) -> int:
    if min_candidate_frames <= 0:
        raise ValueError(
            f"`min_candidate_frames` must be positive, got {min_candidate_frames}."
        )
    if max_candidate_frames < min_candidate_frames:
        raise ValueError(
            "`max_candidate_frames` must be greater than or equal to "
            "`min_candidate_frames`."
        )

    duration = duration_seconds
    if duration is None and fps is not None and fps > 0 and total_frames > 0:
        duration = float(total_frames) / float(fps)

    if duration is None or duration <= 0:
        candidate_count = min_candidate_frames
    else:
        candidate_count = int(duration)
        candidate_count = max(min_candidate_frames, candidate_count)
        candidate_count = min(max_candidate_frames, candidate_count)

    return max(1, min(int(total_frames), int(candidate_count)))


def _decode_frames_at_indices(
    video_path: str,
    *,
    frame_indices: list[int],
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> FrameSelectionResult:
    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    frames: list[np.ndarray] = []
    decoded_indices: list[int] = []
    try:
        for frame_idx in frame_indices:
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
            frames.append(frame_rgb)
            decoded_indices.append(int(frame_idx))
    finally:
        cap.release()
        if transcoded_path is not None:
            import shutil

            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    if not frames:
        raise RuntimeError(f"No frames were decoded from video: {video_path}")

    video_np = np.stack(frames, axis=0)
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata={
            "video_path": video_path,
            "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
            "sampled_indices": decoded_indices,
            "num_frames": len(decoded_indices),
            "total_frames": total_frames,
            "fps": fps if fps > 0 else None,
            "frame_shape": list(video_np.shape[1:]),
            "ensure_qwen_compatibility": ensure_qwen_compatibility,
            "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
        },
    )


def _build_segment_uniform_indices(
    *,
    total_frames: int,
    num_candidates: int,
    segment_count: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    if segment_count <= 0:
        raise ValueError(f"`segment_count` must be positive, got {segment_count}.")

    effective_segment_count = min(segment_count, total_frames)
    segment_boundaries = np.linspace(
        0,
        total_frames,
        effective_segment_count + 1,
    ).round().astype(int)
    base_per_segment = max(1, num_candidates // effective_segment_count)
    remainder = max(0, num_candidates - (base_per_segment * effective_segment_count))

    indices: list[int] = []
    for segment_idx in range(effective_segment_count):
        start = int(segment_boundaries[segment_idx])
        end = int(segment_boundaries[segment_idx + 1])
        if end <= start:
            end = min(total_frames, start + 1)

        picks = base_per_segment + (1 if segment_idx < remainder else 0)
        local = np.linspace(
            start,
            max(start, end - 1),
            picks,
        ).round().astype(int)
        for frame_idx in local.tolist():
            clipped_idx = min(max(int(frame_idx), start), max(start, end - 1))
            indices.append(clipped_idx)

    deduped = sorted(set(indices))
    if len(deduped) >= num_candidates:
        return deduped[:num_candidates]

    if deduped:
        remaining_pool = sorted(set(range(total_frames)) - set(deduped))
        if remaining_pool:
            extra = np.linspace(
                0,
                len(remaining_pool) - 1,
                num_candidates - len(deduped),
            ).round().astype(int)
            deduped.extend(remaining_pool[int(i)] for i in extra.tolist())
            deduped = sorted(set(deduped))

    return deduped[:num_candidates]


def _sample_candidates(
    video_path: str,
    *,
    strategy: str,
    num_candidates: int,
    total_frames: int,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
    segment_count: int,
) -> FrameSelectionResult:
    if strategy == "global_uniform":
        return uniform_sampling(
            video_path=video_path,
            num_frames=num_candidates,
            max_side=max_side,
            ensure_qwen_compatibility=ensure_qwen_compatibility,
            qwen_factor=qwen_factor,
        )

    if strategy == "segment_uniform":
        candidate_indices = _build_segment_uniform_indices(
            total_frames=total_frames,
            num_candidates=num_candidates,
            segment_count=segment_count,
        )
        return _decode_frames_at_indices(
            video_path,
            frame_indices=candidate_indices,
            max_side=max_side,
            ensure_qwen_compatibility=ensure_qwen_compatibility,
            qwen_factor=qwen_factor,
        )

    available = ", ".join(sorted(CANDIDATE_SAMPLING_STRATEGIES))
    raise ValueError(
        f"Unsupported candidate_sampling_strategy: {strategy}. Available: [{available}]"
    )


def _compute_relevance_scores(
    frame_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_embedding.ndim > 1:
        query_embedding = query_embedding.squeeze(0)

    frame_embeddings = frame_embeddings.to(dtype=torch.float32)
    query_embedding = query_embedding.to(
        device=frame_embeddings.device,
        dtype=torch.float32,
    )

    similarity = torch.matmul(frame_embeddings, query_embedding)
    relevance = torch.sigmoid(similarity)
    return similarity, relevance


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


def _compute_pairwise_auxiliary_similarity(
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    *,
    metric: str,
) -> float:
    frame_a_np = frame_a.detach().cpu().numpy().astype(np.uint8)
    frame_b_np = frame_b.detach().cpu().numpy().astype(np.uint8)
    gray_a = _prepare_gray_frame(frame_a_np)
    gray_b = _prepare_gray_frame(frame_b_np)

    if metric == "ssvd":
        return _compute_ssim(gray_a, gray_b)

    if metric == "sfvd":
        flow = cv2.calcOpticalFlowFarneback(
            gray_a,
            gray_b,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = float(np.linalg.norm(flow, axis=2).mean())
        return float(1.0 / (1.0 + magnitude))

    raise ValueError(f"Unsupported auxiliary redundancy metric: {metric}")


def _build_auxiliary_similarity_matrix(
    frames: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor | None:
    normalized_metric = metric.strip().lower()
    if normalized_metric == "ofvd":
        normalized_metric = "sfvd"
    if normalized_metric == "null":
        return None
    if normalized_metric not in AUXILIARY_REDUNDANCY_METRICS:
        available = ", ".join(sorted(AUXILIARY_REDUNDANCY_METRICS))
        raise ValueError(
            f"Unsupported auxiliary redundancy metric: {metric}. Available: [{available}]"
        )

    frame_count = int(frames.shape[0])
    similarity_matrix = torch.zeros((frame_count, frame_count), dtype=torch.float32)
    for row in range(frame_count):
        similarity_matrix[row, row] = 1.0
        for col in range(row + 1, frame_count):
            similarity = _compute_pairwise_auxiliary_similarity(
                frames[row],
                frames[col],
                metric=normalized_metric,
            )
            similarity_matrix[row, col] = similarity
            similarity_matrix[col, row] = similarity
    return similarity_matrix


def _prepare_normalized_embeddings(frame_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(
        frame_embeddings.to(dtype=torch.float32),
        dim=-1,
    )


def _build_preselection(
    relevance_scores: torch.Tensor,
    *,
    top_k: int,
) -> list[int]:
    if top_k <= 0:
        raise ValueError(f"`preselection_top_k` must be positive, got {top_k}.")

    candidate_count = int(relevance_scores.shape[0])
    if top_k >= candidate_count:
        return list(range(candidate_count))

    top_indices = torch.topk(
        relevance_scores,
        k=top_k,
        largest=True,
        sorted=True,
    ).indices
    return [int(index) for index in top_indices.tolist()]


def _select_diverse_frames(
    preselected_indices: list[int],
    *,
    relevance_scores: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    candidate_timestamps: list[float | None],
    final_k: int,
    visual_similarity_threshold: float,
    temporal_suppression_seconds: float,
    auxiliary_similarity_matrix: torch.Tensor | None,
    auxiliary_redundancy_weight: float,
) -> tuple[list[int], list[str]]:
    if final_k <= 0:
        raise ValueError(f"`num_frames` must be positive, got {final_k}.")

    selected: list[int] = []
    relaxed_reasons: list[str] = []
    remaining = set(preselected_indices)

    def passes_constraints(candidate_idx: int, *, ignore_temporal: bool) -> bool:
        candidate_embedding = normalized_embeddings[candidate_idx]
        candidate_time = candidate_timestamps[candidate_idx]

        for selected_idx in selected:
            if not ignore_temporal and temporal_suppression_seconds > 0:
                selected_time = candidate_timestamps[selected_idx]
                if candidate_time is not None and selected_time is not None:
                    if abs(candidate_time - selected_time) < temporal_suppression_seconds:
                        return False

            cosine_similarity = torch.dot(
                candidate_embedding,
                normalized_embeddings[selected_idx],
            ).item()
            if cosine_similarity >= visual_similarity_threshold:
                return False
        return True

    def pick_next(*, ignore_temporal: bool) -> bool:
        if not remaining or len(selected) >= final_k:
            return False

        best_idx: int | None = None
        best_score = -float("inf")
        for candidate_idx in sorted(remaining):
            if not passes_constraints(candidate_idx, ignore_temporal=ignore_temporal):
                continue

            score = float(relevance_scores[candidate_idx].item())
            if auxiliary_similarity_matrix is not None and selected:
                max_similarity = max(
                    float(auxiliary_similarity_matrix[candidate_idx, selected_idx].item())
                    for selected_idx in selected
                )
                score -= auxiliary_redundancy_weight * max_similarity

            if score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is None:
            return False

        selected.append(best_idx)
        remaining.remove(best_idx)
        return True

    while len(selected) < final_k and pick_next(ignore_temporal=False):
        pass

    if len(selected) < final_k:
        relaxed_reasons.append("temporal_suppression_relaxed")
        while len(selected) < final_k and pick_next(ignore_temporal=True):
            pass

    if len(selected) < final_k:
        relaxed_reasons.append("visual_suppression_relaxed")
        remaining_ordered = sorted(
            remaining,
            key=lambda index: float(relevance_scores[index].item()),
            reverse=True,
        )
        for candidate_idx in remaining_ordered:
            if len(selected) >= final_k:
                break
            selected.append(candidate_idx)
            remaining.remove(candidate_idx)

    selected = sorted(selected)
    return selected, relaxed_reasons


def _compute_candidate_timestamps(
    *,
    candidate_indices: list[int],
    fps: float | None,
) -> list[float | None]:
    if fps is None or fps <= 0:
        return [None for _ in candidate_indices]
    return [float(index) / float(fps) for index in candidate_indices]


def _build_soft_selection_metadata(
    normalized_embeddings: torch.Tensor,
    relevance_scores: torch.Tensor,
    *,
    selected_indices: list[int],
    enable_soft_weighting: bool,
) -> dict[str, Any]:
    if not enable_soft_weighting:
        return {
            "enabled": False,
        }

    selected_scores = relevance_scores[selected_indices]
    weight_total = float(selected_scores.sum().item())
    if weight_total <= 0.0:
        weights = torch.full_like(selected_scores, 1.0 / max(len(selected_indices), 1))
    else:
        weights = selected_scores / selected_scores.sum()

    weighted_feature = torch.sum(
        normalized_embeddings[selected_indices] * weights.unsqueeze(-1),
        dim=0,
    )
    return {
        "enabled": True,
        "weights": [float(weight.item()) for weight in weights],
        "weighted_feature": weighted_feature.detach().cpu().tolist(),
    }


def siglip_adaptive_frame_sampling(
    video_path: str,
    *,
    num_frames: int = 8,
    embed_fn: EmbeddingFunction | None = None,
    query: str = "",
    query_file: str | None = None,
    min_candidate_frames: int = 48,
    max_candidate_frames: int = 128,
    candidate_sampling_strategy: str = "global_uniform",
    segment_count: int = 16,
    preselection_top_k: int = 32,
    visual_similarity_threshold: float = 0.9,
    temporal_suppression_seconds: float = 2.0,
    auxiliary_redundancy_metric: str = "null",
    auxiliary_redundancy_weight: float = 0.15,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
    enable_soft_weighting: bool = False,
    **_: Any,
) -> FrameSelectionResult:
    if embed_fn is None:
        raise ValueError(
            "`embed_fn` is required for query-conditioned SigLIPAFS frame selection."
        )
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    candidate_sampling_strategy = candidate_sampling_strategy.strip().lower()
    auxiliary_redundancy_metric = auxiliary_redundancy_metric.strip().lower()
    if auxiliary_redundancy_metric == "ofvd":
        auxiliary_redundancy_metric = "sfvd"

    query_text = _resolve_query_text(
        query=query,
        query_file=query_file,
    )

    cap, total_frames, fps_value, transcoded_path = _open_video_for_sampling(video_path)
    cap.release()
    if transcoded_path is not None:
        import shutil

        shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    duration = (
        float(total_frames) / float(fps_value)
        if fps_value is not None and fps_value > 0
        else None
    )

    num_candidates = _resolve_candidate_count(
        total_frames=total_frames,
        fps=fps_value,
        duration_seconds=duration,
        min_candidate_frames=min_candidate_frames,
        max_candidate_frames=max_candidate_frames,
    )

    candidates = _sample_candidates(
        video_path,
        strategy=candidate_sampling_strategy,
        num_candidates=num_candidates,
        total_frames=total_frames,
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
        segment_count=segment_count,
    )

    frame_embeddings, query_embedding = embed_fn(candidates.frames, query_text)
    similarity_scores, relevance_scores = _compute_relevance_scores(
        frame_embeddings,
        query_embedding,
    )
    normalized_embeddings = _prepare_normalized_embeddings(frame_embeddings)

    preselected_indices = _build_preselection(
        relevance_scores,
        top_k=min(preselection_top_k, int(relevance_scores.shape[0])),
    )
    auxiliary_similarity_matrix = _build_auxiliary_similarity_matrix(
        candidates.frames[preselected_indices],
        metric=auxiliary_redundancy_metric,
    )
    expanded_auxiliary_similarity_matrix: torch.Tensor | None = None
    if auxiliary_similarity_matrix is not None:
        candidate_count = int(candidates.frames.shape[0])
        expanded_auxiliary_similarity_matrix = torch.zeros(
            (candidate_count, candidate_count),
            dtype=auxiliary_similarity_matrix.dtype,
        )
        for row, source_row in enumerate(preselected_indices):
            for col, source_col in enumerate(preselected_indices):
                expanded_auxiliary_similarity_matrix[source_row, source_col] = (
                    auxiliary_similarity_matrix[row, col]
                )

    candidate_indices = list(candidates.metadata.get("sampled_indices", []))
    candidate_timestamps = _compute_candidate_timestamps(
        candidate_indices=candidate_indices,
        fps=fps_value,
    )
    final_indices, relaxed_reasons = _select_diverse_frames(
        preselected_indices,
        relevance_scores=relevance_scores,
        normalized_embeddings=normalized_embeddings,
        candidate_timestamps=candidate_timestamps,
        final_k=min(num_frames, len(preselected_indices)),
        visual_similarity_threshold=visual_similarity_threshold,
        temporal_suppression_seconds=temporal_suppression_seconds,
        auxiliary_similarity_matrix=expanded_auxiliary_similarity_matrix,
        auxiliary_redundancy_weight=auxiliary_redundancy_weight,
    )

    soft_selection = _build_soft_selection_metadata(
        normalized_embeddings,
        relevance_scores,
        selected_indices=final_indices,
        enable_soft_weighting=enable_soft_weighting,
    )

    selected_candidate_indices = [
        candidate_indices[index]
        for index in final_indices
    ]
    selected_timestamps = [
        candidate_timestamps[index]
        for index in final_indices
    ]

    return FrameSelectionResult(
        frames=candidates.frames[final_indices],
        metadata={
            **candidates.metadata,
            "sampling_method": "siglip_afs",
            "siglip_afs_variant": "dynamic_siglip_relevance",
            "num_candidates": num_candidates,
            "candidate_sampling_strategy": candidate_sampling_strategy,
            "segment_count": (
                segment_count if candidate_sampling_strategy == "segment_uniform" else None
            ),
            "preselection_top_k": min(preselection_top_k, len(preselected_indices)),
            "selected_from_candidates": final_indices,
            "sampled_indices": selected_candidate_indices,
            "selected_timestamps": selected_timestamps,
            "relevance_scores": [
                float(relevance_scores[index].item())
                for index in final_indices
            ],
            "candidate_relevance_scores": [
                float(score.item()) for score in relevance_scores
            ],
            "candidate_similarity_scores": [
                float(score.item()) for score in similarity_scores
            ],
            "query_text": query_text,
            "redundancy_reduction": {
                "temporal_suppression_seconds": temporal_suppression_seconds,
                "visual_similarity_threshold": visual_similarity_threshold,
                "auxiliary_redundancy_metric": auxiliary_redundancy_metric,
                "auxiliary_redundancy_weight": auxiliary_redundancy_weight,
                "relaxed_reasons": relaxed_reasons,
            },
            "soft_selection": soft_selection,
        },
    )
