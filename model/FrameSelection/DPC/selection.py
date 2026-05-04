from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from model.base.selection import FrameSelectionResult, QWEN_VISION_FACTOR, uniform_sampling
from model.PatchSelection.DenseDPS.selection_v1 import (
    _aggregate_query_scores,
    _load_queries,
    _prepare_frame_arrays,
    _resolve_clip_dtype,
    _resolve_device_key,
)
from model.PatchSelection.DenseDPS.selection_v2 import (
    _load_maskclip_components_v2,
    _load_text_embeddings_v2,
)


@dataclass(slots=True)
class DPCSegmentationResult:
    requested_k: int
    effective_k: int
    centers: list[int]
    segments: list[dict[str, int]]


def _resolve_selector_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_k_set(k_set: Sequence[int] | None, *, threshold: int) -> list[int]:
    if threshold <= 0:
        raise ValueError(f"`threshold` must be positive, got {threshold}.")
    values = list(k_set) if k_set is not None else [threshold, threshold * 2, threshold * 3]
    resolved: list[int] = []
    for value in values:
        k = int(value)
        if k <= 0:
            raise ValueError(f"All `k_set` values must be positive, got {k}.")
        if k not in resolved:
            resolved.append(k)
    if not resolved:
        raise ValueError("`k_set` must include at least one positive k.")
    if resolved[0] != threshold:
        raise ValueError(
            "The first DPC clustering k must match `threshold`: "
            f"first_k={resolved[0]}, threshold={threshold}."
        )
    return resolved


def _compute_dpc_centers(
    embeddings: torch.Tensor,
    *,
    k: int,
    knn_k: int,
) -> list[int]:
    center_ranking = _rank_dpc_centers(embeddings, kneighbor_count=knn_k)
    return center_ranking[: min(int(k), int(embeddings.shape[0]))]


def _rank_dpc_centers(
    embeddings: torch.Tensor,
    *,
    kneighbor_count: int,
) -> list[int]:
    frame_count = int(embeddings.shape[0])
    if frame_count <= 0:
        raise ValueError("DPC-KNN requires at least one frame embedding.")
    if frame_count == 1:
        return [0]

    normalized = F.normalize(embeddings.to(dtype=torch.float32), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    distance = (1.0 - similarity).clamp_min(0.0)
    distance.fill_diagonal_(float("inf"))

    neighbor_count = min(max(1, int(kneighbor_count)), frame_count - 1)
    knn_distances = torch.topk(
        distance,
        k=neighbor_count,
        dim=1,
        largest=False,
        sorted=False,
    ).values
    rho = torch.exp(-(knn_distances**2)).sum(dim=1)

    finite_distance = distance.masked_fill(torch.isinf(distance), 0.0)
    higher_density = rho.unsqueeze(1) < rho.unsqueeze(0)
    distance_to_higher = distance.masked_fill(~higher_density, float("inf"))
    delta = distance_to_higher.min(dim=1).values
    max_finite_distance = finite_distance.max(dim=1).values
    delta = torch.where(torch.isinf(delta), max_finite_distance, delta)

    center_scores = rho * delta
    ranking_bias = torch.linspace(
        0.0,
        -1e-6,
        steps=frame_count,
        device=center_scores.device,
        dtype=center_scores.dtype,
    )
    ranked_centers = torch.argsort(
        center_scores + ranking_bias,
        descending=True,
    )
    return [int(index) for index in ranked_centers.detach().cpu().tolist()]


def _build_temporal_segments(
    centers: Sequence[int],
    *,
    frame_count: int,
    requested_k: int,
) -> DPCSegmentationResult:
    if frame_count <= 0:
        raise ValueError("Cannot build temporal segments without frames.")
    temporal_centers = sorted({int(center) for center in centers})
    if not temporal_centers:
        raise ValueError("DPC-KNN did not produce any centers.")

    starts = [0]
    for left, right in zip(temporal_centers, temporal_centers[1:]):
        starts.append(((left + right) // 2) + 1)

    segments: list[dict[str, int]] = []
    for segment_index, center in enumerate(temporal_centers):
        start = starts[segment_index]
        end = (
            starts[segment_index + 1] - 1
            if segment_index + 1 < len(starts)
            else frame_count - 1
        )
        segments.append(
            {
                "segment_index": segment_index,
                "start_pool_index": int(start),
                "end_pool_index": int(end),
                "center_pool_index": int(center),
            }
        )

    return DPCSegmentationResult(
        requested_k=int(requested_k),
        effective_k=len(temporal_centers),
        centers=temporal_centers,
        segments=segments,
    )


def _compute_clip_global_embeddings(
    frames: torch.Tensor,
    *,
    clip_model_name: str,
    clip_dtype: str | torch.dtype | None,
    clip_do_center_crop: bool | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    device_key = _resolve_device_key(device)
    image_processor, _, vision_model, _ = _load_maskclip_components_v2(
        clip_model_name,
        device_key,
        clip_dtype_key,
        clip_do_center_crop,
    )
    frame_arrays = _prepare_frame_arrays(frames)
    embedding_chunks: list[torch.Tensor] = []

    with torch.inference_mode():
        for start in range(0, len(frame_arrays), batch_size):
            batch_frames = frame_arrays[start : start + batch_size]
            pixel_values = image_processor(
                images=batch_frames,
                return_tensors="pt",
            )["pixel_values"]
            if resolved_clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=resolved_clip_dtype,
                    non_blocking=True,
                )

            if hasattr(vision_model, "_forward_to_final_block_input") and hasattr(
                vision_model,
                "_forward_global_latent_from_final_block_input",
            ):
                hidden_states = vision_model._forward_to_final_block_input(pixel_values)
                image_latents = vision_model._forward_global_latent_from_final_block_input(
                    hidden_states
                )
            else:
                _, image_latents = vision_model(pixel_values)
            embedding_chunks.append(F.normalize(image_latents, dim=-1).to(dtype=torch.float32))

    if not embedding_chunks:
        raise ValueError("No frames were provided for CLIP global embedding.")
    return torch.cat(embedding_chunks, dim=0)


def _compute_clip_global_embeddings_and_score_maps(
    frames: torch.Tensor,
    *,
    query_file: str,
    clip_model_name: str,
    clip_dtype: str | torch.dtype | None,
    clip_do_center_crop: bool | None,
    aggregation: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], tuple[str, ...], str]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    device_key = _resolve_device_key(device)
    queries = _load_queries(query_file)
    image_processor, _, vision_model, _ = _load_maskclip_components_v2(
        clip_model_name,
        device_key,
        clip_dtype_key,
        clip_do_center_crop,
    )
    text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        device_key,
        clip_dtype_key,
        clip_do_center_crop,
        queries,
    ).to(device=device)
    text_embeddings_t = text_embeddings.transpose(0, 1).contiguous()

    frame_arrays = _prepare_frame_arrays(frames)
    embedding_chunks: list[torch.Tensor] = []
    dense_score_chunks: list[torch.Tensor] = []
    frame_score_chunks: list[torch.Tensor] = []
    clip_grid: tuple[int, int] | None = None
    patch_size = int(vision_model.config.patch_size)

    with torch.inference_mode():
        for start in range(0, len(frame_arrays), batch_size):
            batch_frames = frame_arrays[start : start + batch_size]
            pixel_values = image_processor(
                images=batch_frames,
                return_tensors="pt",
            )["pixel_values"]
            if resolved_clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=resolved_clip_dtype,
                    non_blocking=True,
                )

            patch_embeddings, image_latents = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            image_latents = F.normalize(image_latents, dim=-1)

            patch_scores = patch_embeddings @ text_embeddings_t
            patch_scores = _aggregate_query_scores(
                patch_scores,
                aggregation=aggregation,
            )
            frame_scores = image_latents @ text_embeddings_t
            frame_scores = _aggregate_query_scores(
                frame_scores,
                aggregation=aggregation,
            )

            clip_h = int(pixel_values.shape[-2] // patch_size)
            clip_w = int(pixel_values.shape[-1] // patch_size)
            if clip_h * clip_w != int(patch_scores.shape[1]):
                raise ValueError(
                    "Failed to infer CLIP patch grid from processor output: "
                    f"expected {clip_h}x{clip_w}={clip_h * clip_w} patches, "
                    f"but got {int(patch_scores.shape[1])}."
                )
            if clip_grid is None:
                clip_grid = (clip_h, clip_w)
            elif clip_grid != (clip_h, clip_w):
                raise ValueError(
                    "CLIP processor returned inconsistent patch grids across batches: "
                    f"{clip_grid} vs {(clip_h, clip_w)}."
                )

            embedding_chunks.append(image_latents.to(dtype=torch.float32))
            dense_score_chunks.append(
                patch_scores.view(patch_scores.shape[0], clip_h, clip_w).to(
                    dtype=torch.float32
                )
            )
            frame_score_chunks.append(frame_scores.to(dtype=torch.float32))

    if clip_grid is None:
        raise ValueError("No frames were provided for CLIP score caching.")
    return (
        torch.cat(embedding_chunks, dim=0),
        torch.cat(dense_score_chunks, dim=0),
        torch.cat(frame_score_chunks, dim=0),
        clip_grid,
        queries,
        clip_dtype_key,
    )


def _compute_query_scores(
    frame_embeddings: torch.Tensor,
    *,
    query_file: str,
    clip_model_name: str,
    clip_dtype: str | torch.dtype | None,
    clip_do_center_crop: bool | None,
    aggregation: str,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[str, ...], str]:
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    device_key = _resolve_device_key(device)
    queries = _load_queries(query_file)
    text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        device_key,
        clip_dtype_key,
        clip_do_center_crop,
        queries,
    ).to(device=frame_embeddings.device, dtype=frame_embeddings.dtype)
    scores = frame_embeddings @ text_embeddings.transpose(0, 1).contiguous()
    return _aggregate_query_scores(scores, aggregation=aggregation), queries, clip_dtype_key


def _top_relevant_centers(
    centers: Sequence[int],
    relevance_scores: torch.Tensor,
    *,
    tau: float,
    threshold: int,
) -> tuple[list[int], list[int], int]:
    center_list = [int(center) for center in centers]
    high_relevance = [
        center for center in center_list if float(relevance_scores[center].item()) >= float(tau)
    ]
    high_relevance_sorted = sorted(
        high_relevance,
        key=lambda index: float(relevance_scores[index].item()),
        reverse=True,
    )
    return high_relevance_sorted[:threshold], high_relevance, len(high_relevance)


def _fallback_centers(
    centers: Sequence[int],
    relevance_scores: torch.Tensor,
    *,
    tau: float,
    threshold: int,
) -> tuple[list[int], list[int], int]:
    selected, high_relevance, high_count = _top_relevant_centers(
        centers,
        relevance_scores,
        tau=tau,
        threshold=threshold,
    )
    selected_set = set(selected)
    if len(selected) < threshold:
        remaining = [
            int(center)
            for center in centers
            if int(center) not in selected_set
        ]
        remaining_sorted = sorted(
            remaining,
            key=lambda index: float(relevance_scores[index].item()),
            reverse=True,
        )
        for center in remaining_sorted:
            if len(selected) >= threshold:
                break
            selected.append(center)
            selected_set.add(center)
    return selected[:threshold], high_relevance, high_count


def _pad_to_threshold(
    selected: Sequence[int],
    *,
    centers: Sequence[int],
    relevance_scores: torch.Tensor,
    threshold: int,
) -> tuple[list[int], bool]:
    padded = [int(index) for index in selected]
    if len(padded) >= threshold:
        return padded[:threshold], False

    if padded:
        source = sorted(
            padded,
            key=lambda index: float(relevance_scores[index].item()),
            reverse=True,
        )
    else:
        source = sorted(
            [int(center) for center in centers],
            key=lambda index: float(relevance_scores[index].item()),
            reverse=True,
        )
    if not source:
        raise ValueError("Cannot pad DPC selection without any selected frames or centers.")

    cursor = 0
    while len(padded) < threshold:
        padded.append(source[cursor % len(source)])
        cursor += 1
    return padded, True


def _segment_index_by_center(segments: Sequence[dict[str, int]]) -> dict[int, int]:
    return {
        int(segment["center_pool_index"]): int(segment["segment_index"])
        for segment in segments
    }


def dpc_sampling(
    video_path: str,
    *,
    query_file: str,
    threshold: int = 10,
    tau: float = 0.2,
    k_set: Sequence[int] | None = None,
    dpc_pool_size: int = 512,
    kneighbor_count: int = 5,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    clip_dtype: str | torch.dtype | None = "bf16",
    clip_do_center_crop: bool | None = False,
    aggregation: str = "max",
    batch_size: int = 32,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
    device: str | torch.device | None = None,
    cache_dense_scores: bool = False,
    **_: Any,
) -> FrameSelectionResult:
    if dpc_pool_size <= 0:
        raise ValueError(f"`dpc_pool_size` must be positive, got {dpc_pool_size}.")
    resolved_threshold = int(threshold)
    resolved_k_set = _resolve_k_set(k_set, threshold=resolved_threshold)
    selector_device = _resolve_selector_device(device)

    pool_selection = uniform_sampling(
        video_path=video_path,
        num_frames=int(dpc_pool_size),
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )
    pool_frames = pool_selection.frames
    if pool_frames is None or int(pool_frames.shape[0]) <= 0:
        raise ValueError("DPC frame selection requires at least one decoded frame.")
    pool_count = int(pool_frames.shape[0])

    dense_score_maps: torch.Tensor | None = None
    image_frame_scores: torch.Tensor | None = None
    clip_grid: tuple[int, int] | None = None
    if cache_dense_scores:
        (
            frame_embeddings,
            dense_score_maps,
            image_frame_scores,
            clip_grid,
            queries,
            clip_dtype_key,
        ) = _compute_clip_global_embeddings_and_score_maps(
            pool_frames,
            query_file=query_file,
            clip_model_name=clip_model_name,
            clip_dtype=clip_dtype,
            clip_do_center_crop=clip_do_center_crop,
            aggregation=aggregation,
            batch_size=batch_size,
            device=selector_device,
        )
        relevance_scores = image_frame_scores
    else:
        frame_embeddings = _compute_clip_global_embeddings(
            pool_frames,
            clip_model_name=clip_model_name,
            clip_dtype=clip_dtype,
            clip_do_center_crop=clip_do_center_crop,
            batch_size=batch_size,
            device=selector_device,
        )
        relevance_scores, queries, clip_dtype_key = _compute_query_scores(
            frame_embeddings,
            query_file=query_file,
            clip_model_name=clip_model_name,
            clip_dtype=clip_dtype,
            clip_do_center_crop=clip_do_center_crop,
            aggregation=aggregation,
            device=selector_device,
        )

    loop_history: list[dict[str, Any]] = []
    final_segmentation: DPCSegmentationResult | None = None
    selected_pool_indices: list[int] = []
    high_relevance_pool_indices: list[int] = []
    high_relevance_count = 0
    stopped_by_threshold = False
    dpc_center_ranking = _rank_dpc_centers(
        frame_embeddings,
        kneighbor_count=kneighbor_count,
    )

    for requested_k in resolved_k_set:
        effective_k = min(int(requested_k), pool_count)
        centers_by_score = dpc_center_ranking[:effective_k]
        segmentation = _build_temporal_segments(
            centers_by_score,
            frame_count=pool_count,
            requested_k=requested_k,
        )
        top_selected, high_relevance, high_count = _top_relevant_centers(
            segmentation.centers,
            relevance_scores,
            tau=tau,
            threshold=resolved_threshold,
        )
        loop_history.append(
            {
                "requested_k": int(requested_k),
                "effective_k": int(segmentation.effective_k),
                "center_pool_indices": list(segmentation.centers),
                "high_relevance_count": int(high_count),
                "satisfied_threshold": bool(high_count >= resolved_threshold),
            }
        )
        final_segmentation = segmentation
        high_relevance_pool_indices = high_relevance
        high_relevance_count = high_count
        if high_count >= resolved_threshold:
            selected_pool_indices = top_selected
            stopped_by_threshold = True
            break

    if final_segmentation is None:
        raise RuntimeError("DPC-KNN loop did not run.")

    fallback_used = False
    if not stopped_by_threshold:
        selected_pool_indices, high_relevance_pool_indices, high_relevance_count = (
            _fallback_centers(
                final_segmentation.centers,
                relevance_scores,
                tau=tau,
                threshold=resolved_threshold,
            )
        )
        fallback_used = True

    selected_pool_indices, padded_to_threshold = _pad_to_threshold(
        selected_pool_indices,
        centers=final_segmentation.centers,
        relevance_scores=relevance_scores,
        threshold=resolved_threshold,
    )
    selected_pool_indices = sorted(selected_pool_indices)

    sampled_indices = [
        int(index)
        for index in pool_selection.metadata.get("sampled_indices", list(range(pool_count)))
    ]
    selected_original_indices = [
        sampled_indices[index] if 0 <= index < len(sampled_indices) else index
        for index in selected_pool_indices
    ]
    segment_lookup = _segment_index_by_center(final_segmentation.segments)
    selected_segment_indices = [
        segment_lookup.get(index, -1)
        for index in selected_pool_indices
    ]
    selected_scores = [
        float(relevance_scores[index].item())
        for index in selected_pool_indices
    ]
    ddps_clip_score_cache: dict[str, Any] | None = None
    if cache_dense_scores:
        if dense_score_maps is None or image_frame_scores is None or clip_grid is None:
            raise RuntimeError("Dense score caching was requested but not populated.")
        selected_tensor = torch.tensor(
            selected_pool_indices,
            device=dense_score_maps.device,
            dtype=torch.long,
        )
        ddps_clip_score_cache = {
            "dense_score_maps": dense_score_maps.index_select(0, selected_tensor),
            "image_frame_scores": image_frame_scores.index_select(0, selected_tensor),
            "clip_grid": [int(clip_grid[0]), int(clip_grid[1])],
            "clip_model_name": clip_model_name,
            "clip_dtype": clip_dtype_key,
            "clip_do_center_crop": (
                None if clip_do_center_crop is None else bool(clip_do_center_crop)
            ),
            "query_file": str(Path(query_file).expanduser()),
            "queries": list(queries),
            "query_count": len(queries),
            "aggregation": aggregation,
            "frame_count": int(len(selected_pool_indices)),
            "source_pool_indices": list(selected_pool_indices),
        }
    metadata = {
        **pool_selection.metadata,
        "method": "DPC",
        "sampling_method": "DPC",
        "dpc_variant": "dpc_knn_temporal_segmentation_ddps_clip_global",
        "dpc_pool_size": int(dpc_pool_size),
        "actual_pool_frame_count": int(pool_count),
        "k_set": [int(k) for k in resolved_k_set],
        "final_k": int(final_segmentation.requested_k),
        "effective_final_k": int(final_segmentation.effective_k),
        "threshold": int(resolved_threshold),
        "tau": float(tau),
        "kneighbor_count": int(kneighbor_count),
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": aggregation,
        "cache_dense_scores": bool(cache_dense_scores),
        "segment_boundaries": final_segmentation.segments,
        "dpc_center_ranking_pool_indices": list(dpc_center_ranking),
        "dpc_center_pool_indices": list(final_segmentation.centers),
        "dpc_center_original_indices": [
            sampled_indices[index] if 0 <= index < len(sampled_indices) else index
            for index in final_segmentation.centers
        ],
        "high_relevance_count": int(high_relevance_count),
        "high_relevance_pool_indices": sorted(high_relevance_pool_indices),
        "selected_from_candidates": list(selected_pool_indices),
        "selected_segment_indices": list(selected_segment_indices),
        "selected_original_indices": list(selected_original_indices),
        "sampled_indices_before_dpc": list(sampled_indices),
        "sampled_indices": list(selected_original_indices),
        "representative_query_similarities": [
            float(relevance_scores[index].item())
            for index in final_segmentation.centers
        ],
        "selected_query_similarities": selected_scores,
        "loop_history": loop_history,
        "fallback_used": bool(fallback_used),
        "padded_to_threshold": bool(padded_to_threshold),
        "ddps_clip_score_cache": ddps_clip_score_cache,
        "num_frames": int(len(selected_pool_indices)),
    }

    return FrameSelectionResult(
        frames=pool_frames[selected_pool_indices],
        metadata=metadata,
    )
