from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model.base.selection import FrameSelectionResult, PatchSelectionResult
from model.PatchSelection.DenseDPS.selection_v1 import (
    _align_temporal_score_maps,
    _coerce_video_frames,
    _compute_qwen_merge_mean_scores,
    _compute_sliding_window_merged_scores,
    _compute_window_score_maps,
    _extract_video_grid,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_selection_mode,
    _resolve_spatial_merge_size,
    _resolve_temporal_patch_size,
    _resize_score_maps,
    SUPPORTED_QWEN_BACKENDS,
)
from model.PatchSelection.DenseDPS.selection_v2 import _resolve_total_budget
from model.PatchSelection.DenseDPS.selection_v4 import (
    _allocate_budget_with_softmax_capacities,
    _compute_eligible_patch_mask,
    _select_topk_from_eligible_patches,
    maskclip_patch_selection as _fallback_maskclip_patch_selection_v4,
    preload_maskclip_patch_selection,
)


def _load_dpc_dense_score_cache(
    frame_selection: FrameSelectionResult,
    *,
    frame_count: int,
    clip_model_name: str,
    clip_dtype_key: str,
    clip_do_center_crop: bool | None,
    query_file: str,
    aggregation: str,
    selector_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], dict[str, Any]] | None:
    cache = frame_selection.metadata.get("ddps_clip_score_cache")
    if not isinstance(cache, dict):
        return None
    if int(cache.get("frame_count", -1)) != int(frame_count):
        return None
    if str(cache.get("clip_model_name")) != str(clip_model_name):
        return None
    if str(cache.get("clip_dtype")) != str(clip_dtype_key):
        return None
    expected_center_crop = None if clip_do_center_crop is None else bool(clip_do_center_crop)
    if cache.get("clip_do_center_crop") != expected_center_crop:
        return None
    if str(cache.get("query_file")) != str(Path(query_file).expanduser()):
        return None
    if str(cache.get("aggregation")) != str(aggregation):
        return None

    dense_score_maps = cache.get("dense_score_maps")
    image_frame_scores = cache.get("image_frame_scores")
    clip_grid = cache.get("clip_grid")
    if not torch.is_tensor(dense_score_maps) or not torch.is_tensor(image_frame_scores):
        return None
    if not isinstance(clip_grid, (list, tuple)) or len(clip_grid) != 2:
        return None

    metadata = {
        "clip_score_cache_used": True,
        "clip_score_cache_source": "dpc_frame_selection",
        "clip_score_cache_source_pool_indices": list(
            cache.get("source_pool_indices", [])
        ),
    }
    return (
        dense_score_maps.to(device=selector_device, dtype=torch.float32),
        image_frame_scores.to(device=selector_device, dtype=torch.float32),
        (int(clip_grid[0]), int(clip_grid[1])),
        metadata,
    )


def dpc_ddps_patch_selection(
    video_features: torch.Tensor,
    *,
    frame_selection: FrameSelectionResult,
    model_inputs: dict[str, Any],
    extraction_metadata: dict[str, Any],
    model: Any,
    backend: str,
    query_file: str,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_do_center_crop: bool | None = None,
    keep_ratio: float = 0.5,
    total_budget: int | None = None,
    temperature: float = 1.0,
    patch_score_threshold: float = 0.0,
    selection_mode: str = "naive_mean",
    window_size: int = 2,
    window_stride: int = 1,
    batch_size: int = 8,
    aggregation: str = "max",
    spatial_merge_size: int | None = None,
    device: str | torch.device | None = None,
    **kwargs: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "DPC-DDPS patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )
    if patch_score_threshold is None:
        raise ValueError("`patch_score_threshold` must be a non-null float.")

    resolved_selection_mode = _resolve_selection_mode(selection_mode)
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
    selector_device = _resolve_device(device, video_features)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    cached_scores = _load_dpc_dense_score_cache(
        frame_selection,
        frame_count=frame_count,
        clip_model_name=clip_model_name,
        clip_dtype_key=clip_dtype_key,
        clip_do_center_crop=clip_do_center_crop,
        query_file=query_file,
        aggregation=aggregation,
        selector_device=selector_device,
    )
    if cached_scores is None:
        return _fallback_maskclip_patch_selection_v4(
            video_features=video_features,
            frame_selection=frame_selection,
            model_inputs=model_inputs,
            extraction_metadata=extraction_metadata,
            model=model,
            backend=backend,
            query_file=query_file,
            clip_model_name=clip_model_name,
            clip_dtype=clip_dtype,
            clip_do_center_crop=clip_do_center_crop,
            keep_ratio=keep_ratio,
            total_budget=total_budget,
            temperature=temperature,
            patch_score_threshold=patch_score_threshold,
            selection_mode=selection_mode,
            window_size=window_size,
            window_stride=window_stride,
            batch_size=batch_size,
            aggregation=aggregation,
            spatial_merge_size=spatial_merge_size,
            device=device,
            **kwargs,
        )

    dense_score_maps, image_frame_scores, clip_grid, cache_metadata = cached_scores
    grid_t, raw_grid_h, raw_grid_w = _extract_video_grid(
        model_inputs=model_inputs,
        extraction_metadata=extraction_metadata,
    )
    merge_size = int(
        spatial_merge_size
        if spatial_merge_size is not None
        else _resolve_spatial_merge_size(model, default=2)
    )
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")
    temporal_patch_size = _resolve_temporal_patch_size(
        model,
        frame_count=frame_count,
        grid_t=grid_t,
        default=1,
    )

    raw_score_maps = _resize_score_maps(
        dense_score_maps,
        target_height=raw_grid_h,
        target_width=raw_grid_w,
    )
    raw_score_maps = _align_temporal_score_maps(
        raw_score_maps,
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
    )
    image_frame_scores = _align_temporal_score_maps(
        image_frame_scores.view(frame_count, 1, 1),
        grid_t=grid_t,
        temporal_patch_size=temporal_patch_size,
    ).flatten()

    mode_metadata: dict[str, Any] = {}
    if resolved_selection_mode == "naive_mean":
        merged_scores = _compute_qwen_merge_mean_scores(
            raw_score_maps,
            merge_size=merge_size,
        )
        mode_metadata["score_reduction"] = "mean_within_qwen_merge_unit"
    else:
        window_score_maps = _compute_window_score_maps(
            raw_score_maps,
            window_size=window_size,
            window_stride=window_stride,
        )
        merged_scores = _compute_sliding_window_merged_scores(
            window_score_maps,
            raw_height=raw_grid_h,
            raw_width=raw_grid_w,
            merge_size=merge_size,
            window_size=window_size,
            window_stride=window_stride,
        )
        mode_metadata["window_size"] = int(window_size)
        mode_metadata["window_stride"] = int(window_stride)

    expected_tokens = int(video_features.shape[0])
    if int(merged_scores.numel()) != expected_tokens:
        raise ValueError(
            "Merged token score count does not match extracted video feature count: "
            f"scores={int(merged_scores.numel())}, features={expected_tokens}."
        )

    total_budget_value = _resolve_total_budget(
        merged_scores=merged_scores,
        keep_ratio=keep_ratio,
        total_budget=total_budget,
    )
    eligible_mask, eligible_counts = _compute_eligible_patch_mask(
        merged_scores,
        patch_score_threshold=patch_score_threshold,
    )
    allocated_budget, raw_budget, frame_weights = (
        _allocate_budget_with_softmax_capacities(
            image_frame_scores,
            total_budget=total_budget_value,
            capacities=eligible_counts,
            temperature=temperature,
        )
    )
    selected_indices, frame_metadata = _select_topk_from_eligible_patches(
        merged_scores,
        eligible_mask=eligible_mask,
        frame_scores=image_frame_scores,
        frame_weights=frame_weights,
        raw_budget=raw_budget,
        allocated_budget=allocated_budget,
        eligible_counts=eligible_counts,
    )

    selected_indices = selected_indices.to(
        device=video_features.device,
        dtype=torch.long,
    )
    selected_token_count = int(selected_indices.numel())
    allocatable_budget = min(total_budget_value, int(eligible_counts.sum().item()))
    metadata = {
        "selector_type": "dpc_ddps_patch_selection_v4_cached",
        "allocation_strategy": "one_pass_capacity_aware",
        "selection_mode": resolved_selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(query_file).expanduser()),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
        "total_budget": int(total_budget_value),
        "allocatable_budget": int(allocatable_budget),
        "temperature": float(temperature),
        "patch_score_threshold": float(patch_score_threshold),
        "spatial_merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "frame_count": frame_count,
        "temporal_grid_t": grid_t,
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "raw_grid_thw": [grid_t, raw_grid_h, raw_grid_w],
        "merged_grid_thw": [
            grid_t,
            raw_grid_h // merge_size,
            raw_grid_w // merge_size,
        ],
        "eligible_token_count": int(eligible_counts.sum().item()),
        "selected_token_count": selected_token_count,
        "video_token_count": expected_tokens,
        "underfilled_token_count": int(total_budget_value - selected_token_count),
        "reallocated_token_count": 0,
        "frame_importance_scores": [float(score) for score in image_frame_scores.tolist()],
        "frame_softmax_weights": [float(weight) for weight in frame_weights.tolist()],
        "frame_raw_budgets": [float(budget) for budget in raw_budget.tolist()],
        "initial_frame_allocated_budgets": [
            int(budget) for budget in allocated_budget.tolist()
        ],
        "final_frame_allocated_budgets": [
            int(budget) for budget in allocated_budget.tolist()
        ],
        "frame_allocated_budgets": [
            int(budget) for budget in allocated_budget.tolist()
        ],
        "per_frame": frame_metadata,
        **cache_metadata,
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


dpc_ddps_patch_selection.preload = preload_maskclip_patch_selection
