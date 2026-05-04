from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...base.selection import FrameSelectionResult, PatchSelectionResult
from .selection_v1 import (
    _align_temporal_score_maps,
    _coerce_video_frames,
    _compute_qwen_merge_mean_scores,
    _compute_sliding_window_merged_scores,
    _compute_window_score_maps,
    _expand_scores_for_frame_duplication,
    _extract_video_grid,
    _load_queries,
    _prepare_frame_arrays,
    _resolve_patch_scoring_frames,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_device_key,
    _resolve_selection_mode,
    _resolve_spatial_merge_size,
    _resolve_temporal_patch_size,
    _resize_score_maps,
    SUPPORTED_QWEN_BACKENDS,
)
from .selection_v2 import (
    _compute_dense_patch_score_maps_and_frame_scores,
    _load_maskclip_components_v2,
    _load_text_embeddings_v2,
    _resolve_total_budget,
    preload_maskclip_patch_selection as _preload_maskclip_patch_selection_v2,
)


def preload_maskclip_patch_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    model: Any,
    **kwargs: Any,
) -> None:
    _preload_maskclip_patch_selection_v2(
        clip_model_name=clip_model_name,
        clip_dtype=clip_dtype,
        device=device,
        model=model,
        **kwargs,
    )


def _allocate_budget_with_softmax_capacities(
    frame_scores: torch.Tensor,
    *,
    total_budget: int,
    capacities: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"`temperature` must be positive, got {temperature}.")
    if total_budget < 0:
        raise ValueError(f"`total_budget` must be non-negative, got {total_budget}.")

    scores = frame_scores.to(dtype=torch.float32).flatten()
    scaled_scores = scores / temperature
    resolved_capacities = capacities.to(device=scores.device, dtype=torch.long).flatten()
    if resolved_capacities.shape != scores.shape:
        raise ValueError(
            "Capacity shape must match frame score shape: "
            f"scores={tuple(scores.shape)}, capacities={tuple(resolved_capacities.shape)}."
        )
    if torch.any(resolved_capacities < 0):
        raise ValueError("Capacities must be non-negative.")

    total_capacity = int(resolved_capacities.sum().item())
    allocatable_budget = min(int(total_budget), total_capacity)
    quotas = torch.zeros_like(scores, dtype=torch.float32)

    weights = torch.zeros_like(scores, dtype=torch.float32)
    active_capacity_mask = resolved_capacities > 0
    active_indices = torch.nonzero(active_capacity_mask, as_tuple=False).flatten()
    if active_indices.numel() > 0:
        weights[active_indices] = torch.softmax(
            scaled_scores[active_indices],
            dim=0,
        )

    if allocatable_budget == 0:
        return (
            torch.zeros_like(resolved_capacities),
            quotas,
            weights,
        )

    remaining_budget = float(allocatable_budget)
    active_mask = active_capacity_mask.clone()
    while remaining_budget > 1e-6:
        current_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if current_indices.numel() == 0:
            break

        current_weights = torch.softmax(scaled_scores[current_indices], dim=0)
        current_capacities = resolved_capacities[current_indices].to(dtype=torch.float32)
        tentative = current_weights * remaining_budget
        saturated = tentative >= current_capacities

        if torch.any(saturated):
            saturated_indices = current_indices[saturated]
            quotas[saturated_indices] = resolved_capacities[saturated_indices].to(
                dtype=torch.float32
            )
            remaining_budget -= float(
                resolved_capacities[saturated_indices].sum().item()
            )
            active_mask[saturated_indices] = False
            continue

        quotas[current_indices] = tentative
        remaining_budget = 0.0

    allocated = torch.floor(quotas).to(dtype=torch.long)
    remaining_integer_budget = allocatable_budget - int(allocated.sum().item())
    if remaining_integer_budget > 0:
        fractional = quotas - allocated.to(dtype=quotas.dtype)
        available_mask = allocated < resolved_capacities
        candidate_indices = torch.nonzero(available_mask, as_tuple=False).flatten()
        if candidate_indices.numel() > 0:
            candidate_fractional = fractional[candidate_indices]
            candidate_scores = scores[candidate_indices]
            ranking = torch.argsort(
                candidate_fractional + candidate_scores * 1e-6,
                descending=True,
            )
            ranked_indices = candidate_indices[ranking]
            allocated[ranked_indices[:remaining_integer_budget]] += 1

    if int(allocated.sum().item()) != allocatable_budget:
        raise RuntimeError(
            "Allocated patch budget does not match the allocatable budget: "
            f"allocated={int(allocated.sum().item())}, allocatable={allocatable_budget}."
        )

    return allocated, quotas, weights


def _compute_eligible_patch_mask(
    merged_scores: torch.Tensor,
    *,
    patch_score_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible_mask = merged_scores >= float(patch_score_threshold)
    eligible_counts = eligible_mask.reshape(eligible_mask.shape[0], -1).sum(dim=1)
    return eligible_mask, eligible_counts.to(dtype=torch.long)


def _select_topk_from_eligible_patches(
    merged_scores: torch.Tensor,
    *,
    eligible_mask: torch.Tensor,
    frame_scores: torch.Tensor,
    frame_weights: torch.Tensor,
    raw_budget: torch.Tensor,
    allocated_budget: torch.Tensor,
    eligible_counts: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    merged_h = int(merged_scores.shape[1])
    merged_w = int(merged_scores.shape[2])
    tokens_per_frame = merged_h * merged_w
    frame_count = int(merged_scores.shape[0])
    frame_token_scores = merged_scores.reshape(frame_count, -1)
    eligible_flat = eligible_mask.reshape(frame_count, -1)
    keep_counts = allocated_budget.to(dtype=torch.long)

    min_scores = frame_token_scores.min(dim=1).values
    max_scores = frame_token_scores.max(dim=1).values
    if torch.any(keep_counts > eligible_counts):
        violating_frames = torch.nonzero(
            keep_counts > eligible_counts,
            as_tuple=False,
        ).flatten()
        first_frame_idx = int(violating_frames[0].item())
        raise ValueError(
            "Allocated frame budget exceeds the eligible patch count: "
            f"frame_index={first_frame_idx}, "
            f"keep_count={int(keep_counts[first_frame_idx].item())}, "
            f"eligible_count={int(eligible_counts[first_frame_idx].item())}."
        )

    selected_indices = torch.empty(0, device=merged_scores.device, dtype=torch.long)
    max_keep_count = int(keep_counts.max().item()) if frame_count > 0 else 0
    if max_keep_count > 0:
        masked_scores = frame_token_scores.masked_fill(
            ~eligible_flat,
            torch.finfo(frame_token_scores.dtype).min,
        )
        topk_indices = torch.topk(
            masked_scores,
            k=max_keep_count,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        rank_positions = torch.arange(
            max_keep_count,
            device=merged_scores.device,
            dtype=keep_counts.dtype,
        ).unsqueeze(0)
        chosen_mask = rank_positions < keep_counts.unsqueeze(1)
        invalid_index = topk_indices.new_full((), tokens_per_frame)
        selected_per_frame = topk_indices.masked_fill(~chosen_mask, invalid_index)
        selected_per_frame = torch.sort(selected_per_frame, dim=1).values
        valid_mask = selected_per_frame != invalid_index
        frame_offsets = (
            torch.arange(
                frame_count,
                device=merged_scores.device,
                dtype=topk_indices.dtype,
            ).unsqueeze(1)
            * tokens_per_frame
        )
        selected_indices = (selected_per_frame + frame_offsets).masked_select(valid_mask)

    eligible_count_list = eligible_counts.tolist()
    allocated_list = keep_counts.tolist()
    min_score_list = min_scores.tolist()
    max_score_list = max_scores.tolist()
    frame_score_list = frame_scores.tolist()
    frame_weight_list = frame_weights.tolist()
    raw_budget_list = raw_budget.tolist()

    frame_metadata = [
        {
            "frame_index": frame_idx,
            "tokens_per_frame": tokens_per_frame,
            "eligible_count": int(eligible_count_list[frame_idx]),
            "initial_allocated_budget": int(allocated_list[frame_idx]),
            "initial_keep_count": int(allocated_list[frame_idx]),
            "final_keep_count": int(allocated_list[frame_idx]),
            "reallocated_count": 0,
            "score_min": float(min_score_list[frame_idx]),
            "score_max": float(max_score_list[frame_idx]),
            "frame_importance_score": float(frame_score_list[frame_idx]),
            "softmax_weight": float(frame_weight_list[frame_idx]),
            "raw_budget": float(raw_budget_list[frame_idx]),
        }
        for frame_idx in range(frame_count)
    ]

    return selected_indices, frame_metadata


def maskclip_patch_selection(
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
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "MaskCLIP patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )
    if patch_score_threshold is None:
        raise ValueError(
            "`patch_score_threshold` must be a non-null float for selection_v4."
        )

    resolved_selection_mode = _resolve_selection_mode(selection_mode)
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
    scoring_frames, duplication_info = _resolve_patch_scoring_frames(frame_selection)
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

    selector_device = _resolve_device(device, video_features)
    selector_device_key = _resolve_device_key(selector_device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    queries = _load_queries(query_file)
    image_processor, _, vision_model, _ = _load_maskclip_components_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        clip_do_center_crop,
    )

    text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        clip_do_center_crop,
        queries,
    )
    frame_arrays = _prepare_frame_arrays(scoring_frames)
    dense_score_maps, image_frame_scores, clip_grid = (
        _compute_dense_patch_score_maps_and_frame_scores(
            frame_arrays,
            image_processor=image_processor,
            vision_model=vision_model,
            text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
            aggregation=aggregation,
            batch_size=batch_size,
            device=selector_device,
            clip_dtype=resolved_clip_dtype,
        )
    )
    dense_score_maps = _expand_scores_for_frame_duplication(
        dense_score_maps,
        duplication_info,
    )
    image_frame_scores = _expand_scores_for_frame_duplication(
        image_frame_scores,
        duplication_info,
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
    underfilled_token_count = total_budget_value - selected_token_count
    metadata = {
        "selector_type": "maskclip_patch_selection_v4",
        "allocation_strategy": "one_pass_capacity_aware",
        "selection_mode": resolved_selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "clip_do_center_crop": (
            None if clip_do_center_crop is None else bool(clip_do_center_crop)
        ),
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
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
        "underfilled_token_count": int(underfilled_token_count),
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
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


maskclip_patch_selection.preload = preload_maskclip_patch_selection
