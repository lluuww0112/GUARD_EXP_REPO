from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ...base.selection import FrameSelectionResult, PatchSelectionResult
from .selection_v1 import (
    _aggregate_query_scores,
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
    _load_maskclip_components_v2,
    _load_text_embeddings_v2,
    _resolve_total_budget,
)
from .selection_v4 import (
    _allocate_budget_with_softmax_capacities,
    _compute_eligible_patch_mask,
    _select_topk_from_eligible_patches,
    preload_maskclip_patch_selection,
)


def _add_gaussian_noise_to_query_embeddings(
    text_embeddings: torch.Tensor,
    *,
    noise_scale: float,
    noise_seed: int | None,
) -> torch.Tensor:
    resolved_scale = float(noise_scale)
    if resolved_scale < 0.0:
        raise ValueError(
            f"`query_noise_scale` must be non-negative, got {resolved_scale}."
        )
    if resolved_scale == 0.0:
        return text_embeddings

    original_dtype = text_embeddings.dtype
    embeddings = text_embeddings.to(dtype=torch.float32)
    noise_std = math.sqrt(resolved_scale)
    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(int(noise_seed))

    noise = torch.randn(
        embeddings.shape,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    return F.normalize(embeddings + noise_std * noise, dim=-1).to(
        dtype=original_dtype
    )


def _compute_dense_patch_score_maps_and_clean_frame_scores(
    frame_arrays: list[Any],
    *,
    image_processor: Any,
    vision_model: Any,
    patch_text_embeddings_t: torch.Tensor,
    frame_text_embeddings_t: torch.Tensor,
    aggregation: str,
    batch_size: int,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    score_chunks: list[torch.Tensor] = []
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
            if clip_dtype is None:
                pixel_values = pixel_values.to(device=device, non_blocking=True)
            else:
                pixel_values = pixel_values.to(
                    device=device,
                    dtype=clip_dtype,
                    non_blocking=True,
                )

            patch_embeddings, image_latents = vision_model(pixel_values)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
            image_latents = F.normalize(image_latents, dim=-1)

            patch_scores = patch_embeddings @ patch_text_embeddings_t
            patch_scores = _aggregate_query_scores(
                patch_scores,
                aggregation=aggregation,
            )

            frame_scores = image_latents @ frame_text_embeddings_t
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

            score_chunks.append(
                patch_scores.view(patch_scores.shape[0], clip_h, clip_w)
            )
            frame_score_chunks.append(frame_scores)

    if clip_grid is None:
        raise ValueError("No frames were provided for dense patch scoring.")

    return (
        torch.cat(score_chunks, dim=0),
        torch.cat(frame_score_chunks, dim=0),
        clip_grid,
    )


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
    query_noise_scale: float = 0.0,
    query_noise_seed: int | None = None,
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
            "`patch_score_threshold` must be a non-null float for selection_v5."
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

    clean_text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        clip_do_center_crop,
        queries,
    )
    noisy_text_embeddings = _add_gaussian_noise_to_query_embeddings(
        clean_text_embeddings,
        noise_scale=query_noise_scale,
        noise_seed=query_noise_seed,
    )
    frame_arrays = _prepare_frame_arrays(scoring_frames)
    dense_score_maps, image_frame_scores, clip_grid = (
        _compute_dense_patch_score_maps_and_clean_frame_scores(
            frame_arrays,
            image_processor=image_processor,
            vision_model=vision_model,
            patch_text_embeddings_t=(
                noisy_text_embeddings.transpose(0, 1).contiguous()
            ),
            frame_text_embeddings_t=(
                clean_text_embeddings.transpose(0, 1).contiguous()
            ),
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
        "selector_type": "maskclip_patch_selection_v5",
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
        "query_noise_scale": float(query_noise_scale),
        "query_noise_seed": (
            None if query_noise_seed is None else int(query_noise_seed)
        ),
        "query_noise_applies_to": "patch_scores_only",
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
        "frame_importance_scores": [
            float(score) for score in image_frame_scores.tolist()
        ],
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
