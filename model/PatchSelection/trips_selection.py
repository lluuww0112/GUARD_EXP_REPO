from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..base.selection import FrameSelectionResult, PatchSelectionResult
from .selection_v1 import (
    _align_temporal_score_maps,
    _coerce_video_frames,
    _compute_qwen_merge_mean_scores,
    _compute_sliding_window_merged_scores,
    _compute_window_score_maps,
    _extract_video_grid,
    _load_queries,
    _prepare_frame_arrays,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_device_key,
    _resolve_model_device,
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
)


def preload_trips_lite_patch_selection(
    *,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    model: Any,
    **_: Any,
) -> None:
    preload_device = (
        torch.device(device)
        if device is not None
        else _resolve_model_device(model)
    )
    device_key = _resolve_device_key(preload_device)
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    _load_maskclip_components_v2(clip_model_name, device_key, clip_dtype_key)


def trips_lite_patch_selection(
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
    keep_ratio: float = 0.5,
    total_budget: int | None = None,
    selection_mode: str = "naive_mean",
    window_size: int = 4,
    window_stride: int = 2,
    batch_size: int = 8,
    aggregation: str = "max",
    spatial_merge_size: int | None = None,
    device: str | torch.device | None = None,
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "TRIPS-lite patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )

    resolved_selection_mode = _resolve_selection_mode(selection_mode)
    frames = _coerce_video_frames(frame_selection)
    frame_count = int(frames.shape[0])
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
    )
    text_embeddings = _load_text_embeddings_v2(
        clip_model_name,
        selector_device_key,
        clip_dtype_key,
        queries,
    )
    frame_arrays = _prepare_frame_arrays(frames)
    dense_score_maps, _, clip_grid = _compute_dense_patch_score_maps_and_frame_scores(
        frame_arrays,
        image_processor=image_processor,
        vision_model=vision_model,
        text_embeddings_t=text_embeddings.transpose(0, 1).contiguous(),
        aggregation=aggregation,
        batch_size=batch_size,
        device=selector_device,
        clip_dtype=resolved_clip_dtype,
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
    flat_scores = merged_scores.reshape(-1)
    selected_indices = torch.topk(
        flat_scores,
        k=total_budget_value,
        dim=0,
        largest=True,
        sorted=False,
    ).indices
    selected_indices = torch.sort(selected_indices).values.to(
        device=video_features.device,
        dtype=torch.long,
    )

    selected_scores = flat_scores[selected_indices.to(flat_scores.device)].detach().cpu()

    metadata = {
        "selector_type": "trips_lite_patch_selection",
        "selection_mode": resolved_selection_mode,
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
        "total_budget": int(total_budget_value),
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
        "selected_token_count": int(selected_indices.numel()),
        "video_token_count": expected_tokens,
        "selected_scores": [float(score) for score in selected_scores.tolist()],
        "score_min": float(flat_scores.min().detach().cpu().item()),
        "score_max": float(flat_scores.max().detach().cpu().item()),
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


trips_lite_patch_selection.preload = preload_trips_lite_patch_selection
