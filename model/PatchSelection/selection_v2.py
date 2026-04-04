from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..base.selection import FrameSelectionResult, PatchSelectionResult
from .selection_v1 import (
    SUPPORTED_QWEN_BACKENDS,
    _align_temporal_score_maps,
    _coerce_video_frames,
    _compute_dense_patch_score_maps,
    _encode_text_queries,
    _extract_video_grid,
    _load_maskclip_components,
    _load_queries,
    _resolve_clip_dtype,
    _resolve_device,
    _resolve_model_device,
    _resolve_spatial_merge_size,
    _resolve_temporal_patch_size,
    _resize_score_maps,
    _select_topk_per_frame,
)


def _compute_qwen_merge_mean_scores(
    raw_score_maps: torch.Tensor,
    *,
    merge_size: int,
) -> torch.Tensor:
    if merge_size <= 0:
        raise ValueError(f"`spatial_merge_size` must be positive, got {merge_size}.")

    raw_height = int(raw_score_maps.shape[-2])
    raw_width = int(raw_score_maps.shape[-1])
    if raw_height % merge_size != 0 or raw_width % merge_size != 0:
        raise ValueError(
            "Raw Qwen spatial grid must be divisible by spatial_merge_size: "
            f"grid=({raw_height}, {raw_width}), merge_size={merge_size}."
        )

    return F.avg_pool2d(
        raw_score_maps.unsqueeze(1),
        kernel_size=merge_size,
        stride=merge_size,
    ).squeeze(1)


def preload_maskclip_qwen_merge_patch_selection(
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
    device_key = (
        preload_device.type
        if preload_device.index is None
        else str(preload_device)
    )
    _, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    _load_maskclip_components(clip_model_name, device_key, clip_dtype_key)


def maskclip_naive_patch_selection(
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
    batch_size: int = 8,
    aggregation: str = "max",
    spatial_merge_size: int | None = None,
    device: str | torch.device | None = None,
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "MaskCLIP Qwen-merge patch selection currently supports only Qwen backends, "
            f"got `{backend}`."
        )

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
    temporal_patch_size = _resolve_temporal_patch_size(
        model,
        frame_count=frame_count,
        grid_t=grid_t,
        default=1,
    )

    selector_device = _resolve_device(device, video_features)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    queries = _load_queries(query_file)
    image_processor, tokenizer, vision_model, text_model = _load_maskclip_components(
        clip_model_name,
        selector_device.type if selector_device.index is None else str(selector_device),
        clip_dtype_key,
    )

    text_embeddings = _encode_text_queries(
        queries,
        tokenizer=tokenizer,
        text_model=text_model,
        device=selector_device,
    )
    dense_score_maps, clip_grid = _compute_dense_patch_score_maps(
        frames,
        image_processor=image_processor,
        vision_model=vision_model,
        text_embeddings=text_embeddings,
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
    merged_scores = _compute_qwen_merge_mean_scores(
        raw_score_maps,
        merge_size=merge_size,
    )
    selected_indices, frame_metadata = _select_topk_per_frame(
        merged_scores,
        keep_ratio=keep_ratio,
    )

    expected_tokens = int(video_features.shape[0])
    if int(merged_scores.numel()) != expected_tokens:
        raise ValueError(
            "Merged token score count does not match extracted video feature count: "
            f"scores={int(merged_scores.numel())}, features={expected_tokens}."
        )

    selected_indices = selected_indices.to(
        device=video_features.device,
        dtype=torch.long,
    )
    metadata = {
        "selector_type": "maskclip_qwen_merge_patch_selection",
        "score_reduction": "mean_within_qwen_merge_unit",
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": queries,
        "query_count": len(queries),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
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
        "per_frame": frame_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


maskclip_naive_patch_selection.preload = (
    preload_maskclip_qwen_merge_patch_selection
)
