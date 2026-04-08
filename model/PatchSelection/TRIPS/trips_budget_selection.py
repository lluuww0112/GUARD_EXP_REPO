from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...base.selection import FrameSelectionResult, PatchSelectionResult
from model.PatchSelection.DenseDPS.selection_v1 import (
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
from model.PatchSelection.DenseDPS.selection_v2 import (
    _allocate_budget_with_softmax,
    _compute_dense_patch_score_maps_and_frame_scores,
    _load_maskclip_components_v2,
    _load_text_embeddings_v2,
    _resolve_total_budget,
    preload_maskclip_patch_selection as _preload_maskclip_patch_selection_v2,
)


def preload_budget_fuse_patch_selection(
    *args: Any,
    **kwargs: Any,
) -> None:
    _preload_maskclip_patch_selection_v2(*args, **kwargs)


def _build_fused_token(
    *,
    inattentive_features: torch.Tensor,
    inattentive_scores: torch.Tensor,
    fuse_strategy: str,
) -> torch.Tensor:
    strategy = str(fuse_strategy).strip().lower()
    if strategy == "mean":
        return inattentive_features.mean(dim=0, keepdim=True)
    if strategy == "score_weighted_mean":
        weights = torch.softmax(inattentive_scores.float(), dim=0).to(
            device=inattentive_features.device,
            dtype=inattentive_features.dtype,
        )
        return (inattentive_features * weights.unsqueeze(-1)).sum(
            dim=0,
            keepdim=True,
        )
    raise ValueError(
        f"Unsupported `fuse_strategy`: {fuse_strategy}. "
        "Available: mean, score_weighted_mean."
    )


def _select_topk_with_frame_budget(
    merged_scores: torch.Tensor,
    *,
    allocated_budget: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    frame_count = int(merged_scores.shape[0])
    tokens_per_frame = int(merged_scores.shape[1] * merged_scores.shape[2])
    frame_token_scores = merged_scores.reshape(frame_count, tokens_per_frame)
    keep_counts = allocated_budget.to(dtype=torch.long)

    selected_chunks: list[torch.Tensor] = []
    frame_metadata: list[dict[str, Any]] = []
    min_scores = frame_token_scores.min(dim=1).values
    max_scores = frame_token_scores.max(dim=1).values
    for frame_idx in range(frame_count):
        keep_count = int(keep_counts[frame_idx].item())
        if keep_count <= 0:
            frame_metadata.append(
                {
                    "frame_index": frame_idx,
                    "tokens_per_frame": tokens_per_frame,
                    "keep_count": 0,
                    "score_min": float(min_scores[frame_idx].item()),
                    "score_max": float(max_scores[frame_idx].item()),
                }
            )
            continue

        frame_scores = frame_token_scores[frame_idx]
        topk_indices = torch.topk(
            frame_scores,
            k=keep_count,
            dim=0,
            largest=True,
            sorted=False,
        ).indices
        selected_chunks.append(torch.sort(topk_indices + frame_idx * tokens_per_frame).values)
        frame_metadata.append(
            {
                "frame_index": frame_idx,
                "tokens_per_frame": tokens_per_frame,
                "keep_count": keep_count,
                "score_min": float(min_scores[frame_idx].item()),
                "score_max": float(max_scores[frame_idx].item()),
            }
        )

    if not selected_chunks:
        raise ValueError("Budget-fuse selector produced an empty attentive selection.")
    return torch.cat(selected_chunks, dim=0), frame_metadata


def budget_fuse_patch_selection(
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
    keep_ratio: float = 0.05,
    total_budget: int | None = None,
    temperature: float = 0.2,
    selection_mode: str = "naive_mean",
    window_size: int = 4,
    window_stride: int = 2,
    batch_size: int = 8,
    aggregation: str = "max",
    spatial_merge_size: int | None = None,
    fuse_strategy: str = "score_weighted_mean",
    device: str | torch.device | None = None,
    **_: Any,
) -> PatchSelectionResult:
    if backend not in SUPPORTED_QWEN_BACKENDS:
        raise ValueError(
            "Budget-fuse patch selection currently supports only Qwen backends, "
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
    if expected_tokens < 2:
        raise ValueError(
            "Budget-fuse selection requires at least two video tokens, "
            f"but got {expected_tokens}."
        )

    total_budget_value = _resolve_total_budget(
        merged_scores=merged_scores,
        keep_ratio=keep_ratio,
        total_budget=total_budget,
    )
    tokens_per_frame = int(merged_scores.shape[1] * merged_scores.shape[2])
    allocated_budget, raw_budget, frame_weights = _allocate_budget_with_softmax(
        image_frame_scores,
        total_budget=total_budget_value,
        tokens_per_frame=tokens_per_frame,
        temperature=temperature,
    )
    attentive_indices, frame_metadata = _select_topk_with_frame_budget(
        merged_scores,
        allocated_budget=allocated_budget,
    )
    attentive_indices = attentive_indices.to(
        device=video_features.device,
        dtype=torch.long,
    )

    flat_scores = merged_scores.reshape(-1)
    attentive_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    attentive_mask[attentive_indices.to(flat_scores.device)] = True
    inattentive_indices = torch.nonzero(~attentive_mask, as_tuple=False).flatten()
    if int(inattentive_indices.numel()) == 0:
        raise RuntimeError("Budget-fuse expected inattentive tokens to fuse, but none remained.")

    fused_anchor_index = int(inattentive_indices[0].item())
    fused_feature = _build_fused_token(
        inattentive_features=video_features[inattentive_indices.to(video_features.device)],
        inattentive_scores=flat_scores[inattentive_indices].to(video_features.device),
        fuse_strategy=fuse_strategy,
    )

    selected_indices = torch.cat(
        [
            attentive_indices,
            torch.tensor([fused_anchor_index], device=video_features.device, dtype=torch.long),
        ],
        dim=0,
    )
    selected_indices = torch.sort(selected_indices).values

    feature_rows: list[torch.Tensor] = []
    for index in selected_indices.detach().cpu().tolist():
        if int(index) == fused_anchor_index:
            feature_rows.append(fused_feature)
        else:
            feature_rows.append(video_features[int(index)].unsqueeze(0))
    selected_features = torch.cat(feature_rows, dim=0)

    metadata = {
        "selector_type": "trips_budget_patch_selection",
        "selector_variant": "frame_aware_budget_plus_global_fuse",
        "clip_model_name": clip_model_name,
        "clip_dtype": clip_dtype_key,
        "query_file": str(Path(query_file).expanduser()),
        "queries": list(queries),
        "query_count": len(queries),
        "aggregation": aggregation,
        "keep_ratio": float(keep_ratio),
        "total_budget": int(total_budget_value),
        "temperature": float(temperature),
        "fuse_strategy": str(fuse_strategy).strip().lower(),
        "fused_token_count": 1,
        "fused_anchor_index": fused_anchor_index,
        "selected_token_count": int(selected_indices.numel()),
        "attentive_token_count": int(attentive_indices.numel()),
        "inattentive_token_count": int(inattentive_indices.numel()),
        "video_token_count": expected_tokens,
        "spatial_merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "frame_count": frame_count,
        "temporal_grid_t": grid_t,
        "clip_grid_hw": [int(clip_grid[0]), int(clip_grid[1])],
        "raw_grid_thw": [grid_t, raw_grid_h, raw_grid_w],
        "merged_grid_thw": [grid_t, raw_grid_h // merge_size, raw_grid_w // merge_size],
        "frame_importance_scores": [float(score) for score in image_frame_scores.tolist()],
        "frame_softmax_weights": [float(weight) for weight in frame_weights.tolist()],
        "frame_raw_budgets": [float(budget) for budget in raw_budget.tolist()],
        "frame_allocated_budgets": [int(budget) for budget in allocated_budget.tolist()],
        "per_frame": frame_metadata,
        **mode_metadata,
    }
    return PatchSelectionResult(
        selected_indices=selected_indices,
        selected_features=selected_features,
        metadata=metadata,
    )


budget_fuse_patch_selection.preload = preload_budget_fuse_patch_selection
