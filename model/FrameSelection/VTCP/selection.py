from __future__ import annotations

import math
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from model.PatchSelection.DenseDPS.cilp_model import CLIPTextModel, CLIPVisionModel_v2
from model.base.selection import (
    FrameSelectionResult,
    QWEN_VISION_FACTOR,
    _open_video_for_sampling,
    _resize_frame,
)

from .controller import RenoStrideController


CLIP_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
SUPPORTED_SIMILARITY_METRICS = {"cos_sim", "l1", "l2"}


def _resolve_clip_dtype(
    clip_dtype: str | torch.dtype | None,
) -> tuple[torch.dtype | None, str]:
    if clip_dtype is None:
        return None, "default"
    if isinstance(clip_dtype, torch.dtype):
        return clip_dtype, str(clip_dtype).replace("torch.", "")

    normalized = str(clip_dtype).strip().lower()
    if normalized == "default":
        return None, "default"
    resolved = CLIP_DTYPE_MAP.get(normalized)
    if resolved is None:
        available = ", ".join(sorted(CLIP_DTYPE_MAP))
        raise ValueError(
            f"Unsupported `clip_dtype`: {clip_dtype}. Available: {available}."
        )
    return resolved, normalized


def _resolve_device(
    device: str | torch.device | None,
) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_device_key(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{0 if device.index is None else int(device.index)}"
    if device.index is None:
        return device.type
    return str(device)


def _resolve_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)

    try:
        first_parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")
    return first_parameter.device


@lru_cache(maxsize=4)
def _load_clip_components(
    clip_model_name: str,
    device_key: str,
    clip_dtype_key: str,
) -> tuple[CLIPImageProcessor, Any, CLIPVisionModel_v2, CLIPTextModel]:
    clip_dtype, _ = _resolve_clip_dtype(clip_dtype_key)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel_v2(clip_model_name)
    text_model = CLIPTextModel(clip_model_name)

    if clip_dtype is None:
        vision_model = vision_model.to(device_key)
        text_model = text_model.to(device_key)
    else:
        vision_model = vision_model.to(device=device_key, dtype=clip_dtype)
        text_model = text_model.to(device=device_key, dtype=clip_dtype)

    vision_model.eval()
    text_model.eval()
    return image_processor, tokenizer, vision_model, text_model


def _load_query_text(
    *,
    query_text: str | None,
    query_file: str | None,
) -> tuple[str, str]:
    resolved_query = str(query_text or "").strip()
    if resolved_query:
        return resolved_query, "query_text"

    if query_file is None:
        raise ValueError("`query_text` or `query_file` must provide a non-empty query.")

    query_path = Path(query_file).expanduser()
    if not query_path.exists():
        raise FileNotFoundError(f"Query file does not exist: {query_path}")
    if not query_path.is_file():
        raise ValueError(f"Query file path must point to a file: {query_path}")

    lines = [
        line.strip()
        for line in query_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError(f"Query file does not contain any non-empty lines: {query_path}")
    return lines[0], "query_file"


def preload_vtcp_sampling(
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
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    if preload_device.type == "cpu":
        resolved_clip_dtype = None
        clip_dtype_key = "default"

    del resolved_clip_dtype
    _load_clip_components(
        clip_model_name,
        _resolve_device_key(preload_device),
        clip_dtype_key,
    )


def _prepare_frame_array(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.flags["C_CONTIGUOUS"]:
        return frame_rgb
    return np.ascontiguousarray(frame_rgb)


def _resolve_embedding_downsample_stride(
    *,
    total_frames: int,
    embedding_frame_stride: int,
    embedding_max_frames: int | None,
) -> int:
    if embedding_frame_stride <= 0:
        raise ValueError(
            "`embedding_frame_stride` must be positive, "
            f"got {embedding_frame_stride}."
        )
    if embedding_max_frames is not None and embedding_max_frames <= 0:
        raise ValueError(
            "`embedding_max_frames` must be positive when provided, "
            f"got {embedding_max_frames}."
        )

    resolved_stride = int(embedding_frame_stride)
    if embedding_max_frames is not None and total_frames > 0:
        resolved_stride = max(
            resolved_stride,
            int(math.ceil(total_frames / int(embedding_max_frames))),
        )
    return max(1, resolved_stride)


def _encode_frame_batch(
    batch_frames: list[np.ndarray],
    *,
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModel_v2,
    device: torch.device,
    clip_dtype: torch.dtype | None,
) -> torch.Tensor:
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

    with torch.inference_mode():
        _, image_latents = vision_model(pixel_values)
    return F.normalize(image_latents, dim=-1).to(dtype=torch.float32).cpu()


def _collect_frame_embeddings(
    cap: cv2.VideoCapture,
    *,
    total_frames: int,
    image_processor: CLIPImageProcessor,
    vision_model: CLIPVisionModel_v2,
    device: torch.device,
    clip_dtype: torch.dtype | None,
    batch_size: int,
    embedding_frame_stride: int,
    embedding_max_frames: int | None,
    max_side: int | None,
    qwen_factor: int,
) -> tuple[torch.Tensor, list[int], int]:
    if batch_size <= 0:
        raise ValueError(f"`clip_batch_size` must be positive, got {batch_size}.")

    resolved_stride = _resolve_embedding_downsample_stride(
        total_frames=total_frames,
        embedding_frame_stride=embedding_frame_stride,
        embedding_max_frames=embedding_max_frames,
    )
    frame_indices: list[int] = []
    embedding_chunks: list[torch.Tensor] = []
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []
    frame_idx = 0
    last_frame_index = max(total_frames - 1, 0)

    while True:
        should_sample = (
            frame_idx % resolved_stride == 0
            or frame_idx == last_frame_index
        )
        if should_sample:
            ok, frame_bgr = cap.read()
        else:
            ok = cap.grab()
            frame_bgr = None
        if not ok:
            break
        if not should_sample or frame_bgr is None:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # The CLIP processor resizes frames again for embedding, so Qwen-specific
        # shape alignment here only adds extra OpenCV work without helping VTCP
        # scoring. Keep the cheaper max-side resize during the embedding pass and
        # apply Qwen compatibility only when decoding the final selected frames.
        frame_rgb = _resize_frame(
            frame_rgb,
            max_side=max_side,
            ensure_qwen_compatibility=False,
            qwen_factor=qwen_factor,
        )
        batch_frames.append(_prepare_frame_array(frame_rgb))
        batch_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) >= batch_size:
            embedding_chunks.append(
                _encode_frame_batch(
                    batch_frames,
                    image_processor=image_processor,
                    vision_model=vision_model,
                    device=device,
                    clip_dtype=clip_dtype,
                )
            )
            frame_indices.extend(batch_indices)
            batch_frames = []
            batch_indices = []

    if batch_frames:
        embedding_chunks.append(
            _encode_frame_batch(
                batch_frames,
                image_processor=image_processor,
                vision_model=vision_model,
                device=device,
                clip_dtype=clip_dtype,
            )
        )
        frame_indices.extend(batch_indices)

    if not embedding_chunks:
        raise RuntimeError("No frames were decoded from the video.")

    return torch.cat(embedding_chunks, dim=0), frame_indices, resolved_stride


def _encode_query_embedding(
    query: str,
    *,
    tokenizer: Any,
    text_model: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        text_inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        embedding = text_model(**text_inputs)
    return F.normalize(embedding, dim=-1).to(dtype=torch.float32).cpu().squeeze(0)


def _compute_transition_scores(
    embeddings: torch.Tensor,
    *,
    similarity_metric: str,
) -> np.ndarray:
    metric = str(similarity_metric).strip().lower()
    if metric not in SUPPORTED_SIMILARITY_METRICS:
        available = ", ".join(sorted(SUPPORTED_SIMILARITY_METRICS))
        raise ValueError(
            f"Unsupported `similarity_metric`: {similarity_metric}. Available: {available}."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected frame embeddings with shape (T, D), "
            f"got {tuple(embeddings.shape)}."
        )
    if int(embeddings.shape[0]) == 1:
        return np.zeros(1, dtype=np.float32)

    current = embeddings[1:]
    previous = embeddings[:-1]
    if metric == "cos_sim":
        distances = 1.0 - (current * previous).sum(dim=-1)
    elif metric == "l1":
        distances = torch.abs(current - previous).sum(dim=-1)
    else:
        distances = torch.linalg.norm(current - previous, dim=-1)

    return torch.cat([distances[:1], distances], dim=0).cpu().numpy().astype(np.float32)


def _moving_average(
    scores: np.ndarray,
    *,
    window_size: int,
    warmup_frames: int = 0,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError(f"`sma_window` must be positive, got {window_size}.")
    if warmup_frames < 0:
        raise ValueError(
            f"`sma_warmup_frames` must be non-negative, got {warmup_frames}."
        )
    if scores.size == 0 or window_size == 1:
        return scores.astype(np.float32, copy=True)

    values = scores.astype(np.float32, copy=False)
    window = min(window_size, int(values.size))
    cumsum = np.cumsum(values, dtype=np.float64)
    smoothed = np.empty_like(values, dtype=np.float32)

    for index in range(int(values.size)):
        available = index + 1
        if warmup_frames > 0 and index < warmup_frames:
            progress = min(index + 1, warmup_frames) / float(warmup_frames)
            target_window = 1 + int((window - 1) * progress)
        else:
            target_window = window

        effective_window = max(1, min(target_window, available, window))
        start = available - effective_window
        total = cumsum[index] - (cumsum[start - 1] if start > 0 else 0.0)
        smoothed[index] = float(total / float(effective_window))

    return smoothed


def _traverse_with_dynamic_stride(
    control_scores: np.ndarray,
    *,
    initial_stride: int,
    stride_controller: Callable[[int, float], int],
) -> tuple[list[int], list[int], list[float]]:
    if initial_stride <= 0:
        raise ValueError(f"`initial_stride` must be positive, got {initial_stride}.")
    if control_scores.size == 0:
        return [], [], []

    visited_indices = [0]
    stride_history: list[int] = []
    control_score_history: list[float] = []
    current_index = 0
    current_stride = int(initial_stride)
    last_index = int(control_scores.size - 1)

    while current_index < last_index:
        control_score = float(control_scores[current_index])
        control_score_history.append(control_score)

        proposed_stride = int(stride_controller(current_stride, control_score))
        if proposed_stride <= 0:
            raise ValueError(
                "Stride controller must return a positive integer, "
                f"got {proposed_stride}."
            )

        # Apply the controller output immediately at the current position so
        # both increases and decreases affect the very next step.
        step_stride = proposed_stride
        stride_history.append(step_stride)

        next_index = min(last_index, current_index + step_stride)
        if next_index == current_index:
            break
        if next_index != visited_indices[-1]:
            visited_indices.append(next_index)

        current_index = next_index
        if current_index >= last_index:
            break

        current_stride = proposed_stride

    if visited_indices[-1] != last_index:
        visited_indices.append(last_index)
    return visited_indices, stride_history, control_score_history


def _resolve_final_selection_count(
    visited_count: int,
    *,
    top_ratio: float,
    min_selected_frames: int,
    max_selected_frames: int | None,
) -> int:
    if visited_count <= 0:
        return 0
    if top_ratio <= 0.0:
        raise ValueError(f"`top_ratio` must be positive, got {top_ratio}.")
    if min_selected_frames <= 0:
        raise ValueError(
            f"`min_selected_frames` must be positive, got {min_selected_frames}."
        )
    if max_selected_frames is not None and max_selected_frames <= 0:
        raise ValueError(
            f"`max_selected_frames` must be positive when provided, got {max_selected_frames}."
        )

    selected_count = max(
        int(math.ceil(visited_count * top_ratio)),
        int(min_selected_frames),
    )
    if max_selected_frames is not None:
        selected_count = min(selected_count, int(max_selected_frames))
    return max(1, min(selected_count, visited_count))


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


def _normalize_decoded_frames(frames: list[np.ndarray]) -> torch.Tensor:
    if not frames:
        raise ValueError("No frames were decoded for normalization.")

    base_height, base_width = frames[0].shape[:2]
    normalized_frames: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)

    return torch.from_numpy(np.stack(normalized_frames, axis=0))


def vtcp_sampling(
    video_path: str,
    *,
    query_file: str | None = None,
    query_text: str | None = None,
    top_ratio: float = 0.5,
    min_selected_frames: int = 4,
    max_selected_frames: int | None = None,
    similarity_metric: str = "cos_sim",
    sma_window: int = 5,
    sma_warmup_frames: int = 0,
    initial_stride: int = 4,
    stride_controller: Callable[[int, float], int] | None = None,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    clip_dtype: str | torch.dtype | None = None,
    clip_batch_size: int = 32,
    embedding_frame_stride: int = 1,
    embedding_max_frames: int | None = None,
    device: str | torch.device | None = None,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
    store_diagnostics: bool = False,
) -> FrameSelectionResult:
    query, query_source = _load_query_text(
        query_text=query_text,
        query_file=query_file,
    )
    resolved_device = _resolve_device(device)
    resolved_clip_dtype, clip_dtype_key = _resolve_clip_dtype(clip_dtype)
    if resolved_device.type == "cpu":
        resolved_clip_dtype = None
        clip_dtype_key = "default"

    image_processor, tokenizer, vision_model, text_model = _load_clip_components(
        clip_model_name,
        _resolve_device_key(resolved_device),
        clip_dtype_key,
    )

    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    decoded_video_path = str(transcoded_path) if transcoded_path is not None else video_path
    try:
        frame_embeddings, frame_indices, resolved_embedding_stride = _collect_frame_embeddings(
            cap,
            total_frames=total_frames,
            image_processor=image_processor,
            vision_model=vision_model,
            device=resolved_device,
            clip_dtype=resolved_clip_dtype,
            batch_size=clip_batch_size,
            embedding_frame_stride=embedding_frame_stride,
            embedding_max_frames=embedding_max_frames,
            max_side=max_side,
            qwen_factor=qwen_factor,
        )
        query_embedding = _encode_query_embedding(
            query,
            tokenizer=tokenizer,
            text_model=text_model,
            device=resolved_device,
        )
    finally:
        cap.release()

    try:
        transition_scores = _compute_transition_scores(
            frame_embeddings,
            similarity_metric=similarity_metric,
        )
        smoothed_scores = _moving_average(
            transition_scores,
            window_size=sma_window,
            warmup_frames=sma_warmup_frames,
        )
        control_scores = smoothed_scores
        resolved_controller = stride_controller or RenoStrideController()
        visited_positions, stride_history, visited_control_scores = _traverse_with_dynamic_stride(
            control_scores,
            initial_stride=initial_stride,
            stride_controller=resolved_controller,
        )
        visited_indices = [frame_indices[position] for position in visited_positions]

        selected_count = _resolve_final_selection_count(
            len(visited_indices),
            top_ratio=top_ratio,
            min_selected_frames=min_selected_frames,
            max_selected_frames=max_selected_frames,
        )
        visited_embeddings = frame_embeddings[visited_positions]
        visited_query_scores = torch.matmul(visited_embeddings, query_embedding)
        ranked_positions = torch.topk(
            visited_query_scores,
            k=selected_count,
            largest=True,
            sorted=True,
        ).indices.tolist()
        selected_indices = sorted(
            visited_indices[int(position)]
            for position in ranked_positions
        )

        decode_cap, _, _, _ = _open_video_for_sampling(decoded_video_path)
        try:
            selected_frames, decoded_selected_indices = _decode_frames_at_indices(
                decode_cap,
                np.asarray(selected_indices, dtype=int),
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
        finally:
            decode_cap.release()

        if len(selected_frames) != len(selected_indices):
            raise RuntimeError(
                "Failed to decode all selected frames: "
                f"expected={len(selected_indices)}, got={len(selected_frames)}."
            )

        frames = _normalize_decoded_frames(selected_frames)
        metadata: dict[str, Any] = {
            "video_path": video_path,
            "decoded_video_path": decoded_video_path,
            "sampling_method": "vtcp",
            "stride_controller": type(resolved_controller).__name__,
            "score_threshold": (
                float(getattr(resolved_controller, "score_threshold"))
                if hasattr(resolved_controller, "score_threshold")
                else None
            ),
            "sampled_indices": decoded_selected_indices,
            "selected_original_indices": decoded_selected_indices,
            "visited_indices": visited_indices,
            "num_frames": int(frames.shape[0]),
            "total_frames": total_frames,
            "fps": fps if fps > 0 else None,
            "frame_shape": list(frames.shape[1:]),
            "ensure_qwen_compatibility": ensure_qwen_compatibility,
            "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
            "similarity_metric": similarity_metric,
            "top_ratio": float(top_ratio),
            "sma_window": int(sma_window),
            "sma_warmup_frames": int(sma_warmup_frames),
            "min_selected_frames": int(min_selected_frames),
            "max_selected_frames": (
                int(max_selected_frames)
                if max_selected_frames is not None
                else None
            ),
            "initial_stride": int(initial_stride),
            "stride_history": stride_history,
            "query_source": query_source,
            "query_text": query,
            "clip_model_name": clip_model_name,
            "clip_dtype": clip_dtype_key,
            "clip_batch_size": int(clip_batch_size),
            "embedding_frame_stride": int(embedding_frame_stride),
            "embedding_max_frames": (
                int(embedding_max_frames)
                if embedding_max_frames is not None
                else None
            ),
            "effective_embedding_stride": int(resolved_embedding_stride),
            "embedded_frame_count": int(frame_embeddings.shape[0]),
            "visited_frame_count": len(visited_indices),
        }
        if store_diagnostics:
            metadata["embedded_frame_indices"] = list(frame_indices)
            metadata["raw_transition_scores"] = transition_scores.tolist()
            metadata["smoothed_transition_scores"] = smoothed_scores.tolist()
            metadata["control_signal_scores"] = control_scores.tolist()
            metadata["visited_query_scores"] = visited_query_scores.tolist()
            metadata["visited_control_scores"] = visited_control_scores

        return FrameSelectionResult(
            frames=frames,
            metadata=metadata,
        )
    finally:
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)


vtcp_sampling.preload = preload_vtcp_sampling
