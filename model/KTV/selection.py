import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


DEFAULT_DINOV2_MODEL_ID = "facebook/dinov2-base"
PRELOAD_DINOV2_ON_IMPORT = os.getenv("KTV_PRELOAD_DINOV2_ON_IMPORT", "1") == "1"
PRELOAD_DINOV2_LOCAL_ONLY = os.getenv("KTV_PRELOAD_DINOV2_LOCAL_ONLY", "0") == "1"
PRELOAD_DINOV2_MODEL_ID = os.getenv("KTV_PRELOAD_DINOV2_MODEL_ID", DEFAULT_DINOV2_MODEL_ID)
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

DINOV2_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype

    try:
        return DINOV2_DTYPE_MAP[dtype]
    except KeyError as exc:
        available = ", ".join(sorted(DINOV2_DTYPE_MAP))
        raise ValueError(
            f"Unsupported DINOv2 dtype: {dtype}. Available: {available}"
        ) from exc


def _read_video_frames(
    video_path: str,
    max_side: int | None = 720,
    frame_stride: int | None = 1,
    frame_stride_ratio: float | None = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Failed to read total frame count.")

    if frame_stride_ratio is not None:
        if not 0.0 < frame_stride_ratio <= 1.0:
            cap.release()
            raise ValueError(
                "frame_stride_ratio must be in (0, 1], "
                f"but got {frame_stride_ratio}."
            )
        target_frames = max(1, int(np.ceil(total_frames * frame_stride_ratio)))
        frame_stride = max(1, int(np.ceil(total_frames / target_frames)))
    elif frame_stride is None:
        frame_stride = 1

    if frame_stride <= 0:
        cap.release()
        raise ValueError(f"frame_stride must be >= 1, but got {frame_stride}.")

    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if max_side is not None:
                h, w = frame_rgb.shape[:2]
                scale = min(max_side / max(h, w), 1.0)
                if scale < 1.0:
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (new_w, new_h),
                        interpolation=cv2.INTER_AREA,
                    )

            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames were extracted from video: {video_path}")

    return frames


@lru_cache(maxsize=4)
def _load_dinov2(
    model_id: str,
    local_files_only: bool,
    dtype: torch.dtype | None = None,
):
    processor = AutoImageProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    model.eval()
    return processor, model


def _preload_dinov2_on_import() -> None:
    try:
        _load_dinov2(
            model_id=PRELOAD_DINOV2_MODEL_ID,
            local_files_only=PRELOAD_DINOV2_LOCAL_ONLY,
        )
    except Exception as exc:
        warnings.warn(
            "Failed to preload DINOv2 during module import. "
            "The model will be loaded lazily on first use instead. "
            f"reason={exc}",
            stacklevel=2,
        )


@lru_cache(maxsize=2)
def _load_clip(
    model_id: str,
):
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=False,
    )
    model = AutoModel.from_pretrained(
        model_id,
        local_files_only=False,
    )
    model.eval()

    if not hasattr(model, "get_text_features") or not hasattr(model, "get_image_features"):
        raise TypeError(
            "Loaded model does not expose `get_text_features` / `get_image_features`."
        )

    return processor, model


def preload_dinov2_model(
    model_id: str = DEFAULT_DINOV2_MODEL_ID,
    *,
    local_files_only: bool = False,
    dtype: str | torch.dtype | None = None,
) -> None:
    """DINOv2 모델을 캐시에 미리 로드한다."""
    model_dtype = _resolve_torch_dtype(dtype)
    _load_dinov2(
        model_id=model_id,
        local_files_only=local_files_only,
        dtype=model_dtype,
    )


def preload_clip_model(
    model_id: str = DEFAULT_CLIP_MODEL_ID,
) -> None:
    """CLIP 모델을 캐시에 미리 로드한다."""
    _load_clip(model_id=model_id)


def _move_inputs_to_device(
    inputs: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue

        if (
            dtype is not None
            and torch.is_tensor(value)
            and value.is_floating_point()
        ):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)

    return moved


def _as_feature_tensor(
    value: Any,
    *,
    label: str,
) -> torch.Tensor:
    if torch.is_tensor(value):
        return value

    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        candidate = getattr(value, attr, None)
        if torch.is_tensor(candidate):
            return candidate

    last_hidden_state = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(last_hidden_state):
        if last_hidden_state.ndim < 2:
            raise ValueError(
                f"{label} last_hidden_state must have ndim >= 2, "
                f"but got shape={tuple(last_hidden_state.shape)}."
            )
        return last_hidden_state[:, 0, ...]

    if isinstance(value, (tuple, list)) and value:
        first = value[0]
        if torch.is_tensor(first):
            return first

    raise TypeError(
        f"{label} must be a tensor-like output, but got {type(value)!r}."
    )


def _video_tensor_to_images(video_tensor: torch.Tensor) -> list[np.ndarray]:
    if video_tensor.ndim != 4:
        raise ValueError(
            "Expected sampled frames with shape (T, H, W, C), "
            f"but got {tuple(video_tensor.shape)}."
        )

    frames = video_tensor.detach().cpu()
    if frames.dtype != torch.uint8:
        if (
            frames.is_floating_point()
            and frames.numel() > 0
            and float(frames.max().item()) <= 1.0
        ):
            frames = frames * 255.0
        frames = frames.clamp(0, 255).to(torch.uint8)

    return [frame.numpy() for frame in frames]


def _resolve_query_path(query_file: str) -> Path:
    path = Path(query_file).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / path).resolve()


def _load_query_text(
    *,
    query: str | None = None,
    query_file: str | None = None,
    prompt: str | None = None,
    use_prompt_as_query: bool = False,
) -> str | None:
    if query is not None:
        query_text = query.strip()
        if query_text:
            return query_text

    if query_file is not None:
        query_path = _resolve_query_path(query_file)
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")

        query_text = query_path.read_text(encoding="utf-8").strip()
        if not query_text:
            raise ValueError(f"Query file is empty: {query_path}")
        return query_text

    if use_prompt_as_query and prompt is not None:
        prompt_text = prompt.strip()
        if prompt_text:
            return prompt_text

    return None


def _compute_clip_frame_relevance(
    *,
    video_tensor: torch.Tensor,
    query_text: str,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    dtype: str | torch.dtype | None = None,
    device: str | torch.device | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, but got {batch_size}.")

    model_dtype = _resolve_torch_dtype(dtype)
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if target_device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        model_dtype = torch.float32

    processor, model = _load_clip(
        model_id=model_id,
    )
    if model_dtype is not None:
        model = model.to(device=target_device, dtype=model_dtype)
    else:
        model = model.to(device=target_device)

    frames = _video_tensor_to_images(video_tensor)
    if not frames:
        raise ValueError("video_tensor must contain at least one frame.")

    text_inputs = processor(
        text=[query_text],
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    text_inputs = _move_inputs_to_device(
        text_inputs,
        device=target_device,
        dtype=model_dtype,
    )

    with torch.inference_mode():
        text_features = _as_feature_tensor(
            model.get_text_features(**text_inputs),
            label="text_features",
        )
    if text_features.ndim == 1:
        text_features = text_features.unsqueeze(0)
    text_features = F.normalize(text_features.float(), dim=-1)[0]

    similarities: list[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        image_inputs = processor(
            images=frames[start:start + batch_size],
            return_tensors="pt",
        )
        image_inputs = _move_inputs_to_device(
            image_inputs,
            device=target_device,
            dtype=model_dtype,
        )
        with torch.inference_mode():
            image_features = _as_feature_tensor(
                model.get_image_features(**image_inputs),
                label="image_features",
            )

        image_features = F.normalize(image_features.float(), dim=-1)
        batch_similarity = torch.matmul(image_features, text_features)
        similarities.append(batch_similarity.cpu())

    return torch.cat(similarities, dim=0)


def _extract_frame_features(
    frames: list[np.ndarray],
    model_id: str,
    batch_size: int,
    device: torch.device,
    local_files_only: bool,
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, but got {batch_size}.")

    model_dtype = _resolve_torch_dtype(dtype)
    processor, model = _load_dinov2(
        model_id=model_id,
        local_files_only=local_files_only,
        dtype=model_dtype,
    )
    model = model.to(device=device, dtype=model_dtype)

    features: list[torch.Tensor] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt")
        inputs = {
            key: (
                value.to(device=device, dtype=model_dtype)
                if hasattr(value, "to") and torch.is_tensor(value) and value.is_floating_point()
                else value.to(device)
                if hasattr(value, "to")
                else value
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_features = getattr(outputs, "pooler_output", None)
        if batch_features is None:
            batch_features = outputs.last_hidden_state[:, 0]

        features.append(F.normalize(batch_features, dim=-1).cpu())

    return torch.cat(features, dim=0)


def _select_representative_indices(
    features: np.ndarray,
    num_keyframes: int,
    random_state: int,
    num_threads: int | None = None,
) -> list[int]:
    if num_keyframes <= 0:
        raise ValueError(f"num_keyframes must be >= 1, but got {num_keyframes}.")
    if num_threads is not None and num_threads <= 0:
        raise ValueError(f"num_threads must be >= 1, but got {num_threads}.")

    num_frames = features.shape[0]
    if num_frames <= num_keyframes:
        return list(range(num_frames))

    kmeans = KMeans(
        n_clusters=num_keyframes,
        random_state=random_state,
        n_init="auto",
    )
    with threadpool_limits(limits=num_threads, user_api="blas"):
        labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    selected_indices: list[int] = []
    for cluster_idx in range(num_keyframes):
        member_indices = np.flatnonzero(labels == cluster_idx)
        member_features = features[member_indices]
        centroid = centroids[cluster_idx]
        distances = np.linalg.norm(member_features - centroid, axis=1)
        selected_indices.append(int(member_indices[np.argmin(distances)]))

    return sorted(selected_indices)


def _frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    base_h, base_w = frames[0].shape[:2]
    normalized_frames: list[np.ndarray] = []

    for frame in frames:
        if frame.shape[:2] != (base_h, base_w):
            frame = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_AREA)
        normalized_frames.append(frame)

    return torch.from_numpy(np.stack(normalized_frames, axis=0))


def _split_patch_and_cls_tokens(
    token_features: torch.Tensor,
    cls_tokens: torch.Tensor | None = None,
    has_cls_token: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_features.ndim != 3:
        raise ValueError(
            "token_features must have shape (T, L, D) or (T, L+1, D), "
            f"but got {tuple(token_features.shape)}."
        )

    if has_cls_token:
        if token_features.shape[1] < 2:
            raise ValueError(
                "token_features must contain at least one cls token and one patch token."
            )
        return token_features[:, 1:, :], token_features[:, 0, :]

    if cls_tokens is None:
        raise ValueError("cls_tokens must be provided when has_cls_token=False.")

    if cls_tokens.ndim == 3:
        if cls_tokens.shape[1] != 1:
            raise ValueError(
                "cls_tokens with 3 dimensions must have shape (T, 1, D), "
                f"but got {tuple(cls_tokens.shape)}."
            )
        cls_tokens = cls_tokens[:, 0, :]
    elif cls_tokens.ndim != 2:
        raise ValueError(
            "cls_tokens must have shape (T, D) or (T, 1, D), "
            f"but got {tuple(cls_tokens.shape)}."
        )

    if cls_tokens.shape[0] != token_features.shape[0]:
        raise ValueError("token_features and cls_tokens must have the same frame count.")

    if cls_tokens.shape[-1] != token_features.shape[-1]:
        raise ValueError("token_features and cls_tokens must share the same hidden size.")

    return token_features, cls_tokens


def _minmax_normalize(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    score_min = scores.amin(dim=-1, keepdim=True)
    score_max = scores.amax(dim=-1, keepdim=True)
    denom = score_max - score_min
    normalized = (scores - score_min) / denom.clamp_min(eps)
    return torch.where(denom > eps, normalized, torch.zeros_like(normalized))


def _resolve_keep_ratios(
    num_frames: int,
    beta: float | list[float] | tuple[float, ...] | torch.Tensor,
    frame_relevance: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(beta, (float, int)):
        keep_ratios = torch.full((num_frames,), float(beta), dtype=torch.float32)
    else:
        keep_ratios = torch.as_tensor(beta, dtype=torch.float32)
        if keep_ratios.ndim != 1 or keep_ratios.shape[0] != num_frames:
            raise ValueError(
                "beta must be a scalar or a 1D sequence with one entry per frame."
            )

        if frame_relevance is not None:
            relevance = torch.as_tensor(frame_relevance, dtype=torch.float32)
            if relevance.ndim != 1 or relevance.shape[0] != num_frames:
                raise ValueError(
                    "frame_relevance must be a 1D tensor with one score per frame."
                )

            ranked_frames = torch.argsort(relevance, descending=True)
            ranked_betas = torch.sort(keep_ratios, descending=True).values
            reordered = torch.empty_like(ranked_betas)
            reordered[ranked_frames] = ranked_betas
            keep_ratios = reordered

    return keep_ratios.clamp(0.0, 1.0)


def _resolve_rank_beta_sequence(
    *,
    num_frames: int,
    beta: float | list[float] | tuple[float, ...] | torch.Tensor,
    beta_specific: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    start: float = 1.0,
    floor: float = 0.0,
) -> torch.Tensor:
    """프레임 순위용 beta 시퀀스를 만든다.

    우선순위:
    1) beta_specific (명시적 시퀀스)
    2) beta 스칼라 -> 1, 1-beta, 1-2*beta ... 형태의 등차 수열
    3) beta 시퀀스(하위호환)
    """
    if beta_specific is not None:
        sequence = torch.as_tensor(beta_specific, dtype=torch.float32)
    elif isinstance(beta, (float, int)):
        step = float(beta)
        if step <= 0.0:
            raise ValueError(f"beta must be > 0.0 when used as step, but got {step}.")
        ranks = torch.arange(num_frames, dtype=torch.float32)
        sequence = (float(start) - step * ranks).clamp(min=float(floor), max=1.0)
    else:
        sequence = torch.as_tensor(beta, dtype=torch.float32)

    if sequence.ndim != 1:
        raise ValueError("beta sequence must be 1D.")
    if sequence.shape[0] < num_frames:
        raise ValueError(
            "beta sequence is shorter than sampled frames: "
            f"len(beta)={sequence.shape[0]}, frames={num_frames}."
        )
    if sequence.shape[0] > num_frames:
        sequence = sequence[:num_frames]

    return sequence.clamp(0.0, 1.0)


def _pad_selected_tokens(
    selected_tokens: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not selected_tokens:
        raise ValueError("selected_tokens must not be empty.")

    num_frames = len(selected_tokens)
    hidden_dim = selected_tokens[0].shape[-1]
    max_keep = max(tokens.shape[0] for tokens in selected_tokens)

    padded = selected_tokens[0].new_zeros((num_frames, max_keep, hidden_dim))
    mask = torch.zeros((num_frames, max_keep), dtype=torch.bool, device=padded.device)

    for frame_idx, tokens in enumerate(selected_tokens):
        keep_count = tokens.shape[0]
        padded[frame_idx, :keep_count] = tokens
        mask[frame_idx, :keep_count] = True

    return padded, mask


def dinov2_kmeans_keyframe_selection(
    video_path: str,
    num_keyframes: int = 8,
    max_side: int | None = 720,
    frame_stride: int | None = 1,
    frame_stride_ratio: float | None = None,
    model_id: str = "facebook/dinov2-base",
    dtype: str | torch.dtype | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    random_state: int = 0,
    num_threads: int | None = None,
    local_files_only: bool = False,
) -> torch.Tensor:
    """KTV 논문 방식의 DINOv2 + KMeans keyframe selection.

    전체 프레임에서 DINOv2 frame-level feature를 추출한 뒤, KMeans centroid에
    가장 가까운 프레임을 keyframe으로 선택한다. 반환 텐서는 시간순으로
    정렬된 `(T, H, W, C)` 형태의 `torch.uint8` 텐서다.
    """
    frames = _read_video_frames(
        video_path=video_path,
        max_side=max_side,
        frame_stride=frame_stride,
        frame_stride_ratio=frame_stride_ratio,
    )

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    features = _extract_frame_features(
        frames=frames,
        model_id=model_id,
        batch_size=batch_size,
        device=target_device,
        local_files_only=local_files_only,
        dtype=dtype,
    )

    selected_indices = _select_representative_indices(
        features=features.numpy(),
        num_keyframes=num_keyframes,
        random_state=random_state,
        num_threads=num_threads,
    )
    selected_frames = [frames[idx] for idx in selected_indices]
    return _frames_to_tensor(selected_frames)


def remove_duplicate_frames(
    video_path: str,
    num_keyframes: int = 8,
    max_side: int | None = 720,
    frame_stride: int | None = 1,
    frame_stride_ratio: float | None = None,
    model_id: str = "facebook/dinov2-base",
    dtype: str | torch.dtype | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    random_state: int = 0,
    num_threads: int | None = None,
    local_files_only: bool = False,
) -> torch.Tensor:
    """중복 프레임을 줄이기 위한 DINOv2 기반 keyframe selector."""
    return dinov2_kmeans_keyframe_selection(
        video_path=video_path,
        num_keyframes=num_keyframes,
        max_side=max_side,
        frame_stride=frame_stride,
        frame_stride_ratio=frame_stride_ratio,
        model_id=model_id,
        dtype=dtype,
        batch_size=batch_size,
        device=device,
        random_state=random_state,
        num_threads=num_threads,
        local_files_only=local_files_only,
    )


def key_patch_selection(
    token_features: torch.Tensor,
    cls_tokens: torch.Tensor | None = None,
    *,
    has_cls_token: bool = True,
    keep_cls_token: bool = True,
    alpha: float = 0.8,
    beta: float | list[float] | tuple[float, ...] | torch.Tensor = 0.25,
    frame_relevance: torch.Tensor | None = None,
    q_proj: torch.Tensor | None = None,
    k_proj: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """KTV 논문의 key patch selection을 텐서 레벨에서 수행한다.

    Args:
        token_features: `(T, L+1, D)` 또는 `(T, L, D)` 형태의 토큰 특징.
        cls_tokens: `has_cls_token=False`일 때 사용하는 `(T, D)` 또는 `(T, 1, D)` cls 토큰.
        has_cls_token: `True`이면 `token_features[:, 0]`을 cls 토큰으로 간주한다.
        keep_cls_token: `True`이면 cls 토큰을 항상 유지하고 selected token에 포함한다.
        alpha: importance / redundancy 결합 비율.
        beta: frame별 patch 유지 비율. 스칼라 또는 frame 수 길이의 시퀀스.
        frame_relevance: 질문-프레임 relevance 점수. `beta`가 시퀀스일 때 높은 점수 프레임에
            큰 beta가 배정된다.
        q_proj: importance 계산용 query projection. shape `(D, D_q)` 또는 `(D, D)`.
        k_proj: importance 계산용 key projection. shape `(D, D_q)` 또는 `(D, D)`.

    Returns:
        selected_tokens:
            frame별 선택 patch 토큰 리스트
        selected_tokens_padded:
            `(T, max_keep, D)` 패딩된 선택 토큰
        selected_mask:
            `selected_tokens_padded`의 유효 위치 마스크
        selected_indices:
            frame별 선택된 patch 인덱스 리스트
        scores / importance_scores / redundancy_scores:
            frame별 patch 점수 텐서 `(T, L)`
        keep_ratios / keep_counts:
            frame별 유지 비율과 유지 개수
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], but got {alpha}.")

    patch_tokens, cls_token_features = _split_patch_and_cls_tokens(
        token_features=token_features,
        cls_tokens=cls_tokens,
        has_cls_token=has_cls_token,
    )

    num_frames, num_patches, hidden_dim = patch_tokens.shape
    keep_ratios = _resolve_keep_ratios(
        num_frames=num_frames,
        beta=beta,
        frame_relevance=frame_relevance,
    ).to(device=patch_tokens.device)

    keep_counts = torch.ceil(keep_ratios * num_patches).to(dtype=torch.long)
    keep_counts = keep_counts.clamp(min=1, max=num_patches)

    query_tokens = cls_token_features
    key_tokens = patch_tokens
    if q_proj is not None:
        query_tokens = query_tokens @ q_proj.to(device=query_tokens.device, dtype=query_tokens.dtype)
    if k_proj is not None:
        key_tokens = key_tokens @ k_proj.to(device=key_tokens.device, dtype=key_tokens.dtype)

    importance_logits = torch.einsum(
        "td,tld->tl",
        query_tokens,
        key_tokens,
    ) / float(np.sqrt(query_tokens.shape[-1]))
    importance_scores = torch.softmax(importance_logits, dim=-1)

    normalized_patches = F.normalize(patch_tokens, dim=-1)
    similarity_matrix = torch.einsum(
        "tld,tmd->tlm",
        normalized_patches,
        normalized_patches,
    )
    if num_patches > 1:
        eye = torch.eye(num_patches, device=patch_tokens.device, dtype=patch_tokens.dtype)
        redundancy_scores = (similarity_matrix.sum(dim=-1) - eye.sum(dim=-1)) / (num_patches - 1)
    else:
        redundancy_scores = torch.zeros(
            (num_frames, num_patches),
            dtype=patch_tokens.dtype,
            device=patch_tokens.device,
        )

    importance_norm = _minmax_normalize(importance_scores)
    redundancy_norm = _minmax_normalize(redundancy_scores)
    final_scores = alpha * importance_norm + (1.0 - alpha) * (1.0 - redundancy_norm)

    selected_indices: list[torch.Tensor] = []
    selected_patch_indices: list[torch.Tensor] = []
    selected_tokens: list[torch.Tensor] = []
    for frame_idx in range(num_frames):
        frame_keep = int(keep_counts[frame_idx].item())
        frame_indices = torch.topk(
            final_scores[frame_idx],
            k=frame_keep,
            largest=True,
            sorted=False,
        ).indices
        frame_indices = torch.sort(frame_indices).values

        selected_patch_indices.append(frame_indices)
        frame_tokens = patch_tokens[frame_idx, frame_indices]

        if has_cls_token and keep_cls_token:
            cls_index = torch.zeros(
                1,
                dtype=torch.long,
                device=frame_indices.device,
            )
            full_indices = torch.cat([cls_index, frame_indices + 1], dim=0)
            frame_tokens = torch.cat(
                [cls_token_features[frame_idx].unsqueeze(0), frame_tokens],
                dim=0,
            )
        elif has_cls_token:
            full_indices = frame_indices + 1
        else:
            full_indices = frame_indices

        selected_indices.append(full_indices)
        selected_tokens.append(frame_tokens)

    padded_tokens, selected_mask = _pad_selected_tokens(selected_tokens)
    return {
        "selected_tokens": selected_tokens,
        "selected_tokens_padded": padded_tokens,
        "selected_mask": selected_mask,
        "selected_indices": selected_indices,
        "selected_patch_indices": selected_patch_indices,
        "scores": final_scores,
        "importance_scores": importance_scores,
        "redundancy_scores": redundancy_scores,
        "importance_norm": importance_norm,
        "redundancy_norm": redundancy_norm,
        "keep_ratios": keep_ratios,
        "keep_counts": keep_counts,
    }


def query_conditioned_key_patch_selection(
    token_features: torch.Tensor,
    cls_tokens: torch.Tensor | None = None,
    *,
    has_cls_token: bool = True,
    keep_cls_token: bool = True,
    alpha: float = 0.8,
    beta: float | list[float] | tuple[float, ...] | torch.Tensor = 0.25,
    beta_specific: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    frame_relevance: torch.Tensor | None = None,
    q_proj: torch.Tensor | None = None,
    k_proj: torch.Tensor | None = None,
    query: str | None = None,
    query_file: str | None = None,
    prompt: str | None = None,
    video_tensor: torch.Tensor | None = None,
    use_prompt_as_query: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL_ID,
    clip_dtype: str | torch.dtype | None = None,
    clip_device: str | torch.device | None = None,
    clip_batch_size: int = 16,
    beta_start: float = 1.0,
    beta_floor: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    """쿼리-프레임(CLIP) 관련도 순위로 beta 집합을 프레임에 배정한다.

    - beta_specific: 명시적 내림차순 비율 리스트
    - beta(스칼라): 1, 1-beta, 1-2*beta ... 등차수열 자동 생성
    """
    resolved_relevance = None if frame_relevance is None else torch.as_tensor(frame_relevance)
    if resolved_relevance is None:
        query_text = _load_query_text(
            query=query,
            query_file=query_file,
            prompt=prompt,
            use_prompt_as_query=use_prompt_as_query,
        )
        if query_text is not None:
            if video_tensor is None:
                raise ValueError(
                    "video_tensor is required to compute query-conditioned frame relevance."
                )
            resolved_relevance = _compute_clip_frame_relevance(
                video_tensor=video_tensor,
                query_text=query_text,
                model_id=clip_model_id,
                dtype=clip_dtype,
                device=clip_device,
                batch_size=clip_batch_size,
            )

    num_frames = int(token_features.shape[0])
    effective_beta = _resolve_rank_beta_sequence(
        num_frames=num_frames,
        beta=beta,
        beta_specific=beta_specific,
        start=beta_start,
        floor=beta_floor,
    )
    effective_frame_relevance = resolved_relevance

    outputs = key_patch_selection(
        token_features=token_features,
        cls_tokens=cls_tokens,
        has_cls_token=has_cls_token,
        keep_cls_token=keep_cls_token,
        alpha=alpha,
        beta=effective_beta,
        frame_relevance=effective_frame_relevance,
        q_proj=q_proj,
        k_proj=k_proj,
    )
    if resolved_relevance is not None:
        outputs["frame_relevance"] = resolved_relevance.detach().cpu()
    return outputs


__all__ = [
    "dinov2_kmeans_keyframe_selection",
    "key_patch_selection",
    "preload_clip_model",
    "preload_dinov2_model",
    "query_conditioned_key_patch_selection",
    "remove_duplicate_frames",
]


if PRELOAD_DINOV2_ON_IMPORT:
    _preload_dinov2_on_import()
