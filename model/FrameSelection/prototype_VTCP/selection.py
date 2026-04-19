from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import Any, Callable
import torch.nn.functional as F

import cv2
import numpy as np
import torch

from model.base.selection import (
    FrameSelectionResult,
    QWEN_VISION_FACTOR,
    _open_video_for_sampling,
    _resize_frame,
)

FrameEmbeddingFn = Callable[[torch.Tensor], torch.Tensor]       # (frame_tensor) -> frame_embed
QueryEmbeddingFn = Callable[[str], torch.Tensor]                # (query_text)   -> query_embed


@dataclass(slots=True)
class VTCPScore:
    frame_index: int # frame index = m_k
    score: float # z_k 계산 결과 
    smoothed_score: float
    relevance: float # z_k에서 r_t [0,1]
    intrinsic_importance: float # z_k에서 i_k [0,1]
    novelty: float # z_k에서 n_k [0,1]
    threshold: float # z_k에 대한 threshold (threshold가 dynamic이라서 추적) 
    selected: bool # z_k >= threshold 여부로 나온, a_k = 1 | 0 결과 
    force_selected: bool # T_max로 강제 선택됐는지
    score_stride: float # clip 전 score 기반 stride
    budget_stride: int # clip 전 budget 기반 stride
    stride: int # 점수와 budget에 기반하여 나온 실제 stride_t+1 


def _read_query(query: str, query_file: str | None) -> str:
    query_text = query.strip()
    if query_text:
        return query_text

    if query_file is None:
        raise ValueError("`query` or `query_file` must provide a non-empty query.")

    query_path = Path(query_file).expanduser()
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    query_text = query_path.read_text(encoding="utf-8").strip()
    if not query_text:
        raise ValueError(f"Query file is empty: {query_path}")
    return query_text


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"`{name}` must be positive, got {value}.")


def _normalize_weights(
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[float, float, float]:
    if min(alpha, beta, gamma) < 0:
        raise ValueError(
            "`alpha`, `beta`, and `gamma` must be non-negative "
            f"(got alpha={alpha}, beta={beta}, gamma={gamma})."
        )
    total = alpha + beta + gamma
    _validate_positive("alpha + beta + gamma", total)
    return alpha / total, beta / total, gamma / total


def _cosine_01(a: torch.Tensor, b: torch.Tensor) -> float: # (1 + cos(a, b)) / 2
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    return float(((cos + 1.0) * 0.5).item())

def _clip01(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(float(value), lower, upper))

def _normalize_image_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0)


def _seek_and_read_frame(
    cap: cv2.VideoCapture,
    frame_index: int,
    *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    if not ok:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return _resize_frame(
        frame_rgb,
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )


def _normalize_selected_frames(frames: list[np.ndarray]) -> torch.Tensor:
    if not frames:
        raise RuntimeError("VTCP did not select any frames.")

    base_height, base_width = frames[0].shape[:2]
    normalized_frames = []
    for frame in frames:
        if frame.shape[:2] != (base_height, base_width):
            frame = cv2.resize(
                frame,
                (base_width, base_height),
                interpolation=cv2.INTER_AREA,
            )
        normalized_frames.append(frame)
    return torch.from_numpy(np.stack(normalized_frames, axis=0))

def _compute_novelty(
    frame_embed: torch.Tensor,
    selected_embeds: list[torch.Tensor],
    selected_indices: list[int],
    frame_index: int,
    *,
    tau_frames: float,   # 호출측에서 tau_seconds * fps 로 변환한 값
) -> float:
    if not selected_embeds:
        return 1.0

    penalties = []
    for selected_embed, selected_index in zip(selected_embeds, selected_indices, strict=True):
        similarity = _cosine_01(frame_embed, selected_embed)
        temporal_decay = math.exp(-abs(frame_index - selected_index) / max(tau_frames, 1e-6))
        penalties.append(similarity * temporal_decay)

    return _clip01(1.0 - max(penalties))



def _compute_score( #z_k 계산 
    *,
    relevance: float,
    intrinsic_importance: float,
    novelty: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta_i: float,
    delta_n: float,
) -> float:
    score = (
        alpha * relevance
        + beta * intrinsic_importance * (delta_i + (1.0 - delta_i) * relevance)
        + gamma * novelty * (delta_n + (1.0 - delta_n) * relevance)
    )
    return _clip01(score)


def _dynamic_threshold( # budget 기반해서 threshold 동적으로 조정 
    *,
    theta0: float,
    eta: float,
    budget: int,
    frame_index: int,
    total_frames: int,
    selected_count: int,
    threshold_min: float,
    threshold_max: float,
) -> float:
    expected_selected = budget * frame_index / max(total_frames, 1)
    lag = max(expected_selected - selected_count, 0.0) # 뽑힌 frame 수가 더 적을 때만 수행 
    return float(np.clip(theta0 - eta * lag, threshold_min, threshold_max))

def _stride_from_score_and_budget(# score 기반 stride와 budget 기반 stride 비교 
    *,
    smoothed_score: float,
    frame_index: int,
    total_frames: int,
    selected_count: int,
    budget: int,
    s_min: int,
    s_max: int,
    stride_power: float,
) -> tuple[int, float, int]:
    score_stride = s_min + (s_max - s_min) * (1.0 - smoothed_score) ** stride_power
    remaining_frames = max(total_frames - frame_index - 1, 0)
    remaining_budget = max(budget - selected_count, 1)
    budget_stride = max(1, remaining_frames // remaining_budget)
    clipped = np.clip(min(score_stride, budget_stride), s_min, s_max)
    final_stride = max(1, int(round(float(clipped))))
    return final_stride, float(score_stride), int(budget_stride)

def vtcp_select_from_pool(
    frame_embeds: torch.Tensor,
    query_embed: torch.Tensor,
    *,
    frame_indices: list[int] | None = None,
    num_frames: int = 8,
    alpha: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta_i: float = 0.25,
    delta_n: float = 0.25,
    rho: float = 0.9,
    tau_frames: float = 32.0,
    theta0: float = 0.6,
    eta: float = 0.05,
) -> tuple[list[int], list[dict[str, Any]]]:
    """
    Pre-embedded 후보풀에서 VTCP 점수만 평가하는 오프라인 헬퍼.

    `vtcp_sampling`과 달리 stride jump, score smoothing, T_max 강제선택을
    수행하지 않는다. 주어진 후보를 순차 방문하며 score/threshold 거동을 검증할 때 사용.
    """
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    _validate_positive("tau_frames", tau_frames)

    alpha, beta, gamma = _normalize_weights(alpha, beta, gamma)
    frame_embeds = torch.nn.functional.normalize(
        frame_embeds.to(dtype=torch.float32),
        dim=-1,
    )
    query_embed = torch.nn.functional.normalize(query_embed.to(dtype=torch.float32), dim=-1)
    if query_embed.ndim > 1:
        query_embed = query_embed.squeeze(0)

    total = int(frame_embeds.shape[0])
    indices = frame_indices if frame_indices is not None else list(range(total))
    if len(indices) != total:
        raise ValueError("`frame_indices` length must match `frame_embeds` length.")

    selected: list[int] = [] # selcted frame index 저장 
    selected_embeds: list[torch.Tensor] = [] # selcted frame embed결과 저장 
    running_mean: torch.Tensor | None = None # 현재까지의 평균 embed 
    records: list[dict[str, Any]] = []

    for local_idx, frame_index in enumerate(indices):
        embed = frame_embeds[local_idx]
        relevance = _cosine_01(embed, query_embed)
        intrinsic = (
            1.0
            if running_mean is None
            else _clip01(1.0 - _cosine_01(embed, running_mean))
        )
        novelty = _compute_novelty(
            embed,
            selected_embeds,
            selected,
            frame_index,
            tau_frames=tau_frames,
        )
        score = _compute_score(
            relevance=relevance,
            intrinsic_importance=intrinsic,
            novelty=novelty,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta_i=delta_i,
            delta_n=delta_n,
        )
        threshold = _dynamic_threshold(
            theta0=theta0,
            eta=eta,
            budget=num_frames,
            frame_index=local_idx,
            total_frames=total,
            selected_count=len(selected),
            threshold_min=0.0,
            threshold_max=1.0,
        )
        choose = len(selected) < num_frames and score >= threshold
        if choose:
            selected.append(frame_index)
            selected_embeds.append(embed.detach().clone())

        if running_mean is None:
            running_mean = embed.detach().clone()
        else:
            running_mean = rho * running_mean + (1.0 - rho) * embed

        records.append(
            {
                "frame_index": int(frame_index),
                "score": score,
                "relevance": relevance,
                "intrinsic_importance": intrinsic,
                "novelty": novelty,
                "threshold": threshold,
                "selected": choose,
            }
        )

        if len(selected) >= num_frames:
            break

    return selected, records




def vtcp_sampling(
    video_path: str,
    *,
    num_frames: int = 8,
    embed_frame_fn: FrameEmbeddingFn | None = None,     # CHANGED
    embed_query_fn: QueryEmbeddingFn | None = None,     # CHANGED
    query: str = "",
    query_file: str | None = None,
    alpha: float = 0.5,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta_i: float = 0.25,
    delta_n: float = 0.25,
    rho: float = 0.9,
    tau_seconds: float = 1.0,          # CHANGED: 초 단위, fps로 frame-scale 변환
    s_min: int = 1,
    s_max: int = 16,
    stride_power: float = 1.0,
    score_smoothing_lambda: float = 0.7,
    theta0: float = 0.6,
    eta: float = 0.05,                 # CHANGED: 0.03 -> 0.05
    threshold_min: float = 0.15,       # CHANGED: 0.0 -> 0.15
    threshold_max: float = 1.0,
    hysteresis_gap: float = 0.05,      # NEW: theta_on = theta_k, theta_off = theta_k - gap
    t_max: int | None = None,
    start_index: int = 0,
    max_steps: int | None = None,
    fill_to_budget: bool = True,
    duplicate_sim_threshold: float = 0.95,   # NEW: fill 단계 중복 필터
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    """
    VTCP frame selector (on-the-fly stride 기반).

    tau_seconds 는 '초' 단위로 지정하고, 내부적으로 fps를 곱해 frame-scale
    tau_frames 로 변환된다. fps 미확정 비디오는 fallback 30을 사용.
    """
    if embed_frame_fn is None or embed_query_fn is None:
        raise ValueError("`embed_frame_fn` and `embed_query_fn` are required.")
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    if s_min <= 0 or s_max <= 0:
        raise ValueError(f"`s_min` and `s_max` must be positive, got {s_min}, {s_max}.")
    if s_min > s_max:
        raise ValueError(f"`s_min` must be <= `s_max`, got {s_min} > {s_max}.")
    _validate_positive("stride_power", stride_power)
    _validate_positive("tau_temp", tau_seconds)
    if not 0.0 <= score_smoothing_lambda <= 1.0:
        raise ValueError(
            "`score_smoothing_lambda` must be in [0, 1], "
            f"got {score_smoothing_lambda}."
        )
    if t_max is not None and t_max <= 0:
        raise ValueError(f"`t_max` must be positive when provided, got {t_max}.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError(
            f"`max_steps` must be positive when provided, got {max_steps}."
        )
    if not 0.0 <= duplicate_sim_threshold <= 1.0:
        raise ValueError(
            "`duplicate_sim_threshold` must be in [0, 1], "
            f"got {duplicate_sim_threshold}."
        )

    alpha, beta, gamma = _normalize_weights(alpha, beta, gamma)
    query_text = _read_query(query, query_file)
    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)

    effective_fps = float(fps) if fps and fps > 0 else 30.0
    tau_frames = max(tau_seconds * effective_fps, 1.0)

    # 쿼리는 루프 밖에서 1회만 임베딩
    query_embed = torch.nn.functional.normalize(
        embed_query_fn(query_text).to(dtype=torch.float32), dim=-1,
    )
    if query_embed.ndim > 1:
        query_embed = query_embed.squeeze(0)

    selected_frames: list[np.ndarray] = []
    selected_indices: list[int] = []
    selected_embeds: list[torch.Tensor] = []
    # (score, index, frame, embed) — embed 보관
    visited_candidates: list[tuple[float, int, np.ndarray, torch.Tensor]] = []
    score_records: list[VTCPScore] = []
    running_mean: torch.Tensor | None = None
    smoothed_score = 0.0
    current_index = int(np.clip(start_index, 0, max(total_frames - 1, 0)))
    last_selected_index = -math.inf              # CHANGED: 첫 프레임 강제선택 방지
    last_action_selected = False                 # NEW: hysteresis 상태
    steps = 0
    max_allowed_steps = max_steps or (total_frames + s_min - 1) // s_min

    try:
        while current_index < total_frames and len(selected_indices) < num_frames:
            if steps >= max_allowed_steps:
                break
            frame = _seek_and_read_frame(
                cap, current_index,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            if frame is None:
                break

            # frame embedding만 매 step 계산 (query는 캐싱됨)
            raw_frame_embed = embed_frame_fn(_normalize_image_tensor(frame))
            frame_embed = torch.nn.functional.normalize(
                raw_frame_embed.to(dtype=torch.float32), dim=-1,
            ).squeeze(0)

            relevance = _cosine_01(frame_embed, query_embed)
            intrinsic = (
                1.0 if running_mean is None
                else _clip01(1.0 - _cosine_01(frame_embed, running_mean))
            )
            novelty = _compute_novelty(
                frame_embed, selected_embeds, selected_indices, current_index,
                tau_frames=tau_frames,
            )
            score = _compute_score(
                relevance=relevance, intrinsic_importance=intrinsic, novelty=novelty,
                alpha=alpha, beta=beta, gamma=gamma,
                delta_i=delta_i, delta_n=delta_n,
            )
            smoothed_score = (
                score if steps == 0
                else score_smoothing_lambda * smoothed_score
                     + (1.0 - score_smoothing_lambda) * score
            )
            threshold = _dynamic_threshold(
                theta0=theta0, eta=eta,
                budget=num_frames, frame_index=current_index,
                total_frames=total_frames, selected_count=len(selected_indices),
                threshold_min=threshold_min, threshold_max=threshold_max,
            )

            # Hysteresis: 직전이 선택이었다면 theta_off(낮은 쪽)로 평가
            effective_theta = (
                max(threshold - hysteresis_gap, threshold_min)
                if last_action_selected else threshold
            )
            force_select = (
                t_max is not None
                and math.isfinite(last_selected_index)
                and current_index - last_selected_index >= t_max
            )
            selected = score >= effective_theta or force_select

            if selected:
                selected_frames.append(frame)
                selected_indices.append(current_index)
                selected_embeds.append(frame_embed.detach().clone())
                last_selected_index = current_index
            last_action_selected = bool(selected)

            visited_candidates.append(
                (score, current_index, frame, frame_embed.detach().clone())
            )

            # EMA: 재정규화 제거 (modeling note 정확 일치)
            if running_mean is None:
                running_mean = frame_embed.detach().clone()
            else:
                running_mean = rho * running_mean + (1.0 - rho) * frame_embed

            stride, score_stride, budget_stride = _stride_from_score_and_budget(
                smoothed_score=smoothed_score,
                frame_index=current_index, total_frames=total_frames,
                selected_count=len(selected_indices), budget=num_frames,
                s_min=s_min, s_max=s_max, stride_power=stride_power,
            )
            score_records.append(VTCPScore(
                frame_index=current_index,
                score=score,
                smoothed_score=smoothed_score,
                relevance=relevance,
                intrinsic_importance=intrinsic,
                novelty=novelty,
                threshold=threshold,
                selected=selected,
                force_selected=force_select,
                score_stride=score_stride,
                budget_stride=budget_stride,
                stride=stride,
            ))
            current_index += stride
            steps += 1

        threshold_selected_indices = list(selected_indices)
        budget_filled_indices: list[int] = []

        if fill_to_budget and len(selected_indices) < min(num_frames, total_frames):
            chosen = set(selected_indices)
            for _, cand_idx, cand_frame, cand_embed in sorted(
                visited_candidates, key=lambda item: item[0], reverse=True,
            ):
                if cand_idx in chosen:
                    continue
                # 중복 필터: 이미 담긴 것과 너무 유사하면 넘김 
                if selected_embeds and max(
                    _cosine_01(cand_embed, e) for e in selected_embeds
                ) > duplicate_sim_threshold:
                    continue
                selected_frames.append(cand_frame)
                selected_indices.append(cand_idx)
                selected_embeds.append(cand_embed)
                budget_filled_indices.append(cand_idx)
                chosen.add(cand_idx)
                if len(selected_indices) >= min(num_frames, total_frames):
                    break
                
        if not selected_frames:
            fallback_index = min(max(start_index, 0), total_frames - 1)
            fallback_frame = _seek_and_read_frame(
                cap,
                fallback_index,
                max_side=max_side,
                ensure_qwen_compatibility=ensure_qwen_compatibility,
                qwen_factor=qwen_factor,
            )
            if fallback_frame is None:
                raise RuntimeError(f"No frames were decoded from video: {video_path}")
            selected_frames = [fallback_frame]
            selected_indices = [fallback_index]
    finally:
        cap.release()
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)

    sorted_pairs = sorted(
        zip(selected_indices, selected_frames, strict=True),
        key=lambda item: item[0],
    )
    selected_indices = [int(index) for index, _ in sorted_pairs[:num_frames]]
    selected_frames = [frame for _, frame in sorted_pairs[:num_frames]]
    video_tensor = _normalize_selected_frames(selected_frames)
    
    return FrameSelectionResult(
        frames=video_tensor,
        metadata={
            "method": "prototype_vtcp",
            "video_path": video_path,
            "decoded_video_path": str(transcoded_path) if transcoded_path is not None else video_path,
            "sampled_indices": selected_indices,
            "selected_original_indices": selected_indices,
            "threshold_selected_indices": sorted(threshold_selected_indices),
            "budget_filled_indices": sorted(budget_filled_indices),
            "num_frames": len(selected_indices),
            "total_frames": total_frames,
            "fps": fps if fps > 0 else None,
            "frame_shape": list(video_tensor.shape[1:]),
            "ensure_qwen_compatibility": ensure_qwen_compatibility,
            "qwen_factor": qwen_factor if ensure_qwen_compatibility else None,
            "query_text": query_text,
            "vtcp_params": {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "delta_i": delta_i, "delta_n": delta_n,
            "rho": rho,
            "tau_seconds": tau_seconds,
            "tau_frames_effective": tau_frames,       # NEW
            "effective_fps": effective_fps,           # NEW
            "s_min": s_min, "s_max": s_max,
            "stride_power": stride_power,
            "score_smoothing_lambda": score_smoothing_lambda,
            "theta0": theta0, "eta": eta,
            "threshold_min": threshold_min, "threshold_max": threshold_max,
            "hysteresis_gap": hysteresis_gap,         # NEW
            "t_max": t_max,
            "start_index": start_index,
            "max_steps": max_steps,
            "fill_to_budget": fill_to_budget,
            "duplicate_sim_threshold": duplicate_sim_threshold,  # NEW
            },
            "visited_count": len(score_records),
            "score_trace": [
            {
                "frame_index": r.frame_index,
                "score": r.score,
                "smoothed_score": r.smoothed_score,
                "relevance": r.relevance,
                "intrinsic_importance": r.intrinsic_importance,
                "novelty": r.novelty,
                "threshold": r.threshold,
                "selected": r.selected,
                "force_selected": r.force_selected,
                "score_stride": r.score_stride,
                "budget_stride": r.budget_stride,
                "stride": r.stride,
            }
            for r in score_records
        ],
        },
    )
