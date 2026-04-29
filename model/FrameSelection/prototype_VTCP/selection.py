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


def _as_rgb_uint8_frames(frames: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        frames_array = frames.detach().cpu().numpy()
    else:
        frames_array = np.asarray(frames)

    if frames_array.ndim != 4:
        raise ValueError(
            "`frames` must have shape (T, H, W, C) or (T, C, H, W), "
            f"got {frames_array.shape}."
        )
    if frames_array.shape[-1] not in (1, 3, 4) and frames_array.shape[1] in (1, 3, 4):
        frames_array = np.transpose(frames_array, (0, 2, 3, 1))
    if frames_array.shape[-1] == 1:
        frames_array = np.repeat(frames_array, 3, axis=-1)
    elif frames_array.shape[-1] == 4:
        frames_array = frames_array[..., :3]

    if np.issubdtype(frames_array.dtype, np.floating):
        max_value = float(np.nanmax(frames_array)) if frames_array.size else 1.0
        if max_value <= 1.0:
            frames_array = frames_array * 255.0
    return np.clip(frames_array, 0, 255).astype(np.uint8)


def _draw_polyline(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dashed: bool = False,
) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:], strict=False):
        if not dashed:
            cv2.line(canvas, start, end, color, thickness, lineType=cv2.LINE_AA)
            continue

        x1, y1 = start
        x2, y2 = end
        length = max(int(math.hypot(x2 - x1, y2 - y1)), 1)
        dash = 10
        gap = 7
        for offset in range(0, length, dash + gap):
            ratio_a = offset / length
            ratio_b = min(offset + dash, length) / length
            ax = int(round(x1 + (x2 - x1) * ratio_a))
            ay = int(round(y1 + (y2 - y1) * ratio_a))
            bx = int(round(x1 + (x2 - x1) * ratio_b))
            by = int(round(y1 + (y2 - y1) * ratio_b))
            cv2.line(canvas, (ax, ay), (bx, by), color, thickness, lineType=cv2.LINE_AA)


def _draw_star(
    canvas: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    outline: tuple[int, int, int],
) -> None:
    cx, cy = center
    vertices: list[tuple[int, int]] = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        vertices.append((int(round(cx + r * math.cos(angle))), int(round(cy + r * math.sin(angle)))))
    pts = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [pts], isClosed=True, color=outline, thickness=2, lineType=cv2.LINE_AA)


def visualize_vtcp_selection(
    frame_selection: FrameSelectionResult,
    output_path: str | Path | None = None,
    *,
    width: int = 1800,
    include_frames: bool = True,
    max_thumbnail_height: int = 140,
    x_axis: str = "visited",
    stride_axis_max: float | None = None,
) -> np.ndarray:
    """
    VTCP 선택 결과를 시각화한다.

    x_axis="visited"는 실제 방문한 프레임 범위만 확대해서 보여준다.
    x_axis="video"로 지정하면 전체 비디오 프레임 범위를 기준으로 그린다.
    """
    metadata = frame_selection.metadata
    trace = metadata.get("score_trace") or []
    if not trace:
        raise ValueError("`frame_selection.metadata['score_trace']` is empty.")

    selected_indices = [int(index) for index in metadata.get("sampled_indices", [])]
    threshold_selected = {
        int(index) for index in metadata.get("threshold_selected_indices", [])
    }
    budget_filled = {int(index) for index in metadata.get("budget_filled_indices", [])}
    force_selected = {
        int(record["frame_index"])
        for record in trace
        if bool(record.get("force_selected", False))
    }

    plot_width = int(max(width, 900))
    margin_left = 85
    margin_right = 45
    top_margin = 142
    panel_gap = 70
    score_height = 420
    stride_height = 210
    frame_gap = 18
    thumb_label_height = 34
    use_frames = include_frames and frame_selection.frames is not None
    thumb_height = int(max_thumbnail_height) if use_frames else 0
    height = (
        top_margin
        + score_height
        + panel_gap
        + stride_height
        + (frame_gap + thumb_height + thumb_label_height if use_frames else 30)
        + 40
    )
    canvas = np.full((height, plot_width, 3), 255, dtype=np.uint8)

    colors = {
        "axis": (92, 105, 125),
        "grid": (226, 231, 238),
        "text": (50, 63, 84),
        "score": (42, 107, 232),
        "smoothed": (116, 164, 248),
        "threshold": (35, 150, 112),
        "visited": (255, 162, 34),
        "selected": (232, 50, 49),
        "budget": (142, 91, 230),
        "stride": (11, 152, 114),
        "force": (255, 111, 0),
    }

    def put_text(
        text: str,
        org: tuple[int, int],
        scale: float = 0.65,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color or colors["text"],
            thickness,
            lineType=cv2.LINE_AA,
        )

    frame_indices = [int(record["frame_index"]) for record in trace]
    x_min = min(frame_indices)
    visited_x_max = max(frame_indices + selected_indices)
    if x_axis == "video":
        x_max = max(
            int(metadata.get("total_frames") or visited_x_max),
            visited_x_max,
            x_min + 1,
        )
    elif x_axis == "visited":
        pad = max(1, int(round((visited_x_max - x_min) * 0.04)))
        x_max = max(visited_x_max + pad, x_min + 1)
    else:
        raise ValueError("`x_axis` must be either 'visited' or 'video'.")
    plot_left = margin_left
    plot_right = plot_width - margin_right

    def map_x(frame_index: int) -> int:
        ratio = (frame_index - x_min) / max(x_max - x_min, 1)
        return int(round(plot_left + ratio * (plot_right - plot_left)))

    def map_y(value: float, top: int, panel_height: int, y_min: float, y_max: float) -> int:
        ratio = (float(value) - y_min) / max(y_max - y_min, 1e-6)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return int(round(top + panel_height - ratio * panel_height))

    def draw_axes(top: int, panel_height: int, *, y_min: float, y_max: float, y_label: str) -> None:
        cv2.rectangle(
            canvas,
            (plot_left, top),
            (plot_right, top + panel_height),
            colors["grid"],
            1,
            lineType=cv2.LINE_AA,
        )
        for tick in range(6):
            value = y_min + (y_max - y_min) * tick / 5
            y = map_y(value, top, panel_height, y_min, y_max)
            cv2.line(canvas, (plot_left, y), (plot_right, y), colors["grid"], 1, lineType=cv2.LINE_AA)
            put_text(f"{value:.1f}", (18, y + 5), 0.52, colors["axis"])
        for tick in range(6):
            frame_index = int(round(x_min + (x_max - x_min) * tick / 5))
            x = map_x(frame_index)
            cv2.line(canvas, (x, top), (x, top + panel_height), colors["grid"], 1, lineType=cv2.LINE_AA)
            put_text(str(frame_index), (x - 18, top + panel_height + 28), 0.52, colors["axis"])
        put_text(y_label, (14, top - 14), 0.62, colors["axis"], 2)
        put_text("Original frame index", (plot_left + (plot_right - plot_left) // 2 - 120, top + panel_height + 55), 0.62, colors["axis"], 2)

    title = "VTCP Temporal Timeline"
    query = str(metadata.get("query_text", "") or "")
    params = metadata.get("vtcp_params", {}) or {}
    effective_stride = params.get("s_max")
    put_text(title, (plot_width // 2 - 175, 42), 0.95, colors["text"], 2)
    if query:
        put_text(f"query={query}", (plot_width // 2 - 115, 70), 0.58, colors["text"])
    summary = (
        f"visited={metadata.get('visited_count', len(trace))}  "
        f"selected={len(selected_indices)}  "
        f"threshold={len(threshold_selected)}  "
        f"budget_fill={len(budget_filled)}  "
        f"s_min={params.get('s_min')}  s_max={effective_stride}"
    )
    put_text(summary, (plot_left, 105), 0.58, colors["axis"])

    score_top = top_margin
    draw_axes(score_top, score_height, y_min=0.0, y_max=1.0, y_label="Score")
    score_points = [
        (map_x(int(record["frame_index"])), map_y(float(record["score"]), score_top, score_height, 0.0, 1.0))
        for record in trace
    ]
    smoothed_points = [
        (map_x(int(record["frame_index"])), map_y(float(record["smoothed_score"]), score_top, score_height, 0.0, 1.0))
        for record in trace
    ]
    threshold_points = [
        (map_x(int(record["frame_index"])), map_y(float(record["threshold"]), score_top, score_height, 0.0, 1.0))
        for record in trace
    ]
    _draw_polyline(canvas, score_points, colors["score"], thickness=3)
    _draw_polyline(canvas, smoothed_points, colors["smoothed"], thickness=2, dashed=True)
    _draw_polyline(canvas, threshold_points, colors["threshold"], thickness=2, dashed=True)

    for record in trace:
        x = map_x(int(record["frame_index"]))
        y = map_y(float(record["score"]), score_top, score_height, 0.0, 1.0)
        cv2.circle(canvas, (x, y), 5, colors["visited"], -1, lineType=cv2.LINE_AA)

    for index in selected_indices:
        matching = next((record for record in trace if int(record["frame_index"]) == index), None)
        if matching is None:
            continue
        x = map_x(index)
        y = map_y(float(matching["score"]), score_top, score_height, 0.0, 1.0)
        star_color = colors["force"] if index in force_selected else colors["selected"]
        _draw_star(canvas, (x, y), 17, star_color, colors["text"])

    legend_x = plot_left
    legend_y = score_top - 12
    legend_items = [
        ("score", colors["score"], "line"),
        ("smoothed", colors["smoothed"], "dash"),
        ("threshold", colors["threshold"], "dash"),
        ("visited", colors["visited"], "dot"),
        ("selected", colors["selected"], "star"),
    ]
    for label, color, kind in legend_items:
        if kind == "line":
            cv2.line(canvas, (legend_x, legend_y - 6), (legend_x + 32, legend_y - 6), color, 3, lineType=cv2.LINE_AA)
        elif kind == "dash":
            _draw_polyline(canvas, [(legend_x, legend_y - 6), (legend_x + 32, legend_y - 6)], color, thickness=2, dashed=True)
        elif kind == "dot":
            cv2.circle(canvas, (legend_x + 14, legend_y - 6), 6, color, -1, lineType=cv2.LINE_AA)
        else:
            _draw_star(canvas, (legend_x + 15, legend_y - 6), 12, color, colors["text"])
        put_text(label, (legend_x + 42, legend_y), 0.54, colors["axis"])
        legend_x += 165

    stride_top = score_top + score_height + panel_gap
    s_max_param = params.get("s_max")
    if stride_axis_max is not None:
        stride_y_max = float(stride_axis_max)
    elif s_max_param is not None:
        stride_y_max = float(s_max_param)
    else:
        stride_y_max = max(float(record.get("stride", 1)) for record in trace)
    stride_y_max = max(2.0, math.ceil(stride_y_max))
    draw_axes(stride_top, stride_height, y_min=0.0, y_max=stride_y_max, y_label="Stride")
    stride_points = [
        (
            map_x(int(record["frame_index"])),
            map_y(float(record["stride"]), stride_top, stride_height, 0.0, stride_y_max),
        )
        for record in trace
    ]
    budget_points = [
        (
            map_x(int(record["frame_index"])),
            map_y(
                min(float(record["budget_stride"]), stride_y_max),
                stride_top,
                stride_height,
                0.0,
                stride_y_max,
            ),
        )
        for record in trace
    ]
    _draw_polyline(canvas, stride_points, colors["stride"], thickness=3)
    _draw_polyline(canvas, budget_points, colors["budget"], thickness=2, dashed=True)
    put_text("stride", (plot_left, stride_top - 20), 0.54, colors["stride"])
    put_text("budget stride clipped", (plot_left + 110, stride_top - 20), 0.54, colors["budget"])

    if use_frames:
        frames = _as_rgb_uint8_frames(frame_selection.frames)
        thumb_top = stride_top + stride_height + frame_gap + 42
        count = min(len(frames), len(selected_indices))
        if count > 0:
            available_width = plot_right - plot_left
            gap = 10
            thumb_width = max(70, (available_width - gap * (count - 1)) // count)
            for idx in range(count):
                frame = frames[idx]
                h, w = frame.shape[:2]
                scale = min(thumb_width / max(w, 1), thumb_height / max(h, 1))
                resized_w = max(1, int(round(w * scale)))
                resized_h = max(1, int(round(h * scale)))
                resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
                x = plot_left + idx * (thumb_width + gap) + (thumb_width - resized_w) // 2
                y = thumb_top + (thumb_height - resized_h) // 2
                canvas[y : y + resized_h, x : x + resized_w] = resized
                border_color = colors["budget"] if selected_indices[idx] in budget_filled else colors["selected"]
                cv2.rectangle(
                    canvas,
                    (plot_left + idx * (thumb_width + gap), thumb_top),
                    (plot_left + idx * (thumb_width + gap) + thumb_width, thumb_top + thumb_height),
                    border_color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                label = f"#{idx + 1}  frame {selected_indices[idx]}"
                put_text(label, (plot_left + idx * (thumb_width + gap) + 6, thumb_top + thumb_height + 26), 0.5, colors["axis"])
            put_text("Selected frames", (plot_left, thumb_top - 16), 0.62, colors["axis"], 2)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return canvas


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
