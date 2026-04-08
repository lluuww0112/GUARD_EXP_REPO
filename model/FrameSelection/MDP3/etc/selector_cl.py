"""
MDP3 Frame Selection for Video-LLMs
====================================
Paper: "MDP^3: A Training-free Approach for List-wise Frame Selection in Video-LLMs"
       (Sun et al., 2025)

Implements:
  - Sec 3.1: Conditional Multiple Gaussian Kernel (CMGK) in RKHS
  - Sec 3.2: Greedy DPP MAP inference with Cholesky updates
  - Sec 3.3: Markov Decision DPP with Dynamic Programming (Algorithm 1)

Target model: Qwen3-VL (vision factor=28, SigLIP-compatible embeddings)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable

import cv2
import numpy as np
import torch


# ============================================================
# Constants
# ============================================================
QWEN_VISION_FACTOR = 28
QWEN_MIN_PIXELS = 56 * 56
QWEN_MAX_PIXELS = 14 * 14 * 4 * 1280

# Eq. 3 – Gaussian kernel bandwidths: α_u = 2^i, i ∈ {-3,-2,0,1,2}
# Following Long et al. (MK-MMD) and paper Sec B.3
DEFAULT_ALPHAS = [2 ** i for i in (-3, -2, 0, 1, 2)]


# ============================================================
# Data classes
# ============================================================
@dataclass(slots=True)
class FrameSelectionResult:
    frames: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatchSelectionResult:
    selected_indices: torch.Tensor | None = None
    selected_features: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Video I/O helpers
# ============================================================
def _inspect_video_capture(video_path: str) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    return cap, total_frames, fps


def _transcode_video_for_opencv(video_path: str) -> Path | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return None
    temp_dir = Path(tempfile.mkdtemp(prefix="guard_opencv_decode_"))
    output_path = temp_dir / f"{Path(video_path).stem}_opencv_h264.mp4"
    command = [
        ffmpeg_path, "-y", "-loglevel", "error", "-i", video_path,
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    return output_path if output_path.exists() else None


def _open_video_for_sampling(
    video_path: str,
) -> tuple[cv2.VideoCapture, int, float, Path | None]:
    cap, total_frames, fps = _inspect_video_capture(video_path)
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, None
    cap.release()
    transcoded_path = _transcode_video_for_opencv(video_path)
    if transcoded_path is None:
        raise RuntimeError(
            "Failed to decode video with OpenCV. "
            "Install ffmpeg or transcode to H.264 before running."
        )
    cap, total_frames, fps = _inspect_video_capture(str(transcoded_path))
    if cap.isOpened() and total_frames > 0:
        return cap, total_frames, fps, transcoded_path
    cap.release()
    shutil.rmtree(transcoded_path.parent, ignore_errors=True)
    raise RuntimeError(f"Failed to decode video: {video_path}")


# ============================================================
# Qwen-compatible resize
# ============================================================
def _qwen_smart_resize(
    height: int, width: int, *,
    factor: int = QWEN_VISION_FACTOR,
    min_pixels: int = QWEN_MIN_PIXELS,
    max_pixels: int = QWEN_MAX_PIXELS,
) -> tuple[int, int]:
    if factor <= 0:
        return height, width
    if min(height, width) <= 0:
        raise ValueError(f"Invalid frame size: {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Aspect ratio > 200")
    rh = max(factor, round(height / factor) * factor)
    rw = max(factor, round(width / factor) * factor)
    if rh * rw > max_pixels:
        beta = float(np.sqrt((height * width) / max_pixels))
        rh = max(factor, int(np.floor(height / beta / factor)) * factor)
        rw = max(factor, int(np.floor(width / beta / factor)) * factor)
    elif rh * rw < min_pixels:
        beta = float(np.sqrt(min_pixels / (height * width)))
        rh = max(factor, int(np.ceil(height * beta / factor)) * factor)
        rw = max(factor, int(np.ceil(width * beta / factor)) * factor)
    return rh, rw


def _resize_frame(
    frame_rgb: np.ndarray, *,
    max_side: int | None,
    ensure_qwen_compatibility: bool,
    qwen_factor: int,
) -> np.ndarray:
    out = frame_rgb
    if max_side is not None:
        h, w = out.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            out = cv2.resize(out, (max(1, round(w*scale)), max(1, round(h*scale))),
                             interpolation=cv2.INTER_AREA)
    if ensure_qwen_compatibility:
        h, w = out.shape[:2]
        rh, rw = _qwen_smart_resize(h, w, factor=qwen_factor)
        if (rh, rw) != (h, w):
            out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_AREA)
    return out


# ============================================================
# Baseline: Uniform frame sampling
# ============================================================
def uniform_sampling(
    video_path: str,
    num_frames: int = 8,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    cap, total_frames, fps, transcoded_path = _open_video_for_sampling(video_path)
    indices = ([total_frames // 2] if num_frames == 1
               else np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist())
    target_set = set(indices)
    frames, sampled_indices, idx = [], [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in target_set:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _resize_frame(rgb, max_side=max_side,
                                    ensure_qwen_compatibility=ensure_qwen_compatibility,
                                    qwen_factor=qwen_factor)
                frames.append(rgb)
                sampled_indices.append(idx)
            idx += 1
    finally:
        cap.release()
        if transcoded_path is not None:
            shutil.rmtree(transcoded_path.parent, ignore_errors=True)
    if len(frames) != len(indices):
        raise RuntimeError(f"Expected {len(indices)} frames, got {len(frames)}.")
    bh, bw = frames[0].shape[:2]
    normed = [cv2.resize(f, (bw, bh), interpolation=cv2.INTER_AREA)
              if f.shape[:2] != (bh, bw) else f for f in frames]
    video_np = np.stack(normed, axis=0)
    return FrameSelectionResult(
        frames=torch.from_numpy(video_np),
        metadata={"video_path": video_path, "sampled_indices": sampled_indices,
                  "num_frames": len(normed), "total_frames": total_frames,
                  "fps": fps if fps > 0 else None, "frame_shape": list(video_np.shape[1:])},
    )


# ============================================================
# MDP3 – Sec 3.1: Multiple Gaussian Kernel (Eq. 3)
#   K = { k = Σ_u β_u k_u }, β_u = 1/U (uniform)
#   k_u(x,y) = exp( -||x-y||^2 / (2·α_u) )
# ============================================================
def _multi_gaussian_kernel(
    X: torch.Tensor,
    Y: torch.Tensor | None = None,
    alphas: list[float] | None = None,
) -> torch.Tensor:
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if Y is None:
        Y = X
    dist_sq = (
        (X * X).sum(1, keepdim=True)
        + (Y * Y).sum(1, keepdim=True).T
        - 2.0 * X @ Y.T
    ).clamp(min=0.0)
    K = torch.zeros_like(dist_sq)
    for a in alphas:
        K += torch.exp(-dist_sq / (2.0 * a))
    return K / len(alphas)


# ============================================================
# MDP3 – Sec 3.1: Conditional Multiple Gaussian Kernel (Eq. 4-6)
#   k̃(f_i, f_j | q) = g(f_i,q) · k(f_i,f_j) · g(f_j,q)
#   L̃ = diag(r^(1/λ)) · L · diag(r^(1/λ))
# ============================================================
def _build_conditional_similarity(
    frame_embeds: torch.Tensor,   # (n, d) – must be L2-normalized
    query_embed: torch.Tensor,    # (d,) or (1, d) – must be L2-normalized
    alphas: list[float] | None = None,
    lam: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        L_tilde: (n, n) conditional similarity (Eq. 6 with λ scaling from Eq. 9)
        L:       (n, n) frame-frame similarity
        r:       (n,)   frame-query relevance
    """
    if query_embed.dim() == 1:
        query_embed = query_embed.unsqueeze(0)

    L = _multi_gaussian_kernel(frame_embeds, None, alphas)         # Eq. 5 inner part
    r = _multi_gaussian_kernel(frame_embeds, query_embed, alphas)  # Eq. 6 r_i
    r = r.squeeze(-1)                                              # (n,)

    # Eq. 9: trade-off via r^(1/λ) — equivalent to scaling bandwidth by √λ
    r_scaled = r.clamp(min=1e-10).pow(1.0 / lam)

    # L̃ = diag(r_scaled) · L · diag(r_scaled)  (Eq. 6)
    L_tilde = torch.diag(r_scaled) @ L @ torch.diag(r_scaled)

    return L_tilde, L, r


# ============================================================
# MDP3 – Sec 3.2: Greedy DPP MAP Inference (Eq. 10)
#   With Cholesky incremental updates → O(nk^2)
# ============================================================
def _greedy_dpp_map(
    L_tilde: torch.Tensor,
    k: int,
    candidate_mask: torch.Tensor | None = None,
    condition_indices: list[int] | None = None,
) -> list[int]:
    """
    Greedy MAP for k-DPP.
    Selects k items maximizing marginal log-det gain (Eq. 10).
    Optionally conditioned on previous segment (Eq. 11).
    """
    n = L_tilde.shape[0]
    device, dtype = L_tilde.device, L_tilde.dtype
    if candidate_mask is None:
        candidate_mask = torch.ones(n, dtype=torch.bool, device=device)

    pre = list(condition_indices) if condition_indices else []
    max_sel = k + len(pre)
    V = torch.zeros(max_sel, n, dtype=dtype, device=device)
    diag = L_tilde.diag().clone()

    # Initialize Cholesky for condition indices
    for i, j in enumerate(pre):
        v = L_tilde[j].clone()
        for p in range(i):
            v -= V[p, j] * V[p]
        v /= v[j].clamp(min=1e-10).sqrt()
        V[i] = v
        diag -= v ** 2

    diag = diag.clamp(min=0.0)
    nc = len(pre)
    used = set(pre)
    result: list[int] = []

    for step in range(k):
        gains = diag.clone()
        gains[~candidate_mask] = -float("inf")
        for j in used:
            gains[j] = -float("inf")
        best = gains.argmax().item()
        if gains[best] <= 1e-12:
            break
        result.append(best)
        used.add(best)

        pos = nc + step
        v = L_tilde[best].clone()
        for p in range(pos):
            v -= V[p, best] * V[p]
        v /= v[best].clamp(min=1e-10).sqrt()
        V[pos] = v
        diag = (diag - v ** 2).clamp(min=0.0)

    return result


def _log_det_score(L_tilde: torch.Tensor, indices: list[int]) -> float:
    """log det(L̃_S) — reward computation for DP."""
    if not indices:
        return 0.0
    sub = L_tilde[indices][:, indices]
    sign, logabsdet = torch.linalg.slogdet(sub)
    return logabsdet.item() if sign > 0 else -1e30


# ============================================================
# MDP3 – Sec 3.3: MDP + Dynamic Programming (Algorithm 1)
# ============================================================
def mdp3_frame_selection(
    frame_embeds: torch.Tensor,   # (n, d) L2-normalized VLM embeddings
    query_embed: torch.Tensor,    # (d,) L2-normalized VLM text embedding
    k: int = 8,
    segment_size: int = 32,       # m (paper default, Sec B.3)
    lam: float = 0.2,            # λ trade-off (paper default, Sec B.3)
    alphas: list[float] | None = None,
) -> list[int]:
    """
    Algorithm 1: MDP^3

    State:  (t, C_t) where t = segment index, C_t = cumulative selections
    Action: select k_t frames from segment t
    Reward: log P(S_t | S_{t-1})  (Eq. 15, 18)
    DP:     Q*[t, C_t] = max over k_t { Q*[t-1, C_{t-1}] + R }  (Eq. 19)

    Complexity: O(n·k^3) worst case, O(n·k^2) with parallel updates (Sec A.3)
    """
    n = frame_embeds.shape[0]
    if k >= n:
        return list(range(n))
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # Build global L̃ (Eq. 4-6)
    L_tilde, L, r = _build_conditional_similarity(frame_embeds, query_embed, alphas, lam)

    T = (n + segment_size - 1) // segment_size

    # DP tables: Q*[t][c] and trace[t][c]
    NEG_INF = -float("inf")
    Q_star = [[NEG_INF] * (k + 1) for _ in range(T + 1)]
    trace  = [[[] for _ in range(k + 1)] for _ in range(T + 1)]
    Q_star[0][0] = 0.0

    for t in range(1, T + 1):                        # Line 3
        seg_start = (t - 1) * segment_size
        seg_end = min(t * segment_size, n)
        seg_indices = list(range(seg_start, seg_end))
        m_t = len(seg_indices)

        for c_prev in range(k + 1):                  # Line 6
            if Q_star[t - 1][c_prev] == NEG_INF:
                continue

            prev_trace = trace[t - 1][c_prev]
            # Condition on last selected frame (lazy strategy, Eq. 18)
            cond_global = [prev_trace[-1]] if prev_trace else []

            # Local sub-matrix for [cond ∪ segment]
            local_global = cond_global + seg_indices
            L_local = L_tilde[local_global][:, local_global]
            n_cond = len(cond_global)
            cand_mask = torch.zeros(len(local_global), dtype=torch.bool, device=L_tilde.device)
            cand_mask[n_cond:] = True
            cond_local = list(range(n_cond)) if n_cond else None

            max_kt = min(m_t, k - c_prev)
            for k_t in range(max_kt + 1):             # Line 7
                c_new = c_prev + k_t
                if k_t == 0:
                    rwd, sel_global = 0.0, []
                else:
                    # Greedy DPP on local matrix (Line 9)
                    sel_local = _greedy_dpp_map(L_local, k_t, cand_mask, cond_local)
                    sel_global = [local_global[i] for i in sel_local]

                    # Reward = log P(S_t | S_{t-1}) (Eq. 15)
                    all_sel = cond_global + sel_global
                    rwd = _log_det_score(L_tilde, all_sel)
                    if cond_global:
                        rwd -= _log_det_score(L_tilde, cond_global)

                cur_q = Q_star[t - 1][c_prev] + rwd   # Line 11
                if cur_q > Q_star[t][c_new]:           # Line 12
                    Q_star[t][c_new] = cur_q            # Line 13
                    trace[t][c_new] = prev_trace + sel_global  # Line 14

    # Final: T_{T,k}
    return trace[T][k]


# ============================================================
# End-to-end MDP3 pipeline (video → selected frames)
# ============================================================
def mdp3_sampling(
    video_path: str,
    num_frames: int = 8,
    num_candidates: int = 128,
    embed_fn: Callable | None = None,
    query: str = "",
    segment_size: int = 32,
    lam: float = 0.2,
    alphas: list[float] | None = None,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    """
    Full pipeline:
      1) Uniform sample num_candidates frames
      2) embed_fn(frames, query) → (frame_embeds, query_embed)
         Both MUST be L2-normalized (standard for CLIP/SigLIP/Qwen-VL)
      3) MDP3 selects num_frames
      4) Return selected frames with metadata

    embed_fn signature:
        (frames: Tensor[N,H,W,3], query: str) -> (Tensor[N,d], Tensor[d])
    """
    candidates = uniform_sampling(
        video_path, num_frames=num_candidates,
        max_side=max_side, ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )
    if embed_fn is None:
        raise ValueError(
            "embed_fn required: (frames: Tensor[N,H,W,3], query: str) "
            "-> (frame_embeds: Tensor[N,d], query_embed: Tensor[d])"
        )

    frame_embeds, query_embed = embed_fn(candidates.frames, query)

    # Safety: ensure L2-normalized
    frame_embeds = torch.nn.functional.normalize(frame_embeds, dim=-1)
    query_embed = torch.nn.functional.normalize(query_embed, dim=-1)

    selected = mdp3_frame_selection(
        frame_embeds, query_embed,
        k=num_frames, segment_size=segment_size, lam=lam, alphas=alphas,
    )
    selected_sorted = sorted(selected)
    orig_sampled = candidates.metadata.get("sampled_indices", [])

    return FrameSelectionResult(
        frames=candidates.frames[selected_sorted],
        metadata={
            **candidates.metadata,
            "method": "mdp3",
            "num_candidates": num_candidates,
            "selected_from_candidates": selected_sorted,
            "selected_original_indices": [orig_sampled[i] for i in selected_sorted],
            "num_frames": len(selected_sorted),
            "mdp3_params": {
                "segment_size": segment_size, "lambda": lam,
                "alphas": alphas or DEFAULT_ALPHAS,
            },
        },
    )


# ============================================================
# Identity patch selection (pass-through baseline)
# ============================================================
def identity_patch_selection(
    video_features: torch.Tensor, **_: Any,
) -> PatchSelectionResult:
    return PatchSelectionResult(
        selected_indices=torch.arange(
            video_features.shape[0], device=video_features.device, dtype=torch.long
        ),
    )
