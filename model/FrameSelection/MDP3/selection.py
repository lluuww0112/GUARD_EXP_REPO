"""
MDP3 Frame Selection for Video-LLMs
====================================
Paper: "MDP^3: A Training-free Approach for List-wise Frame Selection in Video-LLMs"
       (Sun et al., 2025)

Implements:
  - Sec 3.1: Conditional Multiple Gaussian Kernel (CMGK) in RKHS
  - Sec 3.2: Greedy DPP MAP inference with Cholesky updates
  - Sec 3.3: Markov Decision DPP with Dynamic Programming (Algorithm 1)

This module keeps the current project integration intact while exposing the
paper-style MDP3 selector as a standard frame-selection function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from model.base.selection import (
    FrameSelectionResult,
    PatchSelectionResult,
    QWEN_VISION_FACTOR,
    uniform_sampling,
)


# ============================================================
# Constants
# ============================================================
# Eq. 3 – Conditional Multiple Gaussian kernel의 alpha_u = 2^i, i in {-3,-2,0,1,2}
DEFAULT_ALPHAS = [2**i for i in (-3, -2, 0, 1, 2)]

# The embedding hook is supplied by the VLM wrapper so the selector itself
# stays framework-agnostic and only depends on frame/query embeddings.
EmbeddingFunction = Callable[[torch.Tensor, str], tuple[torch.Tensor, torch.Tensor]]


# ============================================================
# MDP3 – Sec 3.1: Conditional Multiple Gaussian Kernel (Eq. 3)
#   K = { k = sum_u beta_u k_u }, beta_u = 1/U (uniform)
#   k_u(x,y) = exp( -||x-y||^2 / (2 * alpha_u) )
# ============================================================
def _multi_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    alphas: list[float] | None = None,
) -> torch.Tensor:
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    if y is None:
        y = x

    dist_sq = (
        (x * x).sum(dim=1, keepdim=True)
        + (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
        - 2.0 * x @ y.transpose(0, 1)
    ).clamp(min=0.0)

    kernel = torch.zeros_like(dist_sq)
    for alpha in alphas:
        kernel = kernel + torch.exp(-dist_sq / (2.0 * alpha))
    return kernel / len(alphas)


# ============================================================
# MDP3 – Sec 3.1: Conditional Multiple Gaussian Kernel (Eq. 4-6)
#   k_tilde(f_i, f_j | q) = g(f_i,q) * k(f_i,f_j) * g(f_j,q)
#   L_tilde = diag(r^(1/lambda)) * L * diag(r^(1/lambda))
# ============================================================
def _build_conditional_similarity(
    frame_embeds: torch.Tensor,   # (n, d) – expected to be L2-normalized
    query_embed: torch.Tensor,    # (d,) or (1, d) – expected to be L2-normalized
    *,
    alphas: list[float] | None = None,
    lam: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        L_tilde: (n, n) query-conditioned similarity used by MDP3
        L:       (n, n) frame-frame similarity before conditioning
        r:       (n,)   frame-query relevance
    """
    if lam <= 0:
        raise ValueError(f"`lam` must be positive, got {lam}.")
    if query_embed.ndim == 1:
        query_embed = query_embed.unsqueeze(0)

    similarity = _multi_gaussian_kernel(frame_embeds, None, alphas)
    relevance = _multi_gaussian_kernel(
        frame_embeds,
        query_embed,
        alphas,
    ).squeeze(-1)

    # Eq. 9: trade-off via r^(1/lambda)
    relevance_scaled = relevance.clamp(min=1e-10).pow(1.0 / lam)

    # L_tilde = diag(relevance_scaled) * L * diag(relevance_scaled)
    conditioned = (
        torch.diag(relevance_scaled)
        @ similarity
        @ torch.diag(relevance_scaled)
    )
    return conditioned, similarity, relevance 
    # the similarities among frames while considering the query relevance,
    # framesimilarities,
    # frame-query relevances


# ============================================================
# MDP3 – Sec 3.2: Greedy DPP MAP Inference (Eq. 10)
#   With Cholesky-style incremental updates -> O(nk^2)
# ============================================================
def _greedy_dpp_map(
    l_tilde: torch.Tensor,
    k: int,
    *,
    candidate_mask: torch.Tensor | None = None,
    condition_indices: list[int] | None = None,
) -> list[int]:
    """
    Greedy MAP inference for a k-DPP.

    Optionally conditions on previously selected indices from the prior segment,
    matching the sequential formulation used in the paper.
    """
    n = l_tilde.shape[0]
    device = l_tilde.device
    dtype = l_tilde.dtype

    if candidate_mask is None:
        candidate_mask = torch.ones(n, dtype=torch.bool, device=device)

    conditioned = list(condition_indices or [])
    max_selected = k + len(conditioned)
    chol_vectors = torch.zeros(max_selected, n, dtype=dtype, device=device)
    diagonal = l_tilde.diag().clone()

    # Initialize the Cholesky-style cache for conditioned items first.
    for i, index in enumerate(conditioned):
        vector = l_tilde[index].clone()
        for prev in range(i):
            vector -= chol_vectors[prev, index] * chol_vectors[prev]
        vector /= vector[index].clamp(min=1e-10).sqrt()
        chol_vectors[i] = vector
        diagonal = (diagonal - vector.square()).clamp(min=0.0)

    result: list[int] = []
    used = set(conditioned)
    offset = len(conditioned)

    for step in range(k):
        gains = diagonal.clone()
        gains[~candidate_mask] = -float("inf")
        for used_index in used:
            gains[used_index] = -float("inf")

        best = int(torch.argmax(gains).item())
        if float(gains[best]) <= 1e-12:
            break

        result.append(best)
        used.add(best)

        position = offset + step
        vector = l_tilde[best].clone()
        for prev in range(position):
            vector -= chol_vectors[prev, best] * chol_vectors[prev]
        vector /= vector[best].clamp(min=1e-10).sqrt()
        chol_vectors[position] = vector
        diagonal = (diagonal - vector.square()).clamp(min=0.0)

    return result


def _log_det_score(l_tilde: torch.Tensor, indices: list[int]) -> float:
    """Compute log det(L_tilde[S]) as the segment reward used in DP."""
    if not indices:
        return 0.0

    submatrix = l_tilde[indices][:, indices]
    sign, logabsdet = torch.linalg.slogdet(submatrix)
    return float(logabsdet.item()) if float(sign.item()) > 0 else -1.0e30


# ============================================================
# MDP3 – Sec 3.3: Markov Decision DPP with Dynamic Programming 
#   Algorithm 1 in the paper 
# ============================================================
def mdp3_frame_selection(
    frame_embeds: torch.Tensor,   # (n, d) – L2-normalized frame embeddings
    query_embed: torch.Tensor,    # (d,)   – L2-normalized query embedding
    *,
    k: int = 8,
    segment_size: int = 32,
    lam: float = 0.2,
    alphas: list[float] | None = None,
) -> list[int]:
    """
    Run the segment-wise DP used by MDP3.

    State:  (t, c_t) where t is the segment index and c_t is the cumulative
            number of selected frames up to segment t.
    Action: choose k_t frames from the current segment.
    Reward: log P(S_t | S_{t-1}) approximated via log det differences.
    """
    if k <= 0:
        raise ValueError(f"`k` must be positive, got {k}.")
    if segment_size <= 0:
        raise ValueError(f"`segment_size` must be positive, got {segment_size}.")

    n = int(frame_embeds.shape[0])
    if k >= n:
        return list(range(n))

    if alphas is None:
        alphas = DEFAULT_ALPHAS

    l_tilde, _, _ = _build_conditional_similarity(
        frame_embeds,
        query_embed,
        alphas=alphas,
        lam=lam,
    )

    num_segments = (n + segment_size - 1) // segment_size
    neg_inf = -float("inf")

    # Q_star[t][c] stores the best score after processing t segments and
    # selecting c frames in total. trace[t][c] stores the corresponding path.
    q_star = [[neg_inf] * (k + 1) for _ in range(num_segments + 1)]
    trace: list[list[list[int]]] = [
        [[] for _ in range(k + 1)]
        for _ in range(num_segments + 1)
    ]
    q_star[0][0] = 0.0

    for segment_idx in range(1, num_segments + 1):
        segment_start = (segment_idx - 1) * segment_size
        segment_end = min(segment_idx * segment_size, n)
        segment_indices = list(range(segment_start, segment_end))
        segment_length = len(segment_indices)

        for prev_count in range(k + 1):
            if q_star[segment_idx - 1][prev_count] == neg_inf:
                continue

            previous_selection = trace[segment_idx - 1][prev_count]

            # The current implementation conditions on the last chosen frame of
            # the previous segment, which mirrors the sequential dependency used
            # in the paper while keeping the state compact.
            conditioned_global = [previous_selection[-1]] if previous_selection else []

            local_global = conditioned_global + segment_indices
            l_local = l_tilde[local_global][:, local_global]
            conditioned_count = len(conditioned_global)

            candidate_mask = torch.zeros(
                len(local_global),
                dtype=torch.bool,
                device=l_tilde.device,
            )
            candidate_mask[conditioned_count:] = True
            conditioned_local = (
                list(range(conditioned_count))
                if conditioned_count
                else None
            )

            max_pick = min(segment_length, k - prev_count)
            for pick_count in range(max_pick + 1):
                new_count = prev_count + pick_count
                if pick_count == 0:
                    reward = 0.0
                    selected_global: list[int] = []
                else:
                    selected_local = _greedy_dpp_map(
                        l_local,
                        pick_count,
                        candidate_mask=candidate_mask,
                        condition_indices=conditioned_local,
                    )
                    selected_global = [local_global[index] for index in selected_local]
                    all_selected = conditioned_global + selected_global

                    reward = _log_det_score(l_tilde, all_selected)
                    if conditioned_global:
                        reward -= _log_det_score(l_tilde, conditioned_global)

                candidate_score = q_star[segment_idx - 1][prev_count] + reward
                if candidate_score > q_star[segment_idx][new_count]:
                    q_star[segment_idx][new_count] = candidate_score
                    trace[segment_idx][new_count] = previous_selection + selected_global

    return trace[num_segments][k]


# ============================================================
# End-to-end MDP3 pipeline (video -> candidate frames -> embeddings -> subset)
# ============================================================
def mdp3_sampling(
    video_path: str,
    *,
    num_frames: int = 8,
    num_candidates: int = 128,
    embed_fn: EmbeddingFunction | None = None,
    query: str = "",
    query_file: str | None = None,
    segment_size: int = 32,
    lam: float = 0.2,
    alphas: list[float] | None = None,
    max_side: int | None = 720,
    ensure_qwen_compatibility: bool = True,
    qwen_factor: int = QWEN_VISION_FACTOR,
) -> FrameSelectionResult:
    """
    Full project-facing MDP3 selector.

    1. Uniformly sample candidate frames from the video.
    2. Use the injected embedding function to get frame/query embeddings.
    3. Run the query-conditioned MDP3 DP.
    4. Return the selected frame tensor in the project's standard format.
    """
    if embed_fn is None:
        raise ValueError(
            "`embed_fn` is required for query-conditioned MDP3 frame selection."
        )
    if num_candidates <= 0:
        raise ValueError(f"`num_candidates` must be positive, got {num_candidates}.")

    query_text = query.strip()
    if not query_text and query_file:
        query_path = Path(query_file).expanduser()
        if not query_path.exists():
            raise FileNotFoundError(f"Query file not found: {query_path}")
        query_text = query_path.read_text(encoding="utf-8").strip()
    if not query_text:
        raise ValueError("`query` or `query_file` must provide a non-empty query.")

    candidates = uniform_sampling(
        video_path=video_path,
        num_frames=num_candidates,
        max_side=max_side,
        ensure_qwen_compatibility=ensure_qwen_compatibility,
        qwen_factor=qwen_factor,
    )

    frame_embeds, query_embed = embed_fn(candidates.frames, query_text)

    # CLIP/SigLIP-style embeddings are compared in normalized space.
    frame_embeds = torch.nn.functional.normalize(frame_embeds, dim=-1)
    query_embed = torch.nn.functional.normalize(query_embed, dim=-1)

    selected = mdp3_frame_selection(
        frame_embeds,
        query_embed,
        k=num_frames,
        segment_size=segment_size,
        lam=lam,
        alphas=alphas,
    )
    selected_sorted = sorted(selected)
    original_indices = candidates.metadata.get("sampled_indices", [])

    return FrameSelectionResult(
        frames=candidates.frames[selected_sorted],
        metadata={
            **candidates.metadata,
            "method": "mdp3",
            "num_candidates": num_candidates,
            "selected_from_candidates": selected_sorted,
            "selected_original_indices": [
                original_indices[i] for i in selected_sorted
            ],
            "num_frames": len(selected_sorted),
            "mdp3_params": {
                "segment_size": segment_size,
                "lambda": lam,
                "alphas": alphas or DEFAULT_ALPHAS,
            },
            "query_text": query_text,
        },
    )


# ============================================================
# Identity patch selection (pass-through baseline)
# ============================================================
def identity_patch_selection(
    video_features: torch.Tensor,
    **_: Any,
) -> PatchSelectionResult:
    return PatchSelectionResult(
        selected_indices=torch.arange(
            video_features.shape[0],
            device=video_features.device,
            dtype=torch.long,
        )
    )
