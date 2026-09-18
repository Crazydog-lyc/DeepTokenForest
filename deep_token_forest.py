"""
deep_token_forest.py

Backpropagation-free hierarchical token forest.

Main ideas
----------
1. Replace Top-K-only aggregation with projection-distribution splits.

   For a token representation h_it and a direction w,

       q_it = h_it^T w
       c_i(w, b) = mean_t 1[q_it >= b]

   Each tree node searches (w, b, rho) and routes by

       c_i(w, b) >= rho.

   `c_i` is one point of the empirical survival function of the projected token
   distribution. Multiple token thresholds b allow the tree to inspect the
   whole projected distribution rather than only the largest K token responses.

2. Keep ordinary Newton gradient boosting for supervised learning.  No teacher
   logits and no backpropagation are used.

3. Learn token-dependent concept features between stages.

       g_j(h) = tanh((w_j^T h - b_j) / tau)

   High-value internal nodes from stage s are reused as token channels:

       H^(s+1) = concat(H^(s), standardized g_1, ..., g_M).

   Because H^(s) is retained, feature growth cannot reduce representational
   capacity: a later direction can always put zero weight on all new channels.

4. Optional honest split selection. Candidate directions/thresholds are searched
   on one subset of the node samples and the best few candidates are checked on
   a disjoint in-training holdout. This is intended to reduce winner's-curse
   overfitting when many weak candidate splits are compared.

The implementation is NumPy-first and intentionally independent of autograd.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import heapq
import json
import math
import os

import numpy as np
import torch

Array = np.ndarray


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def _softmax(logits: Array) -> Array:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= np.maximum(np.sum(x, axis=1, keepdims=True), 1e-300)
    return x.astype(np.float32)


def _logloss(y: Array, logits: Array) -> float:
    p = _softmax(logits)
    y = np.asarray(y, dtype=np.int64)
    return float(
        -np.mean(
            np.log(
                np.clip(
                    p[np.arange(len(y)), y],
                    1e-12,
                    1.0,
                )
            )
        )
    )


def _normalize_rows(W: Array, eps: float = 1e-12) -> Array:
    W = np.asarray(W, dtype=np.float32)
    if W.ndim == 1:
        W = W[None, :]
    if W.size == 0:
        d = 0 if W.ndim < 2 else W.shape[1]
        return np.empty((0, d), dtype=np.float32)
    norms = np.linalg.norm(W.astype(np.float64), axis=1)
    keep = np.isfinite(norms) & (norms > eps)
    W = W[keep]
    norms = norms[keep]
    if len(W) == 0:
        return np.empty((0, W.shape[1]), dtype=np.float32)
    return (W / norms[:, None]).astype(np.float32, copy=False)


def _deduplicate_directions(W: Array, cosine_tol: float = 0.99995) -> Array:
    """
    Remove almost identical directions while keeping opposite signs distinct.
    Opposite directions are NOT redundant for distribution thresholds.
    """
    W = _normalize_rows(W)
    if len(W) <= 1:
        return W
    keep: List[int] = []
    for i in range(len(W)):
        if not keep:
            keep.append(i)
            continue
        sims = W[keep].astype(np.float64) @ W[i].astype(np.float64)
        if np.max(sims) < cosine_tol:
            keep.append(i)
    return W[np.asarray(keep, dtype=np.int64)]


def masked_mean_tokens(
    X_tokens: Array,
    attention_mask: Array,
    rows: Optional[Array] = None,
    block_rows: int = 2048,
) -> Array:
    X = X_tokens
    mask = np.asarray(attention_mask)
    if rows is None:
        rows = np.arange(len(X), dtype=np.int64)
    else:
        rows = np.asarray(rows, dtype=np.int64)
    d = int(X.shape[2])
    out = np.empty((len(rows), d), dtype=np.float32)
    for start in range(0, len(rows), block_rows):
        end = min(len(rows), start + block_rows)
        rr = rows[start:end]
        xb = np.asarray(X[rr], dtype=np.float32)
        mb = np.asarray(mask[rr], dtype=np.float32)
        denom = np.maximum(mb.sum(axis=1, keepdims=True), 1.0)
        out[start:end] = (xb * mb[:, :, None]).sum(axis=1) / denom
    return out


def _jsonify(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


# ---------------------------------------------------------------------------
# Learned token concepts
# ---------------------------------------------------------------------------

@dataclass
class TokenConcept:
    w: Array
    token_threshold: float
    temperature: float
    feature_mean: float = 0.0
    feature_scale: float = 1.0
    stage: int = -1
    tree: int = -1
    node_id: int = -1
    depth: int = -1
    n_samples: int = 0
    search_gain: float = 0.0
    honest_gain: float = float("nan")
    effective_gain: float = 0.0
    doc_threshold: float = 0.0

    def metadata(self, include_w: bool = True) -> dict:
        d = asdict(self)
        if include_w:
            d["w"] = np.asarray(self.w, dtype=np.float32).tolist()
        else:
            d.pop("w", None)
        return _jsonify(d)


@dataclass
class _SplitResult:
    w: Array
    token_threshold: float
    doc_threshold: float
    search_gain: float
    honest_gain: float
    effective_gain: float
    scores: Optional[Array] = None
    direction_index: int = -1
    search_rows: int = 0
    honest_rows: int = 0
    token_threshold_count: int = 0
    direction_count: int = 0


# ---------------------------------------------------------------------------
# Projection-distribution Newton tree
# ---------------------------------------------------------------------------

class ProjectionDistributionNewtonTree:
    """
    Best-first multiclass Newton tree over token sequences.

    Node function:
        q_it = h_it @ w
        c_i  = mean_t I[q_it >= b]       (hard empirical survival function)
        or
        c_i  = mean_t sigmoid_like(q_it-b)
        route right iff c_i >= rho

    The split is parameterized by:
        w   : projection direction
        b   : token-level projection threshold
        rho : document-level fraction/soft-fraction threshold
    """

    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        *,
        max_depth: int = 6,
        max_leaves: int = 16,
        min_samples_leaf: int = 100,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_gain: float = 1e-8,
        distribution_activation: str = "hard",
        distribution_temperature: float = 1.0,
        token_threshold_quantiles: Sequence[float] = (
            0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90
        ),
        max_doc_thresholds: int = 64,
        token_threshold_sample_rows: int = 2048,
        n_random_directions: int = 8,
        n_token_prototype_directions: int = 8,
        n_local_perturbations: int = 4,
        local_sigmas: Sequence[float] = (0.10, 0.25),
        beam_width: int = 4,
        honest_fraction: float = 0.15,
        honest_top_candidates: int = 8,
        min_honest_gain_per_sample: float = 0.0,
        score_block_rows: int = 1024,
        direction_chunk_size: int = 8,
        tree_device: str = "auto",
        threshold_n_jobs: int = 0,
        random_state: int = 42,
        stage_index: int = 0,
        tree_index: int = 0,
    ):
        if token_dim < 1:
            raise ValueError("token_dim must be >= 1")
        if output_dim < 2:
            raise ValueError("output_dim must be >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if reg_lambda <= 0:
            raise ValueError("reg_lambda must be > 0")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if distribution_activation not in ("hard", "soft"):
            raise ValueError("distribution_activation must be 'hard' or 'soft'")
        if distribution_temperature <= 0:
            raise ValueError("distribution_temperature must be > 0")
        qs = tuple(float(q) for q in token_threshold_quantiles)
        if not qs or any(q <= 0 or q >= 1 for q in qs):
            raise ValueError("token threshold quantiles must lie strictly in (0,1)")
        if not (0 <= honest_fraction < 0.5):
            raise ValueError("honest_fraction must lie in [0, 0.5)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if honest_top_candidates < 1:
            raise ValueError("honest_top_candidates must be >= 1")

        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.max_depth = int(max_depth)
        self.max_leaves = int(max_leaves)
        self.min_samples_leaf = int(min_samples_leaf)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_gain = float(min_gain)
        self.distribution_activation = str(distribution_activation)
        self.distribution_temperature = float(distribution_temperature)
        self.token_threshold_quantiles = qs
        self.max_doc_thresholds = int(max_doc_thresholds)
        self.token_threshold_sample_rows = int(token_threshold_sample_rows)
        self.n_random_directions = int(n_random_directions)
        self.n_token_prototype_directions = int(n_token_prototype_directions)
        self.n_local_perturbations = int(n_local_perturbations)
        self.local_sigmas = tuple(float(s) for s in local_sigmas if float(s) > 0)
        self.beam_width = int(beam_width)
        self.honest_fraction = float(honest_fraction)
        self.honest_top_candidates = int(honest_top_candidates)
        self.min_honest_gain_per_sample = float(min_honest_gain_per_sample)
        self.score_block_rows = int(score_block_rows)
        self.direction_chunk_size = max(1, int(direction_chunk_size))

        tree_device = str(tree_device).lower()
        if tree_device == "auto":
            tree_device = "cuda" if torch.cuda.is_available() else "cpu"
        if tree_device not in ("cpu", "cuda"):
            raise ValueError("tree_device must be one of: auto, cpu, cuda")
        if tree_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "tree_device='cuda' requested, but torch.cuda.is_available() is False"
            )
        self.tree_device = tree_device
        self._torch_device = torch.device(tree_device)

        # 0 = conservative auto. Each worker still calls the exact same
        # _best_threshold() routine; only independent candidates run in parallel.
        self.threshold_n_jobs = int(threshold_n_jobs)
        if self.threshold_n_jobs <= 0:
            self.threshold_n_jobs = max(
                1, min(8, int(os.cpu_count() or 1))
            )

        self.random_state = int(random_state)
        self.stage_index = int(stage_index)
        self.tree_index = int(tree_index)

        self.root_ = None
        self.n_leaves_ = 0
        self.depth_ = 0
        self.n_internal_nodes_ = 0
        self.search_stats_: List[dict] = []

        self._rng = np.random.default_rng(self.random_state)
        self._node_counter = 0
        self._X = None
        self._mask = None
        self._g = None
        self._h = None
        self._doc_mean = None
        # Optional stage-level CUDA cache. It is owned by the classifier and
        # shared across all trees in a stage; this tree never serializes it.
        self._stage_gpu_cache = None
        self._threshold_executor = None

    # ----------------------------- Newton objective -------------------------

    def _score_GH(self, G: Array, H: Array) -> float:
        return 0.5 * float(np.sum((G * G) / (H + self.reg_lambda)))

    def _parent_score(self, rows: Array) -> float:
        G = np.sum(self._g[rows], axis=0, dtype=np.float64)
        H = np.sum(self._h[rows], axis=0, dtype=np.float64)
        return self._score_GH(G, H)

    def _leaf_value(self, rows: Array) -> Array:
        G = np.sum(self._g[rows], axis=0, dtype=np.float64)
        H = np.sum(self._h[rows], axis=0, dtype=np.float64)
        return (-G / (H + self.reg_lambda)).astype(np.float32)

    def _best_threshold(
        self,
        rows: Array,
        scores: Array,
        parent_score: float,
        min_leaf: Optional[int] = None,
    ) -> Tuple[Optional[float], float]:
        rows = np.asarray(rows, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        n = len(rows)
        min_leaf = self.min_samples_leaf if min_leaf is None else int(min_leaf)
        if n < 2 * min_leaf:
            return None, -np.inf

        order = np.argsort(scores, kind="mergesort")
        s = scores[order]
        rr = rows[order]
        g = np.asarray(self._g[rr], dtype=np.float64)
        h = np.asarray(self._h[rr], dtype=np.float64)
        cg = np.cumsum(g, axis=0)
        ch = np.cumsum(h, axis=0)
        total_g = cg[-1]
        total_h = ch[-1]

        positions = np.arange(min_leaf, n - min_leaf + 1, dtype=np.int64)
        valid = s[positions - 1] < s[positions]
        positions = positions[valid]
        if len(positions) == 0:
            return None, -np.inf
        if len(positions) > self.max_doc_thresholds:
            take = np.linspace(
                0, len(positions) - 1, self.max_doc_thresholds, dtype=np.int64
            )
            positions = positions[np.unique(take)]

        idx = positions - 1
        GL, HL = cg[idx], ch[idx]
        GR = total_g[None, :] - GL
        HR = total_h[None, :] - HL
        gains = 0.5 * np.sum(
            (GL * GL) / (HL + self.reg_lambda)
            + (GR * GR) / (HR + self.reg_lambda),
            axis=1,
        )
        gains = gains - parent_score - self.gamma
        j = int(np.argmax(gains))
        p = int(positions[j])
        return float(0.5 * (s[p - 1] + s[p])), float(gains[j])

    def _fixed_split_gain(
        self,
        rows: Array,
        scores: Array,
        threshold: float,
        min_leaf: int,
    ) -> float:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) < 2 * min_leaf:
            return -np.inf
        right = np.asarray(scores) >= float(threshold)
        nr = int(np.sum(right))
        nl = len(rows) - nr
        if nl < min_leaf or nr < min_leaf:
            return -np.inf
        left_rows = rows[~right]
        right_rows = rows[right]
        parent = self._parent_score(rows)
        GL = np.sum(self._g[left_rows], axis=0, dtype=np.float64)
        HL = np.sum(self._h[left_rows], axis=0, dtype=np.float64)
        GR = np.sum(self._g[right_rows], axis=0, dtype=np.float64)
        HR = np.sum(self._h[right_rows], axis=0, dtype=np.float64)
        return (
            self._score_GH(GL, HL)
            + self._score_GH(GR, HR)
            - parent
            - self.gamma
        )

    # -------------------------- Direction proposals ------------------------

    def _residual_guided_directions(self, rows: Array) -> Array:
        residual = -np.asarray(self._g[rows], dtype=np.float32)
        Z = np.asarray(self._doc_mean[rows], dtype=np.float32)

        banks = []
        # Standard residual-weighted mean direction.
        banks.append(residual.T @ Z)

        # Positive-vs-negative residual prototypes.  This is not identical to
        # the first moment when residual magnitudes are highly non-uniform.
        for c in range(self.output_dim):
            r = residual[:, c]
            pos = r > 0
            neg = r < 0
            if np.any(pos) and np.any(neg):
                wp = np.maximum(r[pos], 1e-8)
                wn = np.maximum(-r[neg], 1e-8)
                mu_p = np.average(Z[pos], axis=0, weights=wp)
                mu_n = np.average(Z[neg], axis=0, weights=wn)
                banks.append((mu_p - mu_n)[None, :])

        return _normalize_rows(np.concatenate(banks, axis=0))

    def _random_directions(self, count: int) -> Array:
        if count <= 0:
            return np.empty((0, self.token_dim), dtype=np.float32)
        return _normalize_rows(
            self._rng.normal(size=(count, self.token_dim)).astype(np.float32)
        )

    def _token_prototype_directions(self, rows: Array, count: int) -> Array:
        """
        Sample actual token vectors from high-residual documents.

        These proposals do not assume that the useful token direction is visible
        in the document mean, which is important for sparse or multimodal token
        evidence.
        """
        if count <= 0 or len(rows) == 0:
            return np.empty((0, self.token_dim), dtype=np.float32)

        residual_norm = np.linalg.norm(
            np.asarray(self._g[rows], dtype=np.float64), axis=1
        )
        residual_norm = np.maximum(residual_norm, 1e-12)
        p = residual_norm / residual_norm.sum()
        replace = len(rows) < count
        chosen_local = self._rng.choice(
            len(rows), size=count, replace=replace, p=p
        )
        dirs = []
        for j in chosen_local:
            r = int(rows[int(j)])
            valid = np.flatnonzero(np.asarray(self._mask[r], dtype=bool))
            if len(valid) == 0:
                continue
            t = int(self._rng.choice(valid))
            v = np.asarray(self._X[r, t], dtype=np.float32)
            # Center using the current document mean to emphasize a token
            # deviation rather than merely the global embedding offset.
            v_centered = v - np.asarray(self._doc_mean[r], dtype=np.float32)
            if np.linalg.norm(v_centered) > 1e-8:
                dirs.append(v_centered)
            if np.linalg.norm(v) > 1e-8:
                dirs.append(v)
        if not dirs:
            return np.empty((0, self.token_dim), dtype=np.float32)
        return _normalize_rows(np.asarray(dirs, dtype=np.float32))

    def _initial_direction_bank(self, rows: Array) -> Array:
        parts = [
            self._residual_guided_directions(rows),
            self._random_directions(self.n_random_directions),
            self._token_prototype_directions(
                rows, self.n_token_prototype_directions
            ),
        ]
        W = np.concatenate([x for x in parts if len(x)], axis=0)
        # Explicitly include both signs. Distribution thresholds make +w and -w
        # genuinely different tail queries.
        W = np.concatenate([W, -W], axis=0)
        return _deduplicate_directions(W)

    def _local_bank(self, seeds: Array) -> Array:
        seeds = _normalize_rows(seeds)
        if len(seeds) == 0 or self.n_local_perturbations <= 0:
            return np.empty((0, self.token_dim), dtype=np.float32)
        out = []
        for w in seeds:
            out.append(w)
            for sigma in self.local_sigmas:
                noise = self._rng.normal(
                    size=(self.n_local_perturbations, self.token_dim)
                ).astype(np.float32)
                cand = w[None, :] + np.float32(sigma) * noise
                out.extend(cand)
        W = _normalize_rows(np.asarray(out, dtype=np.float32))
        W = np.concatenate([W, -W], axis=0)
        return _deduplicate_directions(W)

    # --------------------- Projection-distribution scoring -----------------

    def _sample_rows_for_token_thresholds(self, rows: Array) -> Array:
        rows = np.asarray(rows, dtype=np.int64)
        k = min(len(rows), self.token_threshold_sample_rows)
        if k == len(rows):
            return rows
        return self._rng.choice(rows, size=k, replace=False).astype(np.int64)

    def _gpu_rows(
        self,
        rows: Array,
        *,
        X_cpu: Optional[Array] = None,
        mask_cpu: Optional[Array] = None,
        gpu_cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return selected token rows and mask on CUDA.

        If a stage cache is supplied, values are gathered directly on the GPU.
        Otherwise the exact source rows are copied from CPU as before. In both
        cases token values are converted to float32 before matmul, matching the
        existing CUDA execution path.
        """
        if self._torch_device.type != "cuda":
            raise RuntimeError("_gpu_rows is CUDA-only")

        cache = self._stage_gpu_cache if gpu_cache is None else gpu_cache
        rows = np.asarray(rows, dtype=np.int64)

        if cache is not None:
            idxg = torch.as_tensor(
                rows,
                device=self._torch_device,
                dtype=torch.long,
            )
            xg = cache["X"].index_select(0, idxg)
            if xg.dtype != torch.float32:
                xg = xg.to(torch.float32)
            mg = cache["mask"].index_select(0, idxg)
            if mg.dtype != torch.bool:
                mg = mg.to(torch.bool)
            return xg, mg

        if X_cpu is None or mask_cpu is None:
            raise ValueError("CPU arrays are required when no GPU cache is available")

        xb = np.asarray(X_cpu[rows], dtype=np.float32)
        mb = np.asarray(mask_cpu[rows], dtype=bool)
        xg = torch.from_numpy(
            np.ascontiguousarray(xb)
        ).to(
            self._torch_device,
            dtype=torch.float32,
        )
        mg = torch.from_numpy(
            np.ascontiguousarray(mb)
        ).to(
            self._torch_device,
            dtype=torch.bool,
        )
        return xg, mg

    def _projection_bank(
        self,
        rows: Array,
        W: Array,
    ) -> Tuple[Array, Array]:
        """
        Return projection [N,T,M] and bool mask [N,T].

        CUDA path changes only the backend of H @ W. The returned values remain
        NumPy float32 so token-threshold quantiles and the rest of tree search
        keep exactly the same semantics as the CPU implementation.
        """
        rows = np.asarray(rows, dtype=np.int64)
        W = _normalize_rows(W)
        mb = np.asarray(self._mask[rows], dtype=bool)
        N = len(rows)
        T = int(self._X.shape[1])
        D = int(self._X.shape[2])

        if self._torch_device.type == "cuda":
            with torch.inference_mode():
                xg, _ = self._gpu_rows(
                    rows,
                    X_cpu=self._X,
                    mask_cpu=self._mask,
                )
                wg = torch.from_numpy(
                    np.ascontiguousarray(W)
                ).to(
                    self._torch_device,
                    dtype=torch.float32,
                )
                proj_g = torch.matmul(
                    xg.reshape(N * T, D),
                    wg.T,
                ).reshape(N, T, len(W))
                proj = (
                    proj_g.cpu().numpy().astype(np.float32, copy=False)
                )
            return proj, mb

        xb = np.asarray(self._X[rows], dtype=np.float32)
        proj = (xb.reshape(N * T, D) @ W.T).reshape(N, T, len(W))
        return proj.astype(np.float32, copy=False), mb

    def _estimate_token_thresholds(self, rows: Array, W: Array) -> Array:
        sample_rows = self._sample_rows_for_token_thresholds(rows)
        proj, mb = self._projection_bank(sample_rows, W)
        q = len(self.token_threshold_quantiles)
        out = np.empty((len(W), q), dtype=np.float32)
        for j in range(len(W)):
            vals = proj[:, :, j][mb]
            if len(vals) == 0:
                out[j] = 0.0
            else:
                out[j] = np.quantile(
                    vals.astype(np.float64),
                    self.token_threshold_quantiles,
                ).astype(np.float32)
        return out

    def _score_distribution_bank(
        self,
        rows: Array,
        W: Array,
        token_thresholds: Array,
    ) -> Array:
        """
        Return document scores [N, M, Q].

        Hard mode:
            mean_t 1[h_t @ w_m >= b_mq]

        Soft mode:
            mean_t 0.5 * (1 + tanh((h_t @ w_m - b_mq) / temperature))

        On CUDA, directions M and token thresholds Q are evaluated together.
        Only the projection/distribution statistic is moved to CUDA; Newton gain
        calculation and threshold selection remain unchanged on CPU.
        """
        rows = np.asarray(rows, dtype=np.int64)
        W = _normalize_rows(W)
        Bmat = np.asarray(token_thresholds, dtype=np.float32)

        n = len(rows)
        m = len(W)
        q = Bmat.shape[1]
        T = int(self._X.shape[1])
        out = np.empty((n, m, q), dtype=np.float32)

        if self._torch_device.type == "cuda":
            with torch.inference_mode():
                Wg = torch.from_numpy(
                    np.ascontiguousarray(W)
                ).to(
                    self._torch_device,
                    dtype=torch.float32,
                )
                Bg = torch.from_numpy(
                    np.ascontiguousarray(Bmat)
                ).to(
                    self._torch_device,
                    dtype=torch.float32,
                )

                for start in range(0, n, self.score_block_rows):
                    end = min(n, start + self.score_block_rows)
                    rr = rows[start:end]

                    nb = len(rr)
                    xg, mg = self._gpu_rows(
                        rr,
                        X_cpu=self._X,
                        mask_cpu=self._mask,
                    )

                    # [B*T,D] @ [D,M] -> [B,T,M]
                    proj = torch.matmul(
                        xg.reshape(nb * T, self.token_dim),
                        Wg.T,
                    ).reshape(nb, T, m)

                    denom = (
                        mg.sum(dim=1)
                        .clamp_min(1)
                        .to(torch.float32)
                    )

                    # Broadcast to [B,T,M,Q].
                    if self.distribution_activation == "hard":
                        hits = (
                            proj.unsqueeze(-1)
                            >= Bg[None, None, :, :]
                        )
                        hits = hits & mg[:, :, None, None]
                        scores = (
                            hits.sum(dim=1).to(torch.float32)
                            / denom[:, None, None]
                        )
                    else:
                        z = (
                            proj.unsqueeze(-1)
                            - Bg[None, None, :, :]
                        ) / float(self.distribution_temperature)
                        values = 0.5 * (1.0 + torch.tanh(z))
                        values = (
                            values
                            * mg[:, :, None, None].to(torch.float32)
                        )
                        scores = (
                            values.sum(dim=1)
                            / denom[:, None, None]
                        )

                    out[start:end] = (
                        scores.cpu().numpy().astype(np.float32, copy=False)
                    )
            return out

        # CPU fallback: original exact implementation.
        for start in range(0, n, self.score_block_rows):
            end = min(n, start + self.score_block_rows)
            rr = rows[start:end]
            xb = np.asarray(self._X[rr], dtype=np.float32)
            mb = np.asarray(self._mask[rr], dtype=bool)
            nb = len(rr)
            proj = (xb.reshape(nb * T, self.token_dim) @ W.T).reshape(
                nb, T, m
            )
            denom = np.maximum(mb.sum(axis=1), 1).astype(np.float32)
            for j in range(m):
                p = proj[:, :, j]
                for k in range(q):
                    b = Bmat[j, k]
                    if self.distribution_activation == "hard":
                        a = (p >= b) & mb
                        out[start:end, j, k] = (
                            a.sum(axis=1).astype(np.float32) / denom
                        )
                    else:
                        z = (
                            p - np.float32(b)
                        ) / np.float32(self.distribution_temperature)
                        a = 0.5 * (1.0 + np.tanh(z))
                        a = a * mb.astype(np.float32)
                        out[start:end, j, k] = (
                            a.sum(axis=1, dtype=np.float32) / denom
                        )
        return out

    def _score_one_rule(
        self,
        rows: Array,
        w: Array,
        token_threshold: float,
    ) -> Array:
        W = _normalize_rows(np.asarray(w, dtype=np.float32)[None, :])
        B = np.asarray([[token_threshold]], dtype=np.float32)
        return self._score_distribution_bank(rows, W, B)[:, 0, 0]

    # --------------------------- Candidate search --------------------------

    def _split_search_honest_rows(self, rows: Array) -> Tuple[Array, Array]:
        rows = np.asarray(rows, dtype=np.int64)
        if self.honest_fraction <= 0:
            return rows, np.empty(0, dtype=np.int64)

        n_honest = int(round(self.honest_fraction * len(rows)))
        # Ensure the search subset can still produce legal children.
        if (
            n_honest < 2 * max(5, int(self.min_samples_leaf * self.honest_fraction * 0.5))
            or len(rows) - n_honest < 2 * self.min_samples_leaf
        ):
            return rows, np.empty(0, dtype=np.int64)

        perm = self._rng.permutation(len(rows))
        honest = rows[perm[:n_honest]]
        search = rows[perm[n_honest:]]
        return search, honest

    def _evaluate_direction_bank_search(
        self,
        search_rows: Array,
        W: Array,
    ) -> List[_SplitResult]:
        """
        Exact candidate evaluation with bounded memory.

        The direction bank is processed in chunks. This changes only memory
        usage, not which candidates/thresholds are evaluated.
        """
        W = _deduplicate_directions(W)
        if len(W) == 0:
            return []
        parent = self._parent_score(search_rows)
        candidates: List[_SplitResult] = []
        total_directions = len(W)

        for chunk_start in range(0, total_directions, self.direction_chunk_size):
            chunk_end = min(
                total_directions, chunk_start + self.direction_chunk_size
            )
            Wc = W[chunk_start:chunk_end]
            token_bs = self._estimate_token_thresholds(search_rows, Wc)
            S = self._score_distribution_bank(search_rows, Wc, token_bs)

            # Every (direction, token-threshold) candidate is independent.
            # We call the *same* _best_threshold() function in parallel and
            # collect results in j-major/k-major order. ThreadPoolExecutor.map
            # preserves input order, so tie behavior remains identical.
            tasks = [
                (j, k)
                for j in range(len(Wc))
                for k in range(token_bs.shape[1])
            ]

            def eval_threshold(task):
                j, k = task
                return self._best_threshold(
                    search_rows,
                    S[:, j, k],
                    parent,
                    min_leaf=self.min_samples_leaf,
                )

            if self._threshold_executor is not None and len(tasks) > 1:
                results = list(
                    self._threshold_executor.map(eval_threshold, tasks)
                )
            else:
                results = [eval_threshold(task) for task in tasks]

            result_idx = 0
            for j in range(len(Wc)):
                best = None
                for k in range(token_bs.shape[1]):
                    rho, gain = results[result_idx]
                    result_idx += 1
                    if rho is None:
                        continue
                    if best is None or gain > best.search_gain:
                        best = _SplitResult(
                            w=Wc[j].copy(),
                            token_threshold=float(token_bs[j, k]),
                            doc_threshold=float(rho),
                            search_gain=float(gain),
                            honest_gain=float("nan"),
                            effective_gain=float(gain),
                            direction_index=chunk_start + j,
                            search_rows=len(search_rows),
                            honest_rows=0,
                            token_threshold_count=token_bs.shape[1],
                            direction_count=total_directions,
                        )
                if best is not None:
                    candidates.append(best)

        candidates.sort(key=lambda x: x.search_gain, reverse=True)
        return candidates

    def _honest_select(
        self,
        search_rows: Array,
        honest_rows: Array,
        candidates: List[_SplitResult],
    ) -> Optional[_SplitResult]:
        if not candidates:
            return None
        if len(honest_rows) == 0:
            return candidates[0]

        min_honest_leaf = max(
            5,
            int(round(self.min_samples_leaf * self.honest_fraction * 0.5)),
        )
        top = candidates[: self.honest_top_candidates]
        robust: List[_SplitResult] = []
        for c in top:
            hs = self._score_one_rule(
                honest_rows, c.w, c.token_threshold
            )
            hgain = self._fixed_split_gain(
                honest_rows,
                hs,
                c.doc_threshold,
                min_leaf=min_honest_leaf,
            )
            c.honest_gain = float(hgain)
            c.honest_rows = len(honest_rows)
            if not np.isfinite(hgain):
                continue

            s_per = c.search_gain / max(len(search_rows), 1)
            h_per = hgain / max(len(honest_rows), 1)
            if h_per <= self.min_honest_gain_per_sample:
                continue

            # Conservative robust score: a split is only as strong as its weaker
            # independently measured per-sample gain.
            c.effective_gain = float(
                min(s_per, h_per) * (len(search_rows) + len(honest_rows))
            )
            robust.append(c)

        if not robust:
            return None
        robust.sort(key=lambda x: x.effective_gain, reverse=True)
        return robust[0]

    def _find_best_split(self, rows: Array) -> Optional[_SplitResult]:
        if len(rows) < 2 * self.min_samples_leaf:
            return None

        search_rows, honest_rows = self._split_search_honest_rows(rows)

        W0 = self._initial_direction_bank(search_rows)
        first = self._evaluate_direction_bank_search(search_rows, W0)
        if not first:
            return None

        pool = first[: max(self.beam_width, self.honest_top_candidates)]
        seeds = np.asarray(
            [c.w for c in first[: self.beam_width]], dtype=np.float32
        )
        Wlocal = self._local_bank(seeds)
        if len(Wlocal):
            local = self._evaluate_direction_bank_search(
                search_rows, Wlocal
            )
            pool.extend(local[: max(self.beam_width, self.honest_top_candidates)])

        # Rank unique rule proposals by search gain before honest checking.
        pool.sort(key=lambda x: x.search_gain, reverse=True)
        best = self._honest_select(search_rows, honest_rows, pool)
        if best is None:
            return None

        # Verify the selected rule on all node samples, both for min leaf and
        # for actual routing. The best-first priority remains the robust gain.
        all_scores = self._score_one_rule(
            rows, best.w, best.token_threshold
        )
        right = all_scores >= best.doc_threshold
        if (
            int(np.sum(right)) < self.min_samples_leaf
            or int(np.sum(~right)) < self.min_samples_leaf
        ):
            return None

        if not np.isfinite(best.effective_gain) or best.effective_gain <= self.min_gain:
            return None

        best.scores = all_scores.astype(np.float32, copy=False)
        self.search_stats_.append(
            {
                "stage": self.stage_index,
                "tree": self.tree_index,
                "node_rows": int(len(rows)),
                "search_rows": int(len(search_rows)),
                "honest_rows": int(len(honest_rows)),
                "initial_directions": int(len(W0)),
                "local_directions": int(len(Wlocal)),
                "search_gain": float(best.search_gain),
                "honest_gain": float(best.honest_gain),
                "effective_gain": float(best.effective_gain),
                "token_threshold": float(best.token_threshold),
                "doc_threshold": float(best.doc_threshold),
            }
        )
        return best

    # ------------------------------ Tree fit -------------------------------

    def _new_leaf(self, rows: Array, depth: int) -> dict:
        node_id = self._node_counter
        self._node_counter += 1
        return {
            "node_id": int(node_id),
            "isleaf": True,
            "value": self._leaf_value(rows),
            "depth": int(depth),
            "n_samples": int(len(rows)),
        }

    def fit(
        self,
        X_tokens: Array,
        attention_mask: Array,
        gradients: Array,
        hessians: Array,
        doc_mean: Optional[Array] = None,
        sample_weight: Optional[Array] = None,
        stage_gpu_cache: Optional[dict] = None,
    ):
        X = X_tokens
        mask = np.asarray(attention_mask)
        g = np.asarray(gradients, dtype=np.float32)
        h = np.asarray(hessians, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError("X_tokens must have shape [N,T,D]")
        if X.shape[2] != self.token_dim:
            raise ValueError(
                f"expected token_dim={self.token_dim}, got {X.shape[2]}"
            )
        if mask.shape != X.shape[:2]:
            raise ValueError("attention_mask must have shape [N,T]")
        if g.shape != (len(X), self.output_dim):
            raise ValueError("gradients must have shape [N,C]")
        if h.shape != g.shape:
            raise ValueError("hessians must have shape [N,C]")
        if np.any(h < 0):
            raise ValueError("hessians must be non-negative")

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=np.float32)
            if sw.shape != (len(X),):
                raise ValueError("sample_weight must have shape [N]")
            g = g * sw[:, None]
            h = h * sw[:, None]

        self._X = X
        self._mask = mask
        self._g = g
        self._h = h
        self._stage_gpu_cache = stage_gpu_cache
        self._threshold_executor = (
            ThreadPoolExecutor(max_workers=self.threshold_n_jobs)
            if self.threshold_n_jobs > 1
            else None
        )
        self._doc_mean = (
            masked_mean_tokens(X, mask, block_rows=self.score_block_rows)
            if doc_mean is None
            else np.asarray(doc_mean, dtype=np.float32)
        )

        rows_root = np.arange(len(X), dtype=np.int64)
        self._node_counter = 0
        self.root_ = self._new_leaf(rows_root, 0)
        self.n_leaves_ = 1
        self.depth_ = 0
        self.n_internal_nodes_ = 0

        heap = []
        counter = 0

        def maybe_push(node, rows):
            nonlocal counter
            if node["depth"] >= self.max_depth:
                return
            if len(rows) < 2 * self.min_samples_leaf:
                return
            split = self._find_best_split(rows)
            if split is None:
                return
            heapq.heappush(
                heap,
                (-split.effective_gain, counter, node, rows, split),
            )
            counter += 1

        maybe_push(self.root_, rows_root)

        while heap and self.n_leaves_ < self.max_leaves:
            _, _, node, rows, split = heapq.heappop(heap)
            if not node["isleaf"]:
                continue
            right_mask = split.scores >= split.doc_threshold
            left_rows = rows[~right_mask]
            right_rows = rows[right_mask]
            if (
                len(left_rows) < self.min_samples_leaf
                or len(right_rows) < self.min_samples_leaf
            ):
                continue

            depth = int(node["depth"])
            node_id = int(node["node_id"])
            left = self._new_leaf(left_rows, depth + 1)
            right = self._new_leaf(right_rows, depth + 1)
            node.clear()
            node.update(
                {
                    "node_id": node_id,
                    "isleaf": False,
                    "w": split.w.astype(np.float32, copy=True),
                    "token_threshold": float(split.token_threshold),
                    "doc_threshold": float(split.doc_threshold),
                    "search_gain": float(split.search_gain),
                    "honest_gain": float(split.honest_gain),
                    "effective_gain": float(split.effective_gain),
                    "depth": depth,
                    "n_samples": int(len(rows)),
                    "left": left,
                    "right": right,
                }
            )
            self.n_leaves_ += 1
            self.n_internal_nodes_ += 1
            self.depth_ = max(self.depth_, depth + 1)
            maybe_push(left, left_rows)
            maybe_push(right, right_rows)

        self._X = None
        self._mask = None
        self._g = None
        self._h = None
        self._doc_mean = None
        self._stage_gpu_cache = None
        if self._threshold_executor is not None:
            self._threshold_executor.shutdown(wait=True)
            self._threshold_executor = None
        return self

    # ------------------------------ Inference ------------------------------

    def _score_external(
        self,
        X_tokens: Array,
        attention_mask: Array,
        rows: Array,
        w: Array,
        token_threshold: float,
        gpu_cache: Optional[dict] = None,
    ) -> Array:
        """
        Score one already-selected node rule for routing/prediction.

        CUDA path is used during train/eval prediction after each fitted tree,
        which avoids leaving the GPU idle between candidate-search phases.
        """
        X = X_tokens
        mask = np.asarray(attention_mask)
        rows = np.asarray(rows, dtype=np.int64)
        w = np.asarray(w, dtype=np.float32)
        n = len(rows)
        T = int(X.shape[1])
        out = np.empty(n, dtype=np.float32)

        if self._torch_device.type == "cuda":
            with torch.inference_mode():
                wg = torch.from_numpy(
                    np.ascontiguousarray(w)
                ).to(
                    self._torch_device,
                    dtype=torch.float32,
                )

                for start in range(0, n, self.score_block_rows):
                    end = min(n, start + self.score_block_rows)
                    rr = rows[start:end]
                    nb = len(rr)
                    xg, mg = self._gpu_rows(
                        rr,
                        X_cpu=X,
                        mask_cpu=mask,
                        gpu_cache=gpu_cache,
                    )

                    pg = torch.matmul(
                        xg.reshape(nb * T, self.token_dim),
                        wg,
                    ).reshape(nb, T)

                    denom = (
                        mg.sum(dim=1)
                        .clamp_min(1)
                        .to(torch.float32)
                    )

                    if self.distribution_activation == "hard":
                        scores = (
                            ((pg >= float(token_threshold)) & mg)
                            .sum(dim=1)
                            .to(torch.float32)
                            / denom
                        )
                    else:
                        z = (
                            pg - float(token_threshold)
                        ) / float(self.distribution_temperature)
                        a = 0.5 * (1.0 + torch.tanh(z))
                        a = a * mg.to(torch.float32)
                        scores = a.sum(dim=1) / denom

                    out[start:end] = (
                        scores.cpu().numpy().astype(np.float32, copy=False)
                    )
            return out

        # CPU fallback.
        for start in range(0, n, self.score_block_rows):
            end = min(n, start + self.score_block_rows)
            rr = rows[start:end]
            xb = np.asarray(X[rr], dtype=np.float32)
            mb = np.asarray(mask[rr], dtype=bool)
            nb = len(rr)
            p = (xb.reshape(nb * T, self.token_dim) @ w).reshape(nb, T)
            denom = np.maximum(mb.sum(axis=1), 1).astype(np.float32)
            if self.distribution_activation == "hard":
                a = (p >= token_threshold) & mb
                out[start:end] = a.sum(axis=1).astype(np.float32) / denom
            else:
                z = (
                    p - np.float32(token_threshold)
                ) / np.float32(self.distribution_temperature)
                a = 0.5 * (1.0 + np.tanh(z))
                a *= mb.astype(np.float32)
                out[start:end] = a.sum(axis=1, dtype=np.float32) / denom
        return out

    def predict_values(
        self,
        X_tokens: Array,
        attention_mask: Array,
        gpu_cache: Optional[dict] = None,
    ) -> Array:
        X = X_tokens
        mask = np.asarray(attention_mask)
        if self.root_ is None:
            raise RuntimeError("tree has not been fitted")
        if X.ndim != 3 or X.shape[2] != self.token_dim:
            raise ValueError("token feature shape mismatch")
        if mask.shape != X.shape[:2]:
            raise ValueError("attention mask shape mismatch")

        out = np.empty((len(X), self.output_dim), dtype=np.float32)
        rows_root = np.arange(len(X), dtype=np.int64)
        stack = [(self.root_, rows_root)]
        while stack:
            node, rows = stack.pop()
            if len(rows) == 0:
                continue
            if node["isleaf"]:
                out[rows] = np.asarray(node["value"], dtype=np.float32)
                continue
            scores = self._score_external(
                X,
                mask,
                rows,
                node["w"],
                node["token_threshold"],
                gpu_cache=gpu_cache,
            )
            right = scores >= node["doc_threshold"]
            stack.append((node["left"], rows[~right]))
            stack.append((node["right"], rows[right]))
        return out

    # ----------------------------- Inspection ------------------------------

    def get_depth(self) -> int:
        return int(self.depth_)

    def get_n_leaves(self) -> int:
        return int(self.n_leaves_)

    def get_n_internal_nodes(self) -> int:
        return int(self.n_internal_nodes_)

    def internal_nodes(self) -> List[dict]:
        if self.root_ is None:
            return []
        out = []
        stack = [self.root_]
        while stack:
            node = stack.pop()
            if node["isleaf"]:
                continue
            out.append(node)
            stack.append(node["left"])
            stack.append(node["right"])
        return out

    def _node_to_dict(self, node: dict, include_w: bool = True) -> dict:
        if node["isleaf"]:
            return {
                "node_id": int(node["node_id"]),
                "isleaf": True,
                "depth": int(node["depth"]),
                "n_samples": int(node["n_samples"]),
                "value": np.asarray(node["value"], dtype=np.float32).tolist(),
            }
        d = {
            "node_id": int(node["node_id"]),
            "isleaf": False,
            "depth": int(node["depth"]),
            "n_samples": int(node["n_samples"]),
            "token_threshold": float(node["token_threshold"]),
            "doc_threshold": float(node["doc_threshold"]),
            "search_gain": float(node["search_gain"]),
            "honest_gain": float(node["honest_gain"]),
            "effective_gain": float(node["effective_gain"]),
            "left": self._node_to_dict(node["left"], include_w),
            "right": self._node_to_dict(node["right"], include_w),
        }
        if include_w:
            d["w"] = np.asarray(node["w"], dtype=np.float32).tolist()
        return d

    def to_structure_dict(self, include_w: bool = True) -> dict:
        return {
            "stage": self.stage_index,
            "tree": self.tree_index,
            "token_dim": self.token_dim,
            "output_dim": self.output_dim,
            "depth": self.get_depth(),
            "leaves": self.get_n_leaves(),
            "internal_nodes": self.get_n_internal_nodes(),
            "distribution_activation": self.distribution_activation,
            "root": self._node_to_dict(self.root_, include_w=include_w),
            "search_stats": _jsonify(self.search_stats_),
        }


# ---------------------------------------------------------------------------
# Stage-wise representation growth
# ---------------------------------------------------------------------------

def _concept_activation_block(
    xb: Array,
    mb: Array,
    concepts: Sequence[TokenConcept],
) -> Array:
    if not concepts:
        return np.empty((len(xb), xb.shape[1], 0), dtype=np.float32)
    W = np.stack([np.asarray(c.w, dtype=np.float32) for c in concepts], axis=0)
    b = np.asarray([c.token_threshold for c in concepts], dtype=np.float32)
    temp = np.asarray([c.temperature for c in concepts], dtype=np.float32)
    mean = np.asarray([c.feature_mean for c in concepts], dtype=np.float32)
    scale = np.asarray([max(c.feature_scale, 1e-6) for c in concepts], dtype=np.float32)
    B, T, D = xb.shape
    proj = (xb.reshape(B * T, D) @ W.T).reshape(B, T, len(concepts))
    act = np.tanh(
        (proj - b[None, None, :]) / temp[None, None, :]
    ).astype(np.float32)
    act = (act - mean[None, None, :]) / scale[None, None, :]
    act *= mb[:, :, None].astype(np.float32)
    return act


def append_concept_features(
    X_tokens: Array,
    attention_mask: Array,
    concepts: Sequence[TokenConcept],
    *,
    output_path: Optional[Union[str, Path]] = None,
    output_dtype: str = "float16",
    block_rows: int = 512,
    prefix_mixing: bool = False,
) -> Array:
    if not concepts:
        return X_tokens
    X = X_tokens
    mask = np.asarray(attention_mask)
    n, T, d = X.shape
    m = len(concepts)
    added = m * (2 if prefix_mixing else 1)
    out_shape = (n, T, d + added)
    dtype = np.float16 if output_dtype == "float16" else np.float32

    if output_path is None:
        out = np.empty(out_shape, dtype=dtype)
    else:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            p, mode="w+", dtype=dtype, shape=out_shape
        )

    for start in range(0, n, block_rows):
        end = min(n, start + block_rows)
        xb = np.asarray(X[start:end], dtype=np.float32)
        mb = np.asarray(mask[start:end], dtype=bool)
        a = _concept_activation_block(xb, mb, concepts)
        out[start:end, :, :d] = xb.astype(dtype, copy=False)
        if prefix_mixing:
            valid = mb[:, :, None].astype(np.float32)
            prefix_num = np.cumsum(a * valid, axis=1, dtype=np.float32)
            prefix_den = np.maximum(
                np.cumsum(valid, axis=1, dtype=np.float32), 1.0
            )
            prefix = (prefix_num / prefix_den) * valid
            extra = np.concatenate([a, prefix], axis=2)
        else:
            extra = a
        out[start:end, :, d:] = extra.astype(dtype, copy=False)

    if isinstance(out, np.memmap):
        out.flush()
        path = Path(output_path)
        del out
        return np.load(path, mmap_mode="r")
    return out


def fit_concept_standardization(
    X_tokens: Array,
    attention_mask: Array,
    concepts: Sequence[TokenConcept],
    *,
    max_rows: int = 10000,
    block_rows: int = 512,
    random_state: int = 42,
) -> None:
    if not concepts:
        return
    rng = np.random.default_rng(random_state)
    n = len(X_tokens)
    rows = (
        np.arange(n, dtype=np.int64)
        if n <= max_rows
        else rng.choice(n, size=max_rows, replace=False).astype(np.int64)
    )
    W = np.stack([np.asarray(c.w, dtype=np.float32) for c in concepts], axis=0)
    b = np.asarray([c.token_threshold for c in concepts], dtype=np.float32)
    temp = np.asarray([c.temperature for c in concepts], dtype=np.float32)
    sums = np.zeros(len(concepts), dtype=np.float64)
    sums2 = np.zeros(len(concepts), dtype=np.float64)
    counts = np.zeros(len(concepts), dtype=np.float64)

    T = int(X_tokens.shape[1])
    for start in range(0, len(rows), block_rows):
        end = min(len(rows), start + block_rows)
        rr = rows[start:end]
        xb = np.asarray(X_tokens[rr], dtype=np.float32)
        mb = np.asarray(attention_mask[rr], dtype=bool)
        B, _, D = xb.shape
        proj = (xb.reshape(B * T, D) @ W.T).reshape(B, T, len(concepts))
        a = np.tanh(
            (proj - b[None, None, :]) / temp[None, None, :]
        ).astype(np.float32)
        valid = mb[:, :, None]
        sums += np.sum(np.where(valid, a, 0.0), axis=(0, 1), dtype=np.float64)
        sums2 += np.sum(
            np.where(valid, a * a, 0.0), axis=(0, 1), dtype=np.float64
        )
        counts += np.sum(valid, axis=(0, 1), dtype=np.float64)

    means = sums / np.maximum(counts, 1.0)
    vars_ = sums2 / np.maximum(counts, 1.0) - means * means
    stds = np.sqrt(np.maximum(vars_, 1e-6))
    for j, c in enumerate(concepts):
        c.feature_mean = float(means[j])
        c.feature_scale = float(stds[j])


def concept_statistics(
    X_tokens: Array,
    attention_mask: Array,
    y: Array,
    concepts: Sequence[TokenConcept],
    *,
    max_rows: int = 10000,
    block_rows: int = 512,
    random_state: int = 42,
) -> List[dict]:
    if not concepts:
        return []
    rng = np.random.default_rng(random_state)
    n = len(X_tokens)
    rows = (
        np.arange(n, dtype=np.int64)
        if n <= max_rows
        else rng.choice(n, size=max_rows, replace=False).astype(np.int64)
    )
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    results = []
    for j, c in enumerate(concepts):
        doc_scores = np.empty(len(rows), dtype=np.float32)
        token_sum = 0.0
        token_sum2 = 0.0
        token_count = 0
        for start in range(0, len(rows), block_rows):
            end = min(len(rows), start + block_rows)
            rr = rows[start:end]
            xb = np.asarray(X_tokens[rr], dtype=np.float32)
            mb = np.asarray(attention_mask[rr], dtype=bool)
            B, T, D = xb.shape
            p = (
                xb.reshape(B * T, D) @ np.asarray(c.w, dtype=np.float32)
            ).reshape(B, T)
            hard = (p >= c.token_threshold) & mb
            denom = np.maximum(mb.sum(axis=1), 1)
            doc_scores[start:end] = hard.sum(axis=1) / denom
            a = np.tanh((p - c.token_threshold) / c.temperature)
            vals = a[mb]
            token_sum += float(vals.sum(dtype=np.float64))
            token_sum2 += float((vals.astype(np.float64) ** 2).sum())
            token_count += int(len(vals))

        item = c.metadata(include_w=False)
        item.update(
            {
                "doc_fraction_mean": float(np.mean(doc_scores)),
                "doc_fraction_std": float(np.std(doc_scores)),
                "token_activation_mean": float(token_sum / max(token_count, 1)),
                "token_activation_std": float(
                    math.sqrt(
                        max(
                            token_sum2 / max(token_count, 1)
                            - (token_sum / max(token_count, 1)) ** 2,
                            0.0,
                        )
                    )
                ),
                "class_doc_fraction_mean": {
                    str(int(cls)): float(
                        np.mean(doc_scores[y[rows] == cls])
                    )
                    for cls in classes
                    if np.any(y[rows] == cls)
                },
            }
        )
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Deep token forest classifier
# ---------------------------------------------------------------------------

class DeepTokenForestClassifier:
    """
    Stage-wise projection-distribution Newton boosting with token feature growth.

    Global logits are additive across all trees.  Representation growth happens
    only between stages, so every tree in one stage sees the same H^(s).
    """

    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        *,
        n_stages: int = 3,
        trees_per_stage: int = 20,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_leaves: int = 16,
        min_samples_leaf: int = 100,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        min_gain: float = 1e-8,
        distribution_activation: str = "hard",
        distribution_temperature: float = 1.0,
        token_threshold_quantiles: Sequence[float] = (
            0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90
        ),
        max_doc_thresholds: int = 64,
        token_threshold_sample_rows: int = 2048,
        n_random_directions: int = 8,
        n_token_prototype_directions: int = 8,
        n_local_perturbations: int = 4,
        local_sigmas: Sequence[float] = (0.10, 0.25),
        beam_width: int = 4,
        honest_fraction: float = 0.15,
        honest_top_candidates: int = 8,
        min_honest_gain_per_sample: float = 0.0,
        concepts_per_stage: int = 16,
        concept_temperature: float = 0.5,
        feature_growth: bool = True,
        prefix_mixing: bool = False,
        feature_dtype: str = "float16",
        score_block_rows: int = 1024,
        direction_chunk_size: int = 8,
        tree_device: str = "auto",
        threshold_n_jobs: int = 0,
        cache_stage_on_gpu: bool = True,
        gpu_cache_fraction: float = 0.55,
        transform_block_rows: int = 512,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        diagnostics_dir: Optional[Union[str, Path]] = None,
        random_state: int = 42,
    ):
        if n_stages < 1 or trees_per_stage < 1:
            raise ValueError("n_stages and trees_per_stage must be >= 1")
        if concepts_per_stage < 0:
            raise ValueError("concepts_per_stage must be >= 0")
        if concept_temperature <= 0:
            raise ValueError("concept_temperature must be > 0")

        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.n_stages = int(n_stages)
        self.trees_per_stage = int(trees_per_stage)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.max_leaves = int(max_leaves)
        self.min_samples_leaf = int(min_samples_leaf)
        self.reg_lambda = float(reg_lambda)
        self.gamma = float(gamma)
        self.min_gain = float(min_gain)
        self.distribution_activation = distribution_activation
        self.distribution_temperature = float(distribution_temperature)
        self.token_threshold_quantiles = tuple(token_threshold_quantiles)
        self.max_doc_thresholds = int(max_doc_thresholds)
        self.token_threshold_sample_rows = int(token_threshold_sample_rows)
        self.n_random_directions = int(n_random_directions)
        self.n_token_prototype_directions = int(n_token_prototype_directions)
        self.n_local_perturbations = int(n_local_perturbations)
        self.local_sigmas = tuple(local_sigmas)
        self.beam_width = int(beam_width)
        self.honest_fraction = float(honest_fraction)
        self.honest_top_candidates = int(honest_top_candidates)
        self.min_honest_gain_per_sample = float(min_honest_gain_per_sample)
        self.concepts_per_stage = int(concepts_per_stage)
        self.concept_temperature = float(concept_temperature)
        self.feature_growth = bool(feature_growth)
        self.prefix_mixing = bool(prefix_mixing)
        self.feature_dtype = str(feature_dtype)
        self.score_block_rows = int(score_block_rows)
        self.direction_chunk_size = max(1, int(direction_chunk_size))
        self.tree_device = str(tree_device).lower()
        if self.tree_device == "auto":
            self.tree_device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.tree_device not in ("cpu", "cuda"):
            raise ValueError("tree_device must be one of: auto, cpu, cuda")
        if self.tree_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "tree_device='cuda' requested, but torch.cuda.is_available() is False"
            )
        self.threshold_n_jobs = int(threshold_n_jobs)
        if self.threshold_n_jobs <= 0:
            self.threshold_n_jobs = max(1, min(8, int(os.cpu_count() or 1)))
        self.cache_stage_on_gpu = bool(cache_stage_on_gpu)
        self.gpu_cache_fraction = float(gpu_cache_fraction)
        if not (0.0 < self.gpu_cache_fraction < 0.90):
            raise ValueError("gpu_cache_fraction must lie in (0, 0.90)")
        self.transform_block_rows = int(transform_block_rows)
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.diagnostics_dir = (
            None if diagnostics_dir is None else Path(diagnostics_dir)
        )
        self.random_state = int(random_state)

        self.base_logits_: Optional[Array] = None
        self.stages_: List[dict] = []
        self.history_: List[dict] = []
        self.best_iteration_: Optional[int] = None
        self.best_eval_loss_: Optional[float] = None
        self.final_token_dim_: int = self.token_dim

    @staticmethod
    def _base_logits(y: Array, sample_weight: Optional[Array] = None) -> Array:
        y = np.asarray(y, dtype=np.int64)
        C = int(np.max(y)) + 1
        if sample_weight is None:
            counts = np.bincount(y, minlength=C).astype(np.float64)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
            counts = np.bincount(y, weights=sw, minlength=C).astype(np.float64)
        p = (counts + 1e-6) / (counts.sum() + 1e-6 * C)
        return np.log(p).astype(np.float32)

    def _gradient_hessian(self, y: Array, logits: Array) -> Tuple[Array, Array]:
        p = _softmax(logits)
        g = p.copy()
        g[np.arange(len(y)), y] -= 1.0
        h = p * (1.0 - p)
        return g.astype(np.float32), h.astype(np.float32)

    def _tree_kwargs(self, token_dim: int, stage: int, tree: int) -> dict:
        return {
            "token_dim": token_dim,
            "output_dim": self.output_dim,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "min_samples_leaf": self.min_samples_leaf,
            "reg_lambda": self.reg_lambda,
            "gamma": self.gamma,
            "min_gain": self.min_gain,
            "distribution_activation": self.distribution_activation,
            "distribution_temperature": self.distribution_temperature,
            "token_threshold_quantiles": self.token_threshold_quantiles,
            "max_doc_thresholds": self.max_doc_thresholds,
            "token_threshold_sample_rows": self.token_threshold_sample_rows,
            "n_random_directions": self.n_random_directions,
            "n_token_prototype_directions": self.n_token_prototype_directions,
            "n_local_perturbations": self.n_local_perturbations,
            "local_sigmas": self.local_sigmas,
            "beam_width": self.beam_width,
            "honest_fraction": self.honest_fraction,
            "honest_top_candidates": self.honest_top_candidates,
            "min_honest_gain_per_sample": self.min_honest_gain_per_sample,
            "score_block_rows": self.score_block_rows,
            "direction_chunk_size": self.direction_chunk_size,
            "tree_device": self.tree_device,
            "threshold_n_jobs": self.threshold_n_jobs,
            "random_state": self.random_state + 100003 * stage + 1009 * tree,
            "stage_index": stage,
            "tree_index": tree,
        }

    def _select_concepts(
        self,
        trees: Sequence[ProjectionDistributionNewtonTree],
        stage: int,
        total_rows: int,
    ) -> List[TokenConcept]:
        candidates: List[TokenConcept] = []
        for tree_idx, tree in enumerate(trees):
            for node in tree.internal_nodes():
                candidates.append(
                    TokenConcept(
                        w=np.asarray(node["w"], dtype=np.float32).copy(),
                        token_threshold=float(node["token_threshold"]),
                        temperature=self.concept_temperature,
                        stage=stage,
                        tree=tree_idx,
                        node_id=int(node["node_id"]),
                        depth=int(node["depth"]),
                        n_samples=int(node["n_samples"]),
                        search_gain=float(node["search_gain"]),
                        honest_gain=float(node["honest_gain"]),
                        effective_gain=float(node["effective_gain"]),
                        doc_threshold=float(node["doc_threshold"]),
                    )
                )

        # Effective gain already incorporates node sample count. A tiny coverage
        # preference prevents very small leaves with noisy gains from dominating.
        def utility(c: TokenConcept) -> float:
            coverage = c.n_samples / max(total_rows, 1)
            return c.effective_gain * math.sqrt(max(coverage, 1e-12))

        candidates.sort(key=utility, reverse=True)
        return candidates[: self.concepts_per_stage]

    def _build_stage_gpu_cache(
        self,
        X_tokens: Array,
        attention_mask: Array,
        *,
        label: str,
        budget_bytes: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Copy one representation stage to CUDA exactly once.

        Source dtype is preserved in the cache (normally float16). Each scoring
        block is cast to float32 immediately before matmul, exactly as in the
        uncached CUDA path. This removes repeated PCIe transfers without changing
        candidate directions, thresholds, gains, or tree-growing order.
        """
        if self.tree_device != "cuda" or not self.cache_stage_on_gpu:
            return None

        X = X_tokens
        M = np.asarray(attention_mask)
        x_bytes = int(np.prod(X.shape)) * int(np.dtype(X.dtype).itemsize)
        mask_bytes = int(np.prod(M.shape))  # cached as bool/uint8: one byte
        needed = x_bytes + mask_bytes

        free_bytes, _ = torch.cuda.mem_get_info()
        if budget_bytes is None:
            budget_bytes = int(free_bytes * self.gpu_cache_fraction)

        if needed > min(int(free_bytes * 0.85), int(budget_bytes)):
            print(
                f"[GPU cache] {label}: skip; need={needed/2**30:.2f} GiB, "
                f"free={free_bytes/2**30:.2f} GiB, "
                f"budget={budget_bytes/2**30:.2f} GiB",
                flush=True,
            )
            return None

        if np.dtype(X.dtype) == np.float16:
            tdtype = torch.float16
        elif np.dtype(X.dtype) == np.float32:
            tdtype = torch.float32
        else:
            # Preserve values via float32 for uncommon source dtypes.
            tdtype = torch.float32

        Xg = torch.empty(
            tuple(X.shape),
            device="cuda",
            dtype=tdtype,
        )
        Mg = torch.empty(
            tuple(M.shape),
            device="cuda",
            dtype=torch.bool,
        )

        block = max(64, self.transform_block_rows)
        with torch.inference_mode():
            for start in range(0, len(X), block):
                end = min(len(X), start + block)
                xb = np.asarray(X[start:end])
                mb = np.asarray(M[start:end], dtype=bool)
                xt = torch.from_numpy(np.ascontiguousarray(xb))
                mt = torch.from_numpy(np.ascontiguousarray(mb))
                Xg[start:end].copy_(
                    xt.to(device="cuda", dtype=tdtype)
                )
                Mg[start:end].copy_(
                    mt.to(device="cuda", dtype=torch.bool)
                )

        print(
            f"[GPU cache] {label}: cached {needed/2**30:.2f} GiB "
            f"shape={tuple(X.shape)} dtype={X.dtype}",
            flush=True,
        )
        return {
            "X": Xg,
            "mask": Mg,
            "bytes": needed,
            "label": label,
        }

    def _write_json(self, path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonify(obj), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def fit(
        self,
        X_tokens: Array,
        attention_mask: Array,
        y: Array,
        *,
        eval_set: Optional[Tuple[Array, Array, Array]] = None,
        sample_weight: Optional[Array] = None,
        work_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        X0 = X_tokens
        M0 = np.asarray(attention_mask)
        y = np.asarray(y, dtype=np.int64)
        if X0.ndim != 3:
            raise ValueError("X_tokens must have shape [N,T,D]")
        if X0.shape[2] != self.token_dim:
            raise ValueError("initial token_dim mismatch")
        if M0.shape != X0.shape[:2]:
            raise ValueError("attention_mask shape mismatch")
        if y.shape != (len(X0),):
            raise ValueError("y shape mismatch")
        if np.min(y) < 0 or np.max(y) >= self.output_dim:
            raise ValueError("labels outside output_dim")

        if work_dir is None:
            work_dir = Path("./deep_token_forest_work")
        else:
            work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if self.diagnostics_dir is not None:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.stages_ = []
        self.history_ = []
        self.base_logits_ = self._base_logits(y, sample_weight)
        if len(self.base_logits_) != self.output_dim:
            # Handle a rare train subset with a missing class.
            padded = np.full(self.output_dim, -20.0, dtype=np.float32)
            padded[: len(self.base_logits_)] = self.base_logits_
            self.base_logits_ = padded

        train_logits = np.repeat(
            self.base_logits_[None, :], len(X0), axis=0
        ).astype(np.float32)

        if eval_set is not None:
            Xv0, Mv0, yv = eval_set
            Mv0 = np.asarray(Mv0)
            yv = np.asarray(yv, dtype=np.int64)
            eval_logits = np.repeat(
                self.base_logits_[None, :], len(Xv0), axis=0
            ).astype(np.float32)
        else:
            Xv0 = Mv0 = yv = eval_logits = None

        Xcur = X0
        Mcur = M0
        Xvcur = Xv0
        Mvcur = Mv0

        best_eval = np.inf
        best_global_tree = -1
        stale = 0
        global_tree = 0
        stop_all = False

        for stage in range(self.n_stages):
            input_dim = int(Xcur.shape[2])
            doc_mean = masked_mean_tokens(
                Xcur, Mcur, block_rows=self.score_block_rows
            )
            stage_trees: List[ProjectionDistributionNewtonTree] = []

            stage_train_gpu_cache = None
            stage_eval_gpu_cache = None
            if self.tree_device == "cuda" and self.cache_stage_on_gpu:
                free0, _ = torch.cuda.mem_get_info()
                stage_budget = int(free0 * self.gpu_cache_fraction)
                stage_train_gpu_cache = self._build_stage_gpu_cache(
                    Xcur,
                    Mcur,
                    label=f"stage-{stage}-train",
                    budget_bytes=stage_budget,
                )
                used = (
                    0
                    if stage_train_gpu_cache is None
                    else int(stage_train_gpu_cache["bytes"])
                )
                remaining_budget = max(0, stage_budget - used)
                if eval_set is not None and remaining_budget > 0:
                    stage_eval_gpu_cache = self._build_stage_gpu_cache(
                        Xvcur,
                        Mvcur,
                        label=f"stage-{stage}-val",
                        budget_bytes=remaining_budget,
                    )

            for tree_idx in range(self.trees_per_stage):
                global_tree += 1
                before_train = _logloss(y, train_logits)
                before_eval = (
                    _logloss(yv, eval_logits) if eval_set is not None else None
                )

                g, h = self._gradient_hessian(y, train_logits)
                tree = ProjectionDistributionNewtonTree(
                    **self._tree_kwargs(input_dim, stage, tree_idx)
                )
                tree.fit(
                    Xcur,
                    Mcur,
                    g,
                    h,
                    doc_mean=doc_mean,
                    sample_weight=sample_weight,
                    stage_gpu_cache=stage_train_gpu_cache,
                )

                train_delta = tree.predict_values(
                    Xcur,
                    Mcur,
                    gpu_cache=stage_train_gpu_cache,
                )
                train_logits += np.float32(self.learning_rate) * train_delta
                after_train = _logloss(y, train_logits)

                after_eval = None
                if eval_set is not None:
                    eval_delta = tree.predict_values(
                        Xvcur,
                        Mvcur,
                        gpu_cache=stage_eval_gpu_cache,
                    )
                    eval_logits += np.float32(self.learning_rate) * eval_delta
                    after_eval = _logloss(yv, eval_logits)

                train_improvement = before_train - after_train
                eval_improvement = (
                    None
                    if after_eval is None
                    else before_eval - after_eval
                )
                transfer_ratio = (
                    None
                    if eval_improvement is None
                    or abs(train_improvement) < 1e-15
                    else eval_improvement / train_improvement
                )

                internal = tree.internal_nodes()
                record = {
                    "global_tree": global_tree,
                    "stage": stage,
                    "tree_in_stage": tree_idx,
                    "representation_dim": input_dim,
                    "train_logloss": after_train,
                    "eval_logloss": after_eval,
                    "train_improvement": train_improvement,
                    "eval_improvement": eval_improvement,
                    "transfer_ratio": transfer_ratio,
                    "tree_depth": tree.get_depth(),
                    "tree_leaves": tree.get_n_leaves(),
                    "tree_internal_nodes": tree.get_n_internal_nodes(),
                    "sum_effective_gain": float(
                        sum(float(n["effective_gain"]) for n in internal)
                    ),
                    "mean_effective_gain": float(
                        np.mean([n["effective_gain"] for n in internal])
                        if internal else 0.0
                    ),
                    "mean_search_gain": float(
                        np.mean([n["search_gain"] for n in internal])
                        if internal else 0.0
                    ),
                    "mean_honest_gain": float(
                        np.mean(
                            [
                                n["honest_gain"]
                                for n in internal
                                if np.isfinite(n["honest_gain"])
                            ]
                        )
                        if any(
                            np.isfinite(n["honest_gain"]) for n in internal
                        )
                        else float("nan")
                    ),
                }
                self.history_.append(record)
                stage_trees.append(tree)

                if verbose:
                    msg = (
                        f"[stage {stage+1}/{self.n_stages} "
                        f"tree {tree_idx+1}/{self.trees_per_stage}] "
                        f"dim={input_dim} train={after_train:.6f} "
                    )
                    if after_eval is not None:
                        msg += (
                            f"val={after_eval:.6f} "
                            f"transfer={transfer_ratio:.3f} "
                        )
                    msg += (
                        f"leaves={tree.get_n_leaves()} "
                        f"depth={tree.get_depth()}"
                    )
                    print(msg, flush=True)

                if self.diagnostics_dir is not None:
                    self._write_json(
                        self.diagnostics_dir
                        / f"tree_s{stage:02d}_t{tree_idx:03d}.json",
                        tree.to_structure_dict(include_w=True),
                    )
                    with (
                        self.diagnostics_dir / "training_history.jsonl"
                    ).open("a", encoding="utf-8") as f:
                        f.write(json.dumps(_jsonify(record)) + "\n")

                if after_eval is not None:
                    if after_eval < best_eval - self.early_stopping_min_delta:
                        best_eval = after_eval
                        best_global_tree = global_tree
                        stale = 0
                    else:
                        stale += 1
                    if (
                        self.early_stopping_rounds is not None
                        and self.early_stopping_rounds > 0
                        and stale >= self.early_stopping_rounds
                    ):
                        stop_all = True
                        break

            concepts: List[TokenConcept] = []
            if (
                self.feature_growth
                and not stop_all
                and stage < self.n_stages - 1
                and self.concepts_per_stage > 0
            ):
                concepts = self._select_concepts(
                    stage_trees, stage, len(Xcur)
                )
                fit_concept_standardization(
                    Xcur,
                    Mcur,
                    concepts,
                    random_state=self.random_state + 17 * stage,
                    block_rows=self.transform_block_rows,
                )

                if self.diagnostics_dir is not None:
                    train_stats = concept_statistics(
                        Xcur,
                        Mcur,
                        y,
                        concepts,
                        random_state=self.random_state + 31 * stage,
                        block_rows=self.transform_block_rows,
                    )
                    self._write_json(
                        self.diagnostics_dir
                        / f"concept_stats_train_stage_{stage:02d}.json",
                        train_stats,
                    )
                    if eval_set is not None:
                        val_stats = concept_statistics(
                            Xvcur,
                            Mvcur,
                            yv,
                            concepts,
                            random_state=self.random_state + 37 * stage,
                            block_rows=self.transform_block_rows,
                        )
                        self._write_json(
                            self.diagnostics_dir
                            / f"concept_stats_val_stage_{stage:02d}.json",
                            val_stats,
                        )
                    self._write_json(
                        self.diagnostics_dir
                        / f"concepts_stage_{stage:02d}.json",
                        [c.metadata(include_w=True) for c in concepts],
                    )

            self.stages_.append(
                {
                    "stage": stage,
                    "input_dim": input_dim,
                    "trees": stage_trees,
                    "concepts": concepts,
                }
            )

            # Drop stage CUDA tensors before constructing H^(s+1). All model
            # parameters are already stored as NumPy arrays inside the trees.
            stage_train_gpu_cache = None
            stage_eval_gpu_cache = None
            if self.tree_device == "cuda":
                torch.cuda.empty_cache()

            if stop_all:
                break

            if concepts:
                train_path = work_dir / f"train_stage_{stage+1:02d}.npy"
                Xcur = append_concept_features(
                    Xcur,
                    Mcur,
                    concepts,
                    output_path=train_path,
                    output_dtype=self.feature_dtype,
                    block_rows=self.transform_block_rows,
                    prefix_mixing=self.prefix_mixing,
                )
                if eval_set is not None:
                    val_path = work_dir / f"val_stage_{stage+1:02d}.npy"
                    Xvcur = append_concept_features(
                        Xvcur,
                        Mvcur,
                        concepts,
                        output_path=val_path,
                        output_dtype=self.feature_dtype,
                        block_rows=self.transform_block_rows,
                        prefix_mixing=self.prefix_mixing,
                    )

        self.final_token_dim_ = int(Xcur.shape[2])
        self.best_iteration_ = (
            best_global_tree if best_global_tree >= 0 else global_tree
        )
        self.best_eval_loss_ = (
            None if not np.isfinite(best_eval) else float(best_eval)
        )

        if self.diagnostics_dir is not None:
            self._write_json(
                self.diagnostics_dir / "model_structure.json",
                self.structure_dict(include_w=True),
            )
        return self

    def predict_logits(
        self,
        X_tokens: Array,
        attention_mask: Array,
        *,
        work_dir: Optional[Union[str, Path]] = None,
    ) -> Array:
        if self.base_logits_ is None:
            raise RuntimeError("model has not been fitted")
        Xcur = X_tokens
        M = np.asarray(attention_mask)
        logits = np.repeat(
            self.base_logits_[None, :], len(Xcur), axis=0
        ).astype(np.float32)

        if work_dir is not None:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        for stage_info in self.stages_:
            for tree in stage_info["trees"]:
                logits += np.float32(self.learning_rate) * tree.predict_values(
                    Xcur, M
                )
            concepts = stage_info["concepts"]
            if concepts:
                path = (
                    None
                    if work_dir is None
                    else work_dir
                    / f"predict_stage_{int(stage_info['stage'])+1:02d}.npy"
                )
                Xcur = append_concept_features(
                    Xcur,
                    M,
                    concepts,
                    output_path=path,
                    output_dtype=self.feature_dtype,
                    block_rows=self.transform_block_rows,
                    prefix_mixing=self.prefix_mixing,
                )
        return logits

    def predict_proba(self, X_tokens: Array, attention_mask: Array) -> Array:
        return _softmax(self.predict_logits(X_tokens, attention_mask))

    def predict(self, X_tokens: Array, attention_mask: Array) -> Array:
        return np.argmax(
            self.predict_logits(X_tokens, attention_mask), axis=1
        )

    def structure_dict(self, include_w: bool = True) -> dict:
        stages = []
        for s in self.stages_:
            stages.append(
                {
                    "stage": int(s["stage"]),
                    "input_dim": int(s["input_dim"]),
                    "trees": [
                        t.to_structure_dict(include_w=include_w)
                        for t in s["trees"]
                    ],
                    "concepts": [
                        c.metadata(include_w=include_w)
                        for c in s["concepts"]
                    ],
                }
            )
        return {
            "model": "DeepTokenForestClassifier",
            "initial_token_dim": self.token_dim,
            "final_token_dim": self.final_token_dim_,
            "output_dim": self.output_dim,
            "n_stages_requested": self.n_stages,
            "trees_per_stage": self.trees_per_stage,
            "feature_growth": self.feature_growth,
            "prefix_mixing": self.prefix_mixing,
            "tree_device": self.tree_device,
            "threshold_n_jobs": self.threshold_n_jobs,
            "cache_stage_on_gpu": self.cache_stage_on_gpu,
            "gpu_cache_fraction": self.gpu_cache_fraction,
            "base_logits": (
                None
                if self.base_logits_ is None
                else self.base_logits_.tolist()
            ),
            "stages": stages,
            "history": _jsonify(self.history_),
        }

    def export_concept_token_examples(
        self,
        X_tokens: Array,
        attention_mask: Array,
        input_ids: Array,
        tokenizer,
        *,
        max_docs: int = 3000,
        top_n: int = 20,
        random_state: int = 42,
    ) -> List[dict]:
        """
        Decode high-activation tokens for every selected concept.

        This is diagnostic only; it is never used during training.
        """
        rng = np.random.default_rng(random_state)
        n = len(X_tokens)
        rows = (
            np.arange(n, dtype=np.int64)
            if n <= max_docs
            else rng.choice(n, size=max_docs, replace=False).astype(np.int64)
        )
        Xcur = np.asarray(X_tokens[rows], dtype=np.float32)
        Mcur = np.asarray(attention_mask[rows], dtype=bool)
        IDs = np.asarray(input_ids[rows], dtype=np.int64)
        all_items = []

        for stage_info in self.stages_:
            concepts: List[TokenConcept] = stage_info["concepts"]
            for j, c in enumerate(concepts):
                B, T, D = Xcur.shape
                p = (
                    Xcur.reshape(B * T, D)
                    @ np.asarray(c.w, dtype=np.float32)
                ).reshape(B, T)
                p = np.where(Mcur, p - c.token_threshold, -np.inf)
                flat = p.reshape(-1)
                k = min(top_n, int(np.sum(np.isfinite(flat))))
                if k <= 0:
                    examples = []
                else:
                    idx = np.argpartition(flat, -k)[-k:]
                    idx = idx[np.argsort(flat[idx])[::-1]]
                    examples = []
                    for z in idx:
                        r = int(z // T)
                        t = int(z % T)
                        tok_id = int(IDs[r, t])
                        examples.append(
                            {
                                "sample_row": int(rows[r]),
                                "token_position": t,
                                "token_id": tok_id,
                                "token": tokenizer.decode(
                                    [tok_id], clean_up_tokenization_spaces=False
                                ),
                                "margin": float(p[r, t]),
                            }
                        )
                all_items.append(
                    {
                        "stage": int(stage_info["stage"]),
                        "concept_index": j,
                        "source_tree": c.tree,
                        "source_node_id": c.node_id,
                        "effective_gain": c.effective_gain,
                        "token_threshold": c.token_threshold,
                        "examples": examples,
                    }
                )

            if concepts:
                a = _concept_activation_block(Xcur, Mcur, concepts)
                if self.prefix_mixing:
                    valid = Mcur[:, :, None].astype(np.float32)
                    prefix_num = np.cumsum(a * valid, axis=1, dtype=np.float32)
                    prefix_den = np.maximum(
                        np.cumsum(valid, axis=1, dtype=np.float32), 1.0
                    )
                    prefix = (prefix_num / prefix_den) * valid
                    extra = np.concatenate([a, prefix], axis=2)
                else:
                    extra = a
                Xcur = np.concatenate([Xcur, extra], axis=2).astype(
                    np.float32, copy=False
                )
        return all_items
