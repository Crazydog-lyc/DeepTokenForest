"""
train_agnews_deep_token_forest.py

AG News training entry point for DeepTokenForestClassifier.

Protocol
--------
text
  -> frozen Pythia hidden state H [T,D]
  -> optional TRAIN-ONLY PCA D -> d
  -> projection-distribution Newton boosting
  -> stage-wise token concept feature growth
  -> class logits

There is no teacher model target, no teacher logits, no distillation and no
backpropagation through the tree model.

The script also records:
- per-tree train/validation loss and transfer ratio,
- complete tree structures including w, token threshold and document threshold,
- split search/honest gains,
- selected concept features and class-conditional statistics,
- decoded high-activation tokens for each selected concept,
- confusion matrices and predictions,
- PCA metadata and exact experiment configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from FrozenLMHiddenState import FrozenLMHiddenStateExtractor
from deep_token_forest import DeepTokenForestClassifier


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def _parse_float_tuple(text: str):
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("expected at least one comma-separated float")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Train hierarchical projection-distribution token forest on AG News."
    )
    p.add_argument("--dataset-dir", default="./data/ag_news")
    p.add_argument("--checkpoint", default="./checkpoints/pythia-70m")
    p.add_argument(
        "--output-dir",
        default="./runs/agnews_deep_token_forest",
    )
    p.add_argument(
        "--feature-cache-dir",
        default="./feature_cache/agnews_deep_token_forest",
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Stage feature memmaps. Defaults to OUTPUT_DIR/work.",
    )
    p.add_argument("--overwrite-output", action="store_true")

    # Frozen neural trunk.
    p.add_argument("--hidden-state-index", type=int, default=2)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--lm-batch-size", type=int, default=64)
    p.add_argument("--device", default=None)

    # Base token representation.
    p.add_argument(
        "--token-proj-dim",
        type=int,
        default=128,
        help="PCA output dim. 0 keeps full hidden width.",
    )
    p.add_argument("--pca-max-tokens", type=int, default=50000)
    p.add_argument(
        "--feature-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    p.add_argument("--rebuild-features", action="store_true")

    # Dataset protocol.
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--random-state", type=int, default=42)

    # Deep token forest.
    p.add_argument("--n-stages", type=int, default=4)
    p.add_argument("--trees-per-stage", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=0.10)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-leaves", type=int, default=16)
    p.add_argument("--min-samples-leaf", type=int, default=100)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--min-gain", type=float, default=1e-8)

    # Projection-distribution node.
    p.add_argument(
        "--distribution-activation",
        choices=["hard", "soft"],
        default="hard",
        help="'hard' is empirical survival/CDF; 'soft' is a smooth survival statistic.",
    )
    p.add_argument("--distribution-temperature", type=float, default=1.0)
    p.add_argument(
        "--token-threshold-quantiles",
        default="0.10,0.25,0.40,0.50,0.60,0.75,0.90",
        help="Candidate b quantiles along h@w.",
    )
    p.add_argument("--max-doc-thresholds", type=int, default=64)
    p.add_argument("--token-threshold-sample-rows", type=int, default=2048)

    # Direction search.
    p.add_argument("--n-random-directions", type=int, default=8)
    p.add_argument("--n-token-prototype-directions", type=int, default=8)
    p.add_argument("--n-local-perturbations", type=int, default=4)
    p.add_argument("--local-sigmas", default="0.10,0.25")
    p.add_argument("--beam-width", type=int, default=4)

    # Generalization-aware in-training split selection.
    p.add_argument(
        "--honest-fraction",
        type=float,
        default=0.15,
        help="0 disables honest candidate verification.",
    )
    p.add_argument("--honest-top-candidates", type=int, default=8)
    p.add_argument(
        "--min-honest-gain-per-sample",
        type=float,
        default=0.0,
    )

    # Representation growth.
    p.add_argument("--concepts-per-stage", type=int, default=16)
    p.add_argument("--concept-temperature", type=float, default=0.5)
    p.add_argument(
        "--disable-feature-growth",
        action="store_true",
        help="Keep H fixed across stages; useful as an equal-tree-count control.",
    )
    p.add_argument(
        "--prefix-mixing",
        action="store_true",
        help=(
            "Append causal prefix means of learned token concepts in addition "
            "to token-local concept channels. This is a no-BP sequence mixer."
        ),
    )

    # Runtime / monitoring.
    p.add_argument("--score-block-rows", type=int, default=1024)
    p.add_argument(
        "--direction-chunk-size",
        type=int,
        default=8,
        help=(
            "Candidate directions evaluated together. On CUDA, larger values "
            "increase GPU parallelism but also GPU/host memory usage."
        ),
    )
    p.add_argument(
        "--tree-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Backend for tree projection/distribution scoring. 'auto' uses CUDA "
            "when available; Newton gain/threshold search remains on CPU."
        ),
    )
    p.add_argument(
        "--threshold-n-jobs",
        type=int,
        default=0,
        help=(
            "CPU threads for independent exact document-threshold searches. "
            "0 selects up to 8 workers automatically."
        ),
    )
    p.add_argument(
        "--no-stage-gpu-cache",
        action="store_true",
        help=(
            "Disable exact stage-level CUDA feature caching. By default, training "
            "and validation H^(s) are cached when GPU memory permits."
        ),
    )
    p.add_argument(
        "--gpu-cache-fraction",
        type=float,
        default=0.55,
        help="Maximum fraction of currently free CUDA memory used for stage caches.",
    )
    p.add_argument("--transform-block-rows", type=int, default=512)
    p.add_argument("--early-stopping-rounds", type=int, default=0)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--concept-example-docs",
        type=int,
        default=3000,
    )
    p.add_argument("--concept-top-tokens", type=int, default=20)
    return p.parse_args()


def stratified_train_val_split(labels, validation_fraction, seed):
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * validation_fraction)))
        n_val = min(n_val, len(idx) - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def stratified_subsample(indices, labels, max_samples, seed):
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[indices])
    parts = []
    targets = {}
    total = 0
    for c in classes:
        pool = indices[labels[indices] == c]
        k = max(1, int(round(max_samples * len(pool) / len(indices))))
        k = min(k, len(pool))
        targets[int(c)] = k
        total += k

    while total > max_samples:
        changed = False
        for c in classes:
            c = int(c)
            if total <= max_samples:
                break
            if targets[c] > 1:
                targets[c] -= 1
                total -= 1
                changed = True
        if not changed:
            break

    while total < max_samples:
        changed = False
        for c in classes:
            c = int(c)
            if total >= max_samples:
                break
            pool_size = int(np.sum(labels[indices] == c))
            if targets[c] < pool_size:
                targets[c] += 1
                total += 1
                changed = True
        if not changed:
            break

    for c in classes:
        c = int(c)
        pool = indices[labels[indices] == c]
        parts.append(rng.choice(pool, size=targets[c], replace=False))
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out[:max_samples].astype(np.int64)


def _digest_indices(indices):
    a = np.asarray(indices, dtype=np.int64)
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "model"


def _checkpoint_fingerprint(checkpoint):
    h = hashlib.sha1()
    h.update(str(checkpoint).encode("utf-8"))
    p = Path(checkpoint)
    if p.exists():
        try:
            h.update(str(p.resolve()).encode("utf-8"))
        except Exception:
            pass
        for name in (
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ):
            f = p / name
            if f.exists():
                h.update(name.encode("utf-8"))
                h.update(f.read_bytes())
        for pattern in ("*.safetensors", "*.bin"):
            for f in sorted(p.glob(pattern)):
                st = f.stat()
                h.update(f.name.encode("utf-8"))
                h.update(str(st.st_size).encode("ascii"))
                h.update(str(st.st_mtime_ns).encode("ascii"))
    return h.hexdigest()[:12]


@torch.inference_mode()
def _hidden_batch(extractor, batch_texts, max_length, pad_to_max):
    tokenizer = extractor.tokenizer
    model = extractor.model
    device = extractor.device
    layer = extractor.hidden_state_index

    encoded = tokenizer(
        list(batch_texts),
        padding="max_length" if pad_to_max else True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model returned hidden_states=None")
    if layer >= len(hidden_states):
        raise RuntimeError(
            f"requested hidden_states[{layer}], model returned "
            f"{len(hidden_states)} states"
        )
    return hidden_states[layer], attention_mask, input_ids


def fit_token_pca(
    extractor,
    texts,
    target_dim,
    max_tokens,
    max_length,
    batch_size,
    seed,
):
    hidden_dim = int(extractor.hidden_size)
    if target_dim <= 0 or target_dim >= hidden_dim:
        return {
            "mean": np.zeros(hidden_dim, dtype=np.float32),
            "components": None,
            "output_dim": hidden_dim,
            "explained_variance_ratio": None,
            "explained_variance_curve": None,
        }

    target_dim = min(int(target_dim), hidden_dim)
    max_tokens = max(int(max_tokens), target_dim + 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    chunks = []
    collected = 0

    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        batch = [texts[int(i)] for i in idx]
        h, mask, _ = _hidden_batch(
            extractor,
            batch,
            max_length=max_length,
            pad_to_max=False,
        )
        valid = h[mask.bool()].float().cpu().numpy()
        remaining = max_tokens - collected
        if len(valid) > remaining:
            take = rng.choice(len(valid), size=remaining, replace=False)
            valid = valid[take]
        if len(valid):
            chunks.append(valid.astype(np.float32, copy=False))
            collected += len(valid)
        print(
            f"\rPCA calibration {collected:,}/{max_tokens:,}",
            end="",
            flush=True,
        )
        if collected >= max_tokens:
            break
    print()

    if not chunks:
        raise RuntimeError("PCA calibration produced no valid tokens")
    X = np.concatenate(chunks, axis=0)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    Xc = X.astype(np.float64) - mean.astype(np.float64)
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :target_dim].astype(np.float32)

    total = float(eigvals.sum())
    curve = (
        np.cumsum(eigvals) / total if total > 0 else np.zeros_like(eigvals)
    )
    explained = float(curve[target_dim - 1]) if len(curve) else 0.0
    return {
        "mean": mean,
        "components": components,
        "output_dim": target_dim,
        "explained_variance_ratio": explained,
        "explained_variance_curve": curve.astype(np.float32),
    }


def save_pca(path: Path, pca: dict):
    np.savez_compressed(
        path,
        mean=pca["mean"],
        components=(
            pca["components"]
            if pca["components"] is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        has_components=np.array(
            1 if pca["components"] is not None else 0,
            dtype=np.int64,
        ),
        output_dim=np.array(pca["output_dim"], dtype=np.int64),
        explained_variance_ratio=np.array(
            -1.0
            if pca["explained_variance_ratio"] is None
            else pca["explained_variance_ratio"],
            dtype=np.float64,
        ),
        explained_variance_curve=(
            pca["explained_variance_curve"]
            if pca["explained_variance_curve"] is not None
            else np.empty(0, dtype=np.float32)
        ),
    )


def load_pca(path: Path):
    z = np.load(path)
    has = bool(int(z["has_components"]))
    ratio = float(z["explained_variance_ratio"])
    return {
        "mean": z["mean"].astype(np.float32),
        "components": z["components"].astype(np.float32) if has else None,
        "output_dim": int(z["output_dim"]),
        "explained_variance_ratio": None if ratio < 0 else ratio,
        "explained_variance_curve": (
            z["explained_variance_curve"].astype(np.float32)
            if "explained_variance_curve" in z
            and len(z["explained_variance_curve"])
            else None
        ),
    }


@torch.inference_mode()
def extract_projected_token_features(
    extractor,
    texts,
    labels,
    cache_dir,
    split_tag,
    pca,
    max_length,
    batch_size,
    feature_dtype,
    rebuild,
):
    """
    Cache:
      X         [N,T,D] float16/float32
      mask      [N,T]   uint8
      input_ids [N,T]   int32
      y         [N]     int64
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path = cache_dir / f"{split_tag}_X.npy"
    m_path = cache_dir / f"{split_tag}_mask.npy"
    id_path = cache_dir / f"{split_tag}_input_ids.npy"
    y_path = cache_dir / f"{split_tag}_y.npy"

    if (
        x_path.exists()
        and m_path.exists()
        and id_path.exists()
        and y_path.exists()
        and not rebuild
    ):
        print("Loading cache:", split_tag)
        return (
            np.load(x_path, mmap_mode="r"),
            np.load(m_path, mmap_mode="r"),
            np.load(id_path, mmap_mode="r"),
            np.load(y_path),
        )

    n = len(texts)
    d = int(pca["output_dim"])
    np_dtype = np.float16 if feature_dtype == "float16" else np.float32
    Xmm = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np_dtype, shape=(n, max_length, d)
    )
    Mmm = np.lib.format.open_memmap(
        m_path, mode="w+", dtype=np.uint8, shape=(n, max_length)
    )
    Imm = np.lib.format.open_memmap(
        id_path, mode="w+", dtype=np.int32, shape=(n, max_length)
    )

    comp = pca["components"]
    mean = pca["mean"]
    if comp is not None:
        mean_t = torch.as_tensor(mean, device=extractor.device, dtype=torch.float32)
        comp_t = torch.as_tensor(comp, device=extractor.device, dtype=torch.float32)
    else:
        mean_t = comp_t = None

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        h, mask, input_ids = _hidden_batch(
            extractor,
            texts[start:end],
            max_length=max_length,
            pad_to_max=True,
        )
        z = h.float()
        if comp_t is not None:
            z = (z - mean_t[None, None, :]) @ comp_t
        z = z * mask[:, :, None].to(z.dtype)

        Xmm[start:end] = z.cpu().numpy().astype(np_dtype, copy=False)
        Mmm[start:end] = mask.cpu().numpy().astype(np.uint8, copy=False)
        Imm[start:end] = input_ids.cpu().numpy().astype(np.int32, copy=False)
        print(
            f"\r{split_tag}: {end:,}/{n:,}",
            end="",
            flush=True,
        )
    print()

    Xmm.flush()
    Mmm.flush()
    Imm.flush()
    np.save(y_path, np.asarray(labels, dtype=np.int64))
    del Xmm, Mmm, Imm

    return (
        np.load(x_path, mmap_mode="r"),
        np.load(m_path, mmap_mode="r"),
        np.load(id_path, mmap_mode="r"),
        np.load(y_path),
    )


def softmax_np(logits):
    x = np.asarray(logits, dtype=np.float64)
    x -= np.max(x, axis=1, keepdims=True)
    np.exp(x, out=x)
    x /= np.maximum(np.sum(x, axis=1, keepdims=True), 1e-300)
    return x


def metrics(y_true, logits):
    y = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits)
    p = softmax_np(logits)
    pred = np.argmax(logits, axis=1)
    accuracy = float(np.mean(pred == y))
    loss = float(
        -np.mean(
            np.log(
                np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
            )
        )
    )
    C = logits.shape[1]
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    f1s = []
    per_class = []
    for c in range(C):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1s.append(f1)
        per_class.append(
            {
                "class_id": c,
                "class_name": LABEL_NAMES[c] if c < len(LABEL_NAMES) else str(c),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(cm[c].sum()),
            }
        )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "logloss": loss,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def summarize_transfer(history):
    rows = [r for r in history if r.get("eval_improvement") is not None]
    if not rows:
        return {}
    by_stage = {}
    for r in rows:
        s = int(r["stage"])
        by_stage.setdefault(s, []).append(r)
    out = {}
    for s, rs in by_stage.items():
        train_gain = float(sum(r["train_improvement"] for r in rs))
        val_gain = float(sum(r["eval_improvement"] for r in rs))
        out[str(s)] = {
            "trees": len(rs),
            "train_loss_improvement": train_gain,
            "val_loss_improvement": val_gain,
            "val_over_train_transfer": (
                val_gain / train_gain if abs(train_gain) > 1e-15 else None
            ),
            "mean_tree_transfer_ratio": float(
                np.nanmean(
                    [
                        r["transfer_ratio"]
                        if r["transfer_ratio"] is not None
                        else np.nan
                        for r in rs
                    ]
                )
            ),
        }
    return out


def main():
    args = parse_args()
    token_quantiles = _parse_float_tuple(args.token_threshold_quantiles)
    local_sigmas = _parse_float_tuple(args.local_sigmas)

    out_dir = Path(args.output_dir)
    if out_dir.exists() and args.overwrite_output:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    history_path = diagnostics_dir / "training_history.jsonl"
    if history_path.exists():
        history_path.unlink()

    work_dir = (
        Path(args.work_dir)
        if args.work_dir is not None
        else out_dir / "work"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("1) Dataset protocol")
    print("=" * 90)
    ds = load_from_disk(args.dataset_dir)
    y_train_all = np.asarray(ds["train"]["label"], dtype=np.int64)
    y_test_all = np.asarray(ds["test"]["label"], dtype=np.int64)
    train_idx, val_idx = stratified_train_val_split(
        y_train_all,
        args.validation_fraction,
        args.random_state,
    )
    train_idx = stratified_subsample(
        train_idx,
        y_train_all,
        args.max_train_samples,
        args.random_state + 1,
    )
    val_idx = stratified_subsample(
        val_idx,
        y_train_all,
        args.max_val_samples,
        args.random_state + 2,
    )
    test_idx = stratified_subsample(
        np.arange(len(y_test_all), dtype=np.int64),
        y_test_all,
        args.max_test_samples,
        args.random_state + 3,
    )

    train_text_all = ds["train"]["text"]
    test_text_all = ds["test"]["text"]
    train_texts = [train_text_all[int(i)] for i in train_idx]
    val_texts = [train_text_all[int(i)] for i in val_idx]
    test_texts = [test_text_all[int(i)] for i in test_idx]
    y_train = y_train_all[train_idx]
    y_val = y_train_all[val_idx]
    y_test = y_test_all[test_idx]
    print("train", len(y_train), "val", len(y_val), "test", len(y_test))

    print("\n" + "=" * 90)
    print("2) Frozen language-model trunk")
    print("=" * 90)
    extractor = FrozenLMHiddenStateExtractor(
        model_name=args.checkpoint,
        hidden_state_index=args.hidden_state_index,
        max_length=args.max_length,
        batch_size=args.lm_batch_size,
        device=args.device,
        context_window=1,
        dtype=np.float32,
    )
    print(json.dumps(extractor.describe(), indent=2))

    checkpoint_fp = _checkpoint_fingerprint(args.checkpoint)
    cache_meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": checkpoint_fp,
        "hidden_state_index": args.hidden_state_index,
        "max_length": args.max_length,
        "token_proj_dim": args.token_proj_dim,
        "pca_max_tokens": args.pca_max_tokens,
        "feature_dtype": args.feature_dtype,
        "train_index_digest": _digest_indices(train_idx),
        "random_state": args.random_state,
    }
    cache_key = hashlib.sha1(
        json.dumps(cache_meta, sort_keys=True).encode("utf-8")
    ).hexdigest()[:14]
    cache_dir = Path(args.feature_cache_dir) / (
        f"{_safe_name(Path(args.checkpoint).name)}_{cache_key}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "cache_meta.json").write_text(
        json.dumps(cache_meta, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("3) Train-only PCA")
    print("=" * 90)
    pca_path = cache_dir / "pca_projection.npz"
    if pca_path.exists() and not args.rebuild_features:
        pca = load_pca(pca_path)
        print("Loaded", pca_path)
    else:
        pca = fit_token_pca(
            extractor,
            train_texts,
            target_dim=args.token_proj_dim,
            max_tokens=args.pca_max_tokens,
            max_length=args.max_length,
            batch_size=args.lm_batch_size,
            seed=args.random_state + 10,
        )
        save_pca(pca_path, pca)
        print("Saved", pca_path)
    print("base token dim:", pca["output_dim"])
    print("PCA explained variance:", pca["explained_variance_ratio"])

    print("\n" + "=" * 90)
    print("4) Token feature cache")
    print("=" * 90)
    t0 = time.perf_counter()
    train_tag = f"train_{_digest_indices(train_idx)}"
    val_tag = f"val_{_digest_indices(val_idx)}"
    test_tag = f"test_{_digest_indices(test_idx)}"

    Xtr, Mtr, Itr, ytr_cache = extract_projected_token_features(
        extractor,
        train_texts,
        y_train,
        cache_dir,
        train_tag,
        pca,
        args.max_length,
        args.lm_batch_size,
        args.feature_dtype,
        args.rebuild_features,
    )
    Xv, Mv, Iv, yv_cache = extract_projected_token_features(
        extractor,
        val_texts,
        y_val,
        cache_dir,
        val_tag,
        pca,
        args.max_length,
        args.lm_batch_size,
        args.feature_dtype,
        args.rebuild_features,
    )
    Xte, Mte, Ite, yte_cache = extract_projected_token_features(
        extractor,
        test_texts,
        y_test,
        cache_dir,
        test_tag,
        pca,
        args.max_length,
        args.lm_batch_size,
        args.feature_dtype,
        args.rebuild_features,
    )
    feature_seconds = time.perf_counter() - t0

    assert np.array_equal(ytr_cache, y_train)
    assert np.array_equal(yv_cache, y_val)
    assert np.array_equal(yte_cache, y_test)
    print("X_train", Xtr.shape, Xtr.dtype)
    print("X_val  ", Xv.shape, Xv.dtype)
    print("X_test ", Xte.shape, Xte.dtype)

    # Feature extraction is complete. Keep only metadata/tokenizer needed for
    # diagnostics and move the frozen LM off CUDA so the tree stage can use the
    # freed VRAM. This cannot change tree inputs because Xtr/Xv/Xte are already
    # materialized and immutable.
    concept_tokenizer = extractor.tokenizer
    original_hidden_dim = int(extractor.hidden_size)
    if torch.cuda.is_available():
        try:
            extractor.model.to("cpu")
        except Exception as exc:
            print("Warning: could not move frozen LM to CPU:", exc)
        torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("5) Train Deep Token Forest")
    print("=" * 90)
    model = DeepTokenForestClassifier(
        token_dim=Xtr.shape[2],
        output_dim=len(LABEL_NAMES),
        n_stages=args.n_stages,
        trees_per_stage=args.trees_per_stage,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_leaves=args.max_leaves,
        min_samples_leaf=args.min_samples_leaf,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        min_gain=args.min_gain,
        distribution_activation=args.distribution_activation,
        distribution_temperature=args.distribution_temperature,
        token_threshold_quantiles=token_quantiles,
        max_doc_thresholds=args.max_doc_thresholds,
        token_threshold_sample_rows=args.token_threshold_sample_rows,
        n_random_directions=args.n_random_directions,
        n_token_prototype_directions=args.n_token_prototype_directions,
        n_local_perturbations=args.n_local_perturbations,
        local_sigmas=local_sigmas,
        beam_width=args.beam_width,
        honest_fraction=args.honest_fraction,
        honest_top_candidates=args.honest_top_candidates,
        min_honest_gain_per_sample=args.min_honest_gain_per_sample,
        concepts_per_stage=args.concepts_per_stage,
        concept_temperature=args.concept_temperature,
        feature_growth=not args.disable_feature_growth,
        prefix_mixing=args.prefix_mixing,
        feature_dtype=args.feature_dtype,
        score_block_rows=args.score_block_rows,
        direction_chunk_size=args.direction_chunk_size,
        tree_device=args.tree_device,
        threshold_n_jobs=args.threshold_n_jobs,
        cache_stage_on_gpu=not args.no_stage_gpu_cache,
        gpu_cache_fraction=args.gpu_cache_fraction,
        transform_block_rows=args.transform_block_rows,
        early_stopping_rounds=(
            None
            if args.early_stopping_rounds <= 0
            else args.early_stopping_rounds
        ),
        early_stopping_min_delta=args.early_stopping_min_delta,
        diagnostics_dir=diagnostics_dir,
        random_state=args.random_state,
    )

    print(
        "Tree projection backend:",
        model.tree_device,
        "| direction_chunk_size:",
        model.direction_chunk_size,
        "| score_block_rows:",
        model.score_block_rows,
        "| threshold_n_jobs:",
        model.threshold_n_jobs,
        "| stage_gpu_cache:",
        model.cache_stage_on_gpu,
    )

    train_t0 = time.perf_counter()
    model.fit(
        Xtr,
        Mtr,
        y_train,
        eval_set=(Xv, Mv, y_val),
        work_dir=work_dir,
        verbose=True,
    )
    training_seconds = time.perf_counter() - train_t0

    print("\n" + "=" * 90)
    print("6) Final metrics")
    print("=" * 90)
    val_t0 = time.perf_counter()
    val_logits = model.predict_logits(
        Xv, Mv, work_dir=work_dir / "predict_val"
    )
    val_seconds = time.perf_counter() - val_t0

    test_t0 = time.perf_counter()
    test_logits = model.predict_logits(
        Xte, Mte, work_dir=work_dir / "predict_test"
    )
    test_seconds = time.perf_counter() - test_t0

    val_metrics = metrics(y_val, val_logits)
    test_metrics = metrics(y_test, test_logits)
    print(
        f"VAL  accuracy={val_metrics['accuracy']:.4f} "
        f"macro-F1={val_metrics['macro_f1']:.4f} "
        f"logloss={val_metrics['logloss']:.6f}"
    )
    print(
        f"TEST accuracy={test_metrics['accuracy']:.4f} "
        f"macro-F1={test_metrics['macro_f1']:.4f} "
        f"logloss={test_metrics['logloss']:.6f}"
    )

    print("\n" + "=" * 90)
    print("7) Concept interpretation")
    print("=" * 90)
    concept_examples = model.export_concept_token_examples(
        Xtr,
        Mtr,
        Itr,
        concept_tokenizer,
        max_docs=args.concept_example_docs,
        top_n=args.concept_top_tokens,
        random_state=args.random_state + 999,
    )
    (diagnostics_dir / "concept_top_tokens.json").write_text(
        json.dumps(concept_examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved decoded concept examples.")

    tree_stats = []
    for stage in model.stages_:
        for i, tree in enumerate(stage["trees"]):
            nodes = tree.internal_nodes()
            tree_stats.append(
                {
                    "stage": int(stage["stage"]),
                    "tree": i,
                    "input_dim": int(stage["input_dim"]),
                    "depth": tree.get_depth(),
                    "leaves": tree.get_n_leaves(),
                    "internal_nodes": tree.get_n_internal_nodes(),
                    "sum_effective_gain": float(
                        sum(n["effective_gain"] for n in nodes)
                    ),
                }
            )

    report = {
        "benchmark": "AG News",
        "model": "Deep Token Forest",
        "principles": {
            "distillation": False,
            "backpropagation_in_tree_model": False,
            "node_statistic": (
                "projected token empirical survival function "
                "c_i(w,b)=mean_t 1[h_it^T w >= b]"
            ),
            "representation_growth": (
                "H_{s+1}=concat(H_s, standardized tanh((H_s w-b)/tau))"
            ),
            "supervision": "ground-truth labels only",
        },
        "representation": {
            "checkpoint": args.checkpoint,
            "checkpoint_fingerprint": checkpoint_fp,
            "hidden_state_index": args.hidden_state_index,
            "max_length": args.max_length,
            "original_hidden_dim": original_hidden_dim,
            "base_token_dim": int(Xtr.shape[2]),
            "final_token_dim": int(model.final_token_dim_),
            "pca_explained_variance_ratio": pca["explained_variance_ratio"],
            "feature_growth": not args.disable_feature_growth,
            "prefix_mixing": args.prefix_mixing,
            "tree_device": model.tree_device,
        },
        "protocol": {
            "train_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
            "test_rows": int(len(y_test)),
            "validation_fraction": args.validation_fraction,
            "test_used_for_hyperparameter_selection": False,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "best_iteration": model.best_iteration_,
        "best_eval_loss": model.best_eval_loss_,
        "transfer_summary": summarize_transfer(model.history_),
        "tree_statistics": tree_stats,
        "history": model.history_,
        "efficiency": {
            "feature_extraction_seconds": feature_seconds,
            "training_seconds": training_seconds,
            "validation_inference_seconds": val_seconds,
            "test_inference_seconds": test_seconds,
        },
        "args": vars(args),
    }

    (out_dir / "metrics_deep_token_forest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with open(out_dir / "deep_token_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    np.savez_compressed(
        out_dir / "predictions_deep_token_forest.npz",
        y_val=y_val,
        val_logits=val_logits,
        y_test=y_test,
        test_logits=test_logits,
    )
    (out_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    print("\nSaved all outputs to", out_dir)
    print("Detailed diagnostics:", diagnostics_dir)


if __name__ == "__main__":
    main()
