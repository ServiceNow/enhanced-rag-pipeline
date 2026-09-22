"""
Significance testing across seeds for BRAG benchmark results.

Walks benchmark_results/{dataset}/{strategy}_{timestamp}[_suffix]_seed{N}/ runs,
computes per-query HIT@K and MRR from retrieval_details.jsonl, aggregates
across seeds, and runs paired t-tests against the S1 baseline plus Cohen's d.

Usage:
    uv run python -m evaluation.significance \
        --results-dir benchmark_results \
        --datasets ambignq2000 hotpotqa5000 \
        --strategies S1 S2 S3 S4 HyDE Query2Doc \
        --k 1 5 10 \
        --baseline S1 \
        --out-tex tables/significance.tex \
        --out-json tables/significance.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class RunMetrics:
    """Per-run aggregate: {query_id -> {metric_name -> score}}."""
    dataset: str
    strategy: str
    seed: Optional[int]
    embedding_model: Optional[str]
    run_dir: Path
    per_query: Dict[str, Dict[str, float]]


def compute_hit_at_k(retrieved_docs: List[dict], k: int) -> float:
    top_k = retrieved_docs[:k]
    return 1.0 if any(d.get("is_relevant") for d in top_k) else 0.0


def compute_mrr(retrieved_docs: List[dict], k: Optional[int] = None) -> float:
    """Mean reciprocal rank, optionally truncated at k."""
    docs = retrieved_docs if k is None else retrieved_docs[:k]
    for d in docs:
        if d.get("is_relevant"):
            rank = d.get("rank")
            if rank is None or rank <= 0:
                continue
            return 1.0 / float(rank)
    return 0.0


def compute_recall_at_k(retrieved_docs: List[dict], relevant_docs: List[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    hits = sum(1 for d in top_k if d.get("is_relevant"))
    return hits / float(len(relevant_docs))


def load_run(run_dir: Path) -> Optional[RunMetrics]:
    """Load a single benchmark run's per-query metrics."""
    metrics_path = run_dir / "metrics.json"
    details_path = run_dir / "retrieval_details.jsonl"
    if not metrics_path.exists() or not details_path.exists():
        return None

    with open(metrics_path) as f:
        meta = json.load(f)

    dataset = meta.get("dataset_name", "")
    strategy = meta.get("strategy") or meta.get("config", {}).get("strategy", "")
    seed = meta.get("seed")
    embedding_model = meta.get("embedding_model") or meta.get("config", {}).get("embedding_model")

    if not strategy:
        # Fall back to parsing from directory name: "S3_20260213_191724_bge_seed42"
        parts = run_dir.name.split("_")
        if parts:
            strategy = parts[0]

    if seed is None:
        # Try to parse trailing "_seedN" from dir name
        for tok in reversed(run_dir.name.split("_")):
            if tok.startswith("seed") and tok[4:].lstrip("-").isdigit():
                seed = int(tok[4:])
                break

    per_query: Dict[str, Dict[str, float]] = {}
    with open(details_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["query_id"]
            retrieved = row.get("retrieved_docs", [])
            relevant = row.get("relevant_docs", [])
            scores: Dict[str, float] = {}
            for k in (1, 3, 5, 10, 20):
                scores[f"hit@{k}"] = compute_hit_at_k(retrieved, k)
                scores[f"recall@{k}"] = compute_recall_at_k(retrieved, relevant, k)
            scores["mrr@10"] = compute_mrr(retrieved, k=10)
            per_query[qid] = scores

    return RunMetrics(
        dataset=dataset,
        strategy=strategy,
        seed=seed,
        embedding_model=embedding_model,
        run_dir=run_dir,
        per_query=per_query,
    )


def find_runs(results_dir: Path) -> List[RunMetrics]:
    """Walk benchmark_results/ and load every recognizable run."""
    runs: List[RunMetrics] = []
    if not results_dir.exists():
        return runs
    # Standard layout: results_dir/{dataset_name}/{run_dir}/metrics.json
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        # skip ad-hoc top-level layouts; the runner saves to <dataset>/<run>
        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = load_run(run_dir)
            if run is not None:
                runs.append(run)
    return runs


def aggregate_per_query_mean(runs: List[RunMetrics], metric: str) -> Dict[str, float]:
    """Average per-query score across (presumably 3) seeds of the same strategy."""
    if not runs:
        return {}
    all_qids = set(runs[0].per_query.keys())
    for r in runs[1:]:
        all_qids &= set(r.per_query.keys())
    out: Dict[str, float] = {}
    for qid in all_qids:
        vals = [r.per_query[qid].get(metric, 0.0) for r in runs]
        out[qid] = float(np.mean(vals))
    return out


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (mean of differences / pooled std)."""
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(diff.mean() / sd)


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


@dataclass
class CellResult:
    dataset: str
    strategy: str
    metric: str
    n_seeds: int
    seed_means: List[float]
    mean: float
    std: float
    paired_p: Optional[float]
    cohen_d: Optional[float]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--datasets", nargs="+", default=["ambignq2000", "hotpotqa5000"])
    parser.add_argument("--strategies", nargs="+",
                        default=["S1", "S2", "S3", "S4", "HyDE", "Query2Doc"])
    parser.add_argument("--baseline", default="S1",
                        help="Strategy to compare against in paired tests")
    parser.add_argument("--metrics", nargs="+",
                        default=["hit@1", "hit@5", "hit@10", "mrr@10", "recall@10"])
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5",
                        help="Only consider runs with this embedding model")
    parser.add_argument("--out-tex", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_runs = find_runs(results_dir)
    # Filter by embedding model if specified
    if args.embedding_model:
        all_runs = [
            r for r in all_runs
            if (r.embedding_model is None) or (r.embedding_model == args.embedding_model)
        ]

    cells: List[CellResult] = []
    for dataset in args.datasets:
        baseline_runs = [r for r in all_runs
                         if r.dataset == dataset and r.strategy == args.baseline]
        baseline_per_query: Dict[str, Dict[str, float]] = {}
        for metric in args.metrics:
            baseline_per_query[metric] = aggregate_per_query_mean(baseline_runs, metric)

        for strategy in args.strategies:
            strat_runs = [r for r in all_runs
                          if r.dataset == dataset and r.strategy == strategy]
            if not strat_runs:
                print(f"[warn] no runs found: dataset={dataset} strategy={strategy}")
                continue

            for metric in args.metrics:
                # Per-seed dataset-level mean
                seed_means: List[float] = []
                for r in strat_runs:
                    vals = [r.per_query[qid].get(metric, 0.0) for qid in r.per_query]
                    if vals:
                        seed_means.append(float(np.mean(vals)))
                if not seed_means:
                    continue
                mean = float(np.mean(seed_means))
                std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0

                # Paired test vs baseline, on per-query means averaged across seeds.
                # Require >= 2 seeds: a single-seed strategy gives no estimate of
                # seed variance, so a per-query test would overstate confidence.
                p_val: Optional[float] = None
                d: Optional[float] = None
                if (strategy != args.baseline and len(seed_means) >= 2
                        and baseline_per_query[metric]):
                    strat_pq = aggregate_per_query_mean(strat_runs, metric)
                    shared = sorted(set(strat_pq.keys()) & set(baseline_per_query[metric].keys()))
                    if shared:
                        a = np.array([strat_pq[q] for q in shared])
                        b = np.array([baseline_per_query[metric][q] for q in shared])
                        # All-zero-difference vector breaks ttest_rel; guard it.
                        if not np.allclose(a, b):
                            t_res = stats.ttest_rel(a, b)
                            p_val = float(t_res.pvalue)
                            d = cohen_d(a, b)
                        else:
                            p_val = 1.0
                            d = 0.0

                cells.append(CellResult(
                    dataset=dataset, strategy=strategy, metric=metric,
                    n_seeds=len(seed_means), seed_means=seed_means,
                    mean=mean, std=std, paired_p=p_val, cohen_d=d,
                ))

    # JSON output (full detail)
    if args.out_json:
        out = [{
            "dataset": c.dataset, "strategy": c.strategy, "metric": c.metric,
            "n_seeds": c.n_seeds, "seed_means": c.seed_means,
            "mean": c.mean, "std": c.std,
            "paired_p": c.paired_p, "cohen_d": c.cohen_d,
            "sig": sig_marker(c.paired_p) if c.paired_p is not None else "",
        } for c in cells]
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.out_json}")

    # LaTeX table per dataset
    tex_lines: List[str] = []
    for dataset in args.datasets:
        ds_cells = [c for c in cells if c.dataset == dataset]
        if not ds_cells:
            continue
        tex_lines.append(f"% --- {dataset} ---")
        tex_lines.append(r"\begin{tabular}{l" + "c" * len(args.metrics) + "}")
        tex_lines.append(r"\toprule")
        header = "Strategy & " + " & ".join(m.replace("@", "@") for m in args.metrics) + r" \\"
        tex_lines.append(header)
        tex_lines.append(r"\midrule")
        for strategy in args.strategies:
            row_cells = [c for c in ds_cells if c.strategy == strategy]
            if not row_cells:
                continue
            by_metric = {c.metric: c for c in row_cells}
            row = [strategy]
            for metric in args.metrics:
                c = by_metric.get(metric)
                if c is None:
                    row.append("--")
                    continue
                star = sig_marker(c.paired_p) if c.paired_p is not None else ""
                if c.n_seeds > 1:
                    cell = f"{c.mean:.3f}$_{{\\pm {c.std:.3f}}}$" + (f"$^{{{star}}}$" if star else "")
                else:
                    cell = f"{c.mean:.3f}" + (f"$^{{{star}}}$" if star else "")
                row.append(cell)
            tex_lines.append(" & ".join(row) + r" \\")
        tex_lines.append(r"\bottomrule")
        tex_lines.append(r"\end{tabular}")
        tex_lines.append("")

    tex_str = "\n".join(tex_lines)
    if args.out_tex:
        Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_tex, "w") as f:
            f.write(tex_str)
        print(f"Wrote {args.out_tex}")

    # Console summary
    for dataset in args.datasets:
        print(f"\n=== {dataset} (baseline = {args.baseline}) ===")
        print(f"{'strategy':<12} {'metric':<12} {'mean':>8} {'std':>8} {'p':>10} {'d':>8} sig n_seeds")
        for c in cells:
            if c.dataset != dataset:
                continue
            p_str = f"{c.paired_p:.4f}" if c.paired_p is not None else "  --"
            d_str = f"{c.cohen_d:+.3f}" if c.cohen_d is not None else "  --"
            print(f"{c.strategy:<12} {c.metric:<12} {c.mean:>8.4f} {c.std:>8.4f} "
                  f"{p_str:>10} {d_str:>8} {sig_marker(c.paired_p or 1.0):<3} {c.n_seeds}")


if __name__ == "__main__":
    main()
