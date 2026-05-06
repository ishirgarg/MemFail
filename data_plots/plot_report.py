"""Generate the full report bundle: summary tables + per-dataset / per-model plots.

Output layout under data_plots/figures/:

  tables/
    summary_scores.md
    summary_tokens.md
  success_vs_k/<model>/<dataset>.png
  perf_vs_model/<memory>.png
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from load_data import (
    DATASETS,
    DATASET_TITLES,
    MEMORY_COLORS,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    MODEL_COLORS,
    REPORT_MODELS,
    X_AXIS_LABEL,
    is_correct,
    load_analysis,
    style_k_axis,
    success_rate_with_ci,
)

FIG_DIR = Path(__file__).parent / "figures"

REPORT_KS = [5, 10, 15]


# ----------------------------- aggregation helpers -----------------------------

def success_rate(records) -> Optional[float]:
    if not records:
        return None
    return sum(1 for r in records if is_correct(r)) / len(records)


def avg_tokens(records) -> Optional[float]:
    if not records:
        return None
    totals = [
        (r.get("eval_input_tokens") or 0) + (r.get("eval_output_tokens") or 0)
        for r in records
    ]
    return sum(totals) / len(totals)


def collect_records(dataset: str, memory: str, model: str) -> Dict[int, list]:
    recs = load_analysis(dataset, memory, model=model)
    out: Dict[int, list] = defaultdict(list)
    for r in recs:
        out[r["_k"]].append(r)
    return out


# ----------------------------------- tables -----------------------------------

def _fmt_score(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.3f}"


def _fmt_int(v: Optional[float]) -> str:
    return "--" if v is None else f"{v:.0f}"


def _build_table(metric: str) -> str:
    """metric: 'score' or 'tokens'."""
    cols: List[tuple] = [(model, k) for model in REPORT_MODELS for k in REPORT_KS]
    header_top = "| Dataset | Memory | " + " | ".join(
        f"{model} k={k}" for model, k in cols
    ) + " |"
    sep = "|" + "|".join(["---"] * (2 + len(cols))) + "|"

    lines = [header_top, sep]
    for dataset in DATASETS:
        # Compute every cell first so we can find per-column maxima within the dataset.
        rows: Dict[str, Dict[tuple, Optional[float]]] = {}
        for memory in MEMORY_SYSTEMS:
            cells: Dict[tuple, Optional[float]] = {}
            for model, k in cols:
                bucket = collect_records(dataset, memory, model).get(k, [])
                if not bucket:
                    cells[(model, k)] = None
                elif metric == "score":
                    cells[(model, k)] = success_rate(bucket)
                else:
                    cells[(model, k)] = avg_tokens(bucket)
            rows[memory] = cells

        # Pick best per column (max score, min tokens — both still highlight the "winner").
        best: Dict[tuple, Optional[str]] = {}
        for col in cols:
            vals = [(m, rows[m][col]) for m in MEMORY_SYSTEMS if rows[m][col] is not None]
            if not vals:
                best[col] = None
                continue
            if metric == "score":
                best[col] = max(vals, key=lambda x: x[1])[0]
            else:
                best[col] = min(vals, key=lambda x: x[1])[0]

        for i, memory in enumerate(MEMORY_SYSTEMS):
            cells = rows[memory]
            label_dataset = DATASET_TITLES[dataset] if i == 0 else ""
            cell_strs = []
            for col in cols:
                v = cells[col]
                s = _fmt_score(v) if metric == "score" else _fmt_int(v)
                if best[col] == memory and v is not None:
                    s = f"**{s}**"
                cell_strs.append(s)
            lines.append(
                f"| {label_dataset} | {MEMORY_DISPLAY[memory]} | " + " | ".join(cell_strs) + " |"
            )
    return "\n".join(lines) + "\n"


def write_tables() -> None:
    out_dir = FIG_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    score_md = "# Mean success rate per memory system\n\n"
    score_md += "Bold = best score in each (dataset, column).\n\n"
    score_md += _build_table("score")
    (out_dir / "summary_scores.md").write_text(score_md, encoding="utf-8")
    print(f"wrote {out_dir / 'summary_scores.md'}")

    tok_md = "# Average tokens per graded question\n\n"
    tok_md += "Sum of `eval_input_tokens + eval_output_tokens`. Bold = lowest in each (dataset, column).\n\n"
    tok_md += _build_table("tokens")
    (out_dir / "summary_tokens.md").write_text(tok_md, encoding="utf-8")
    print(f"wrote {out_dir / 'summary_tokens.md'}")


# ------------------------- success-vs-k (per test-taker) -----------------------

def plot_success_vs_k_per_model() -> None:
    base = FIG_DIR / "success_vs_k"
    base.mkdir(parents=True, exist_ok=True)
    n_ds = len(DATASETS)
    for model in REPORT_MODELS:
        fig, axes = plt.subplots(1, n_ds, figsize=(3.6 * n_ds, 3.6), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        any_curve = False
        for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
            ds_any = False
            for memory in MEMORY_SYSTEMS:
                grouped = collect_records(dataset, memory, model)
                ks = sorted(grouped.keys())
                if not ks:
                    continue
                ys, lo, hi = [], [], []
                for k in ks:
                    res = success_rate_with_ci(grouped[k])
                    p, l, h = (None, 0.0, 0.0) if res is None else res
                    ys.append(p)
                    lo.append(l)
                    hi.append(h)
                ax.errorbar(
                    ks, ys, yerr=[lo, hi],
                    marker="o", capsize=3, label=MEMORY_DISPLAY[memory],
                    color=MEMORY_COLORS[memory],
                )
                ds_any = True
                any_curve = True
            ax.set_title(DATASET_TITLES[dataset])
            ax.set_xlabel(X_AXIS_LABEL)
            if i == 0:
                ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            style_k_axis(ax)
            ax.grid(True, alpha=0.3)
            if ds_any:
                ax.legend()
        fig.suptitle(f"Success rate vs. k — {model}")
        fig.tight_layout(pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.94])
        out = base / f"{model}.pdf"
        if any_curve:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
            print(f"wrote {out}")
        plt.close(fig)


# ----------------------- performance vs test-taker model ----------------------

def plot_perf_vs_model() -> None:
    out_dir = FIG_DIR / "perf_vs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ds = len(DATASETS)
    for memory in MEMORY_SYSTEMS:
        fig, axes = plt.subplots(1, n_ds, figsize=(3.6 * n_ds, 3.6), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        any_curve = False
        for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
            ds_any = False
            for model in REPORT_MODELS:
                grouped = collect_records(dataset, memory, model)
                ks = sorted(grouped.keys())
                if not ks:
                    continue
                ys, lo, hi = [], [], []
                for k in ks:
                    res = success_rate_with_ci(grouped[k])
                    p, l, h = (None, 0.0, 0.0) if res is None else res
                    ys.append(p)
                    lo.append(l)
                    hi.append(h)
                ax.errorbar(
                    ks, ys, yerr=[lo, hi],
                    marker="o", capsize=3, label=model,
                    color=MODEL_COLORS.get(model),
                )
                ds_any = True
                any_curve = True
            ax.set_title(DATASET_TITLES[dataset])
            ax.set_xlabel(X_AXIS_LABEL)
            if i == 0:
                ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1)
            style_k_axis(ax)
            ax.grid(True, alpha=0.3)
            if ds_any:
                ax.legend()
        for ax in axes[n_ds:]:
            ax.axis("off")
        fig.suptitle(f"Success vs. k by test-taker model — {MEMORY_DISPLAY[memory]}")
        fig.tight_layout(pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.94])
        out = out_dir / f"{memory}.pdf"
        if any_curve:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
            print(f"wrote {out}")
        plt.close(fig)


# ------------------------------------ main ------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_tables()
    plot_success_vs_k_per_model()
    plot_perf_vs_model()


if __name__ == "__main__":
    main()
