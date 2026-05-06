"""Success rate vs k (num memories retrieved), one subplot per dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from load_data import (
    DATASETS,
    DATASET_TITLES,
    MEMORY_COLORS,
    MEMORY_DISPLAY,
    MEMORY_SYSTEMS,
    X_AXIS_LABEL,
    group_by_k,
    load_analysis,
    style_k_axis,
    success_rate_with_ci,
)

OUT_DIR = Path(__file__).parent / "figures" / "success_vs_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.6 * n_ds, 3.6), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        for mem in MEMORY_SYSTEMS:
            recs = load_analysis(dataset, mem)
            if not recs:
                continue
            grouped = group_by_k(recs)
            ks = sorted(grouped.keys())
            ys, lo, hi = [], [], []
            for k in ks:
                res = success_rate_with_ci(grouped[k])
                p, l, h = (None, 0.0, 0.0) if res is None else res
                ys.append(p)
                lo.append(l)
                hi.append(h)
            ax.errorbar(
                ks, ys, yerr=[lo, hi],
                marker="o", capsize=3,
                label=MEMORY_DISPLAY[mem], color=MEMORY_COLORS[mem],
            )

        ax.set_title(DATASET_TITLES[dataset])
        ax.set_xlabel(X_AXIS_LABEL)
        if i == 0:
            ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        style_k_axis(ax)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Success rate vs. k")
    fig.tight_layout(pad=0.4, w_pad=0.3, rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "overview.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
