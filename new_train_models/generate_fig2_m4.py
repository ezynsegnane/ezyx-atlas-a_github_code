# =============================================================================
# generate_fig2_m4.py  (v2 — journal-quality revision)
# M4 — Validation AUC trajectory: mean ± SD across 10 seeds per variant.
#
# Changes vs v1
# -------------
# - figsize enlarged to (11, 6.5) for better readability
# - dpi raised to 600 for PNG export (PDF is always vector)
# - Inline end-point annotations removed; final mean ± SD moved into the
#   legend label where text is always legible regardless of figure scale
# - All font sizes increased: axis labels 14 pt, ticks 12 pt, legend 11 pt
# - Line width raised to 2.5; shading alpha to 0.25
# - y-axis range padded by 0.004 on each side so the top curve is never
#   clipped
# - Grid alpha raised to 0.5 for better visibility on white background
# - constrained_layout replaces tight_layout to avoid label clipping
#
# Reads : results/seed_json/*.json  (30 existing runs: 3 variants × 10 seeds)
#         OR --seed_json_dir override
# Writes: figures/fig2_training_curves.pdf  (and .png)
#         + submission package figures/fig2_training_curves.pdf  (and .png)
#
# Usage:
#   python new_train_models/generate_fig2_m4.py
#   python new_train_models/generate_fig2_m4.py --seed_json_dir /path/to/jsons
# =============================================================================

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEED_JSON_DIR = PROJECT_ROOT / "results" / "seed_json"
DEFAULT_OUT_DIR       = PROJECT_ROOT / "figures"

# Submission-package figures directory (kept in sync automatically)
PACKAGE_FIG_DIR = (
    PROJECT_ROOT
    / "mdpi_mathematics_submission_package"
    / "MDPI_template_ACS"
    / "figures"
)

VARIANTS = ["none", "demo", "demo+anthro"]
VARIANT_LABELS = {
    "none":        "ECG only (none)",
    "demo":        "ECG + demographics (demo)",
    "demo+anthro": "ECG + all metadata (demo+anthro)",
}
VARIANT_COLORS = {
    "none":        "#4878CF",   # blue
    "demo":        "#D65F5F",   # red
    "demo+anthro": "#6ACC65",   # green
}

# Matplotlib global style
plt.rcParams.update({
    "font.family":       "serif",
    "axes.titlesize":    13,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   11,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "0.75",
})


# ---------------------------------------------------------------------------
def load_histories(seed_json_dir: Path) -> dict:
    """
    Returns dict: variant -> 2-D array of shape (n_seeds, n_epochs)
    aligned to the shortest run per variant.
    """
    raw = {v: [] for v in VARIANTS}

    for fpath in sorted(seed_json_dir.glob("*.json")):
        try:
            d = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        variant = d.get("metadata", {}).get("variant")
        if variant not in VARIANTS:
            continue
        hist = d.get("training_history", [])
        if not hist:
            continue
        aucs = [h["val_auc"] for h in hist]
        raw[variant].append(aucs)

    result = {}
    for v in VARIANTS:
        runs = raw[v]
        if not runs:
            print(f"  WARNING: no data for variant '{v}'")
            continue
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs])   # (n_seeds, n_epochs)
        result[v] = arr
        print(
            f"  {v}: {arr.shape[0]} seeds × {arr.shape[1]} epochs  "
            f"final={arr[:, -1].mean():.4f} ± {arr[:, -1].std(ddof=1):.4f}"
        )
    return result


# ---------------------------------------------------------------------------
def plot_trajectories(histories: dict, out_dirs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    y_vals_all = []

    for v in VARIANTS:
        if v not in histories:
            continue
        arr     = histories[v]
        n_ep    = arr.shape[1]
        epochs  = np.arange(1, n_ep + 1)
        mean    = arr.mean(axis=0)
        sd      = arr.std(axis=0, ddof=1)
        color   = VARIANT_COLORS[v]
        n_seeds = arr.shape[0]

        # Legend label includes the final mean ± SD so readers get the number
        # without having to squint at an end-point annotation
        final_m = arr[:, -1].mean()
        final_s = arr[:, -1].std(ddof=1)
        label = (
            f"{VARIANT_LABELS[v]}  (n={n_seeds})\n"
            f"  final: {final_m:.4f} ± {final_s:.4f}"
        )

        ax.plot(epochs, mean, color=color, linewidth=2.5, label=label)
        ax.fill_between(
            epochs, mean - sd, mean + sd,
            alpha=0.25, color=color, linewidth=0,
        )
        y_vals_all.extend((mean - sd).tolist() + (mean + sd).tolist())

    # --- axes ---
    ax.set_xlabel("Epoch", fontsize=14, labelpad=6)
    ax.set_ylabel("Macro-AUC (validation fold 9)", fontsize=14, labelpad=6)
    ax.set_title(
        "Validation macro-AUC trajectory — mean ± 1 SD across 10 seeds",
        fontsize=13, pad=10,
    )

    # Integer x-ticks for every epoch
    n_ep_max = max(h.shape[1] for h in histories.values())
    ax.set_xlim(0.5, n_ep_max + 0.5)
    ax.set_xticks(range(1, n_ep_max + 1))

    # y-axis: pad by 0.004 on each side so no curve is clipped
    if y_vals_all:
        y_lo = min(y_vals_all) - 0.004
        y_hi = max(y_vals_all) + 0.004
        ax.set_ylim(y_lo, y_hi)

    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(axis="x", linestyle=":",  alpha=0.3, linewidth=0.6)

    ax.legend(
        loc="lower right",
        fontsize=11,
        handlelength=2.2,
        labelspacing=0.8,
    )

    # --- save ---
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext, kw in [
            ("pdf", {}),
            ("png", {"dpi": 600}),
        ]:
            p = out_dir / f"fig2_training_curves.{ext}"
            fig.savefig(p, bbox_inches="tight", **kw)
            print(f"  Saved: {p}")

    plt.close(fig)


# ---------------------------------------------------------------------------
def print_summary_table(histories: dict) -> None:
    print("\n--- Validation AUC trajectory summary (mean ± SD) ---")
    header = f"{'Epoch':>6}"
    for v in VARIANTS:
        if v in histories:
            header += f"  {VARIANT_LABELS[v][:22]:>24}"
    print(header)
    print("-" * (6 + 26 * sum(v in histories for v in VARIANTS)))

    max_ep = max(histories[v].shape[1] for v in VARIANTS if v in histories)
    for ep in range(1, max_ep + 1):
        row = f"{ep:>6}"
        for v in VARIANTS:
            if v not in histories:
                continue
            arr = histories[v]
            if ep <= arr.shape[1]:
                m = arr[:, ep - 1].mean()
                s = arr[:, ep - 1].std(ddof=1)
                row += f"  {m:.4f} ± {s:.4f}      "
            else:
                row += "       —             "
        print(row)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4 v2 — Generate Fig. 2 validation AUC mean±SD trajectory"
    )
    parser.add_argument(
        "--seed_json_dir", type=str,
        default=str(DEFAULT_SEED_JSON_DIR),
        help="Directory with 30 seed-level JSON files",
    )
    parser.add_argument(
        "--out_dir", type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Primary output directory for figures",
    )
    parser.add_argument(
        "--no_package_sync", action="store_true",
        help="Skip copying to submission-package figures/ directory",
    )
    args = parser.parse_args()

    seed_json_dir = Path(args.seed_json_dir)
    out_dir       = Path(args.out_dir)

    # Collect output directories
    out_dirs = [out_dir]
    if not args.no_package_sync and PACKAGE_FIG_DIR.parent.is_dir():
        out_dirs.append(PACKAGE_FIG_DIR)

    print("=" * 70)
    print("M4 v2 — Validation trajectory mean±SD (Fig. 2, journal quality)")
    print("=" * 70)
    print(f"Reading JSONs from : {seed_json_dir}")
    for d in out_dirs:
        print(f"Output directory   : {d}")
    print()

    histories = load_histories(seed_json_dir)
    if not histories:
        print("ERROR: no valid JSON files found.")
        return

    print_summary_table(histories)

    print("\nGenerating figure...")
    plot_trajectories(histories, out_dirs)

    print("\n[OK] M4 v2 done.")


if __name__ == "__main__":
    main()
