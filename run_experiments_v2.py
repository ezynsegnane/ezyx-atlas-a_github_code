"""
run_experiments_v2.py — Orchestrator for the 170-run Scientific Reports campaign.

Experiment taxonomy (170 total)
────────────────────────────────
  [A] Primary ablation         60 runs   3 variants × 20 seeds (2024–2043)
  [B] GLU-width sensitivity    60 runs   3 meta_hid × 20 seeds (demo+anthro only)
  [C] LAUC-weight sensitivity  40 runs   2 lauc_w   × 20 seeds (demo+anthro only)
  [D] No-augmentation          10 runs   10 seeds (2024–2033) (demo+anthro only)

All runs are idempotent: if results_{run_name}.json already exists the training
script skips that run, so the orchestrator can be restarted after interruption.

Usage (Kaggle / local)
──────────────────────
  python run_experiments_v2.py [options]

  --data_root   Path to PTB-XL 1.0.3 directory (default: env EZNX_DATA_REAL)
  --index_path  Path to index_complete.parquet  (default: env EZNX_INDEX_PATH)
  --runs_dir    Output run directory            (default: env EZNX_RUNS_DIR)
  --group       Run only one experiment group: A | B | C | D | all (default: all)
  --dry_run     Print the full run list without executing (useful for auditing)
  --resume_csv  Path to write a CSV progress file (default: runs_dir/progress.csv)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent

SEEDS_20  = list(range(2024, 2044))   # 20 seeds: 2024…2043  (groups A, B, C)
SEEDS_10  = list(range(2024, 2034))   # 10 seeds: 2024…2033  (group D)

VARIANTS  = ["none", "demo", "demo+anthro"]
META_HIDS = [64, 128, 256]            # group B: [64, 256] are non-default; 128 = default
LAUC_WS   = [0.00, 0.16]             # group C: remove / double the margin term

TRAINING_SCRIPT = PROJECT_ROOT / "atlas_a_v5_multiseed_v2.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Run descriptor helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run_name(variant: str, seed: int, meta_hid: int = 128,
              lauc_weight: float = 0.08, no_aug: bool = False) -> str:
    """Mirror the naming logic in atlas_a_v5_multiseed_v2.py::make_run_name."""
    parts = [f"ATLAS_A_v5_{variant}"]
    if meta_hid != 128:
        parts.append(f"metaH{meta_hid}")
    if abs(lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{lauc_weight:g}")
    if no_aug:
        parts.append("noaug")
    parts.append(f"seed{seed}")
    return "_".join(parts)


def build_experiment_list(
    data_root: str, index_path: str, runs_dir: str, groups: str = "all"
) -> List[Dict[str, Any]]:
    """Return ordered list of experiment descriptors."""
    experiments: List[Dict[str, Any]] = []

    def _add(group: str, variant: str, seed: int,
             meta_hid: int = 128, lauc_weight: float = 0.08, no_aug: bool = False):
        name = _run_name(variant, seed, meta_hid, lauc_weight, no_aug)
        result_file = Path(runs_dir) / name / f"results_{name}.json"
        experiments.append({
            "group":       group,
            "run_name":    name,
            "result_file": str(result_file),
            "cmd_args": [
                sys.executable, str(TRAINING_SCRIPT),
                "--variant",     variant,
                "--seed",        str(seed),
                "--meta_hid",    str(meta_hid),
                "--lauc_weight", str(lauc_weight),
                "--data_root",   data_root,
                "--index_path",  index_path,
                "--runs_dir",    runs_dir,
            ] + (["--no_aug"] if no_aug else []),
        })

    do_all = (groups == "all")

    # ── Group A: Primary ablation  (60 runs) ─────────────────────────────────
    if do_all or "A" in groups:
        for variant in VARIANTS:
            for seed in SEEDS_20:
                _add("A", variant, seed)

    # ── Group B: GLU-width / meta_hid sensitivity  (60 runs) ─────────────────
    #   meta_hid = 128 (default) is already in group A → include all 3 widths
    #   so reviewers can see the full sweep in one table.
    if do_all or "B" in groups:
        for mh in META_HIDS:
            for seed in SEEDS_20:
                _add("B", "demo+anthro", seed, meta_hid=mh)

    # ── Group C: LAUC-weight sensitivity  (40 runs) ───────────────────────────
    if do_all or "C" in groups:
        for lw in LAUC_WS:
            for seed in SEEDS_20:
                _add("C", "demo+anthro", seed, lauc_weight=lw)

    # ── Group D: No-augmentation sensitivity  (10 runs) ───────────────────────
    if do_all or "D" in groups:
        for seed in SEEDS_10:
            _add("D", "demo+anthro", seed, no_aug=True)

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
# Progress tracking
# ═══════════════════════════════════════════════════════════════════════════════

def write_progress_csv(experiments: List[Dict], csv_path: Path) -> None:
    """Append / overwrite progress CSV with current status."""
    rows = []
    for exp in experiments:
        done = Path(exp["result_file"]).exists()
        macro_auc = ""
        if done:
            try:
                with open(exp["result_file"], encoding="utf-8") as f:
                    r = json.load(f)
                macro_auc = r.get("test", {}).get("macro_auc", "")
            except Exception:
                pass
        rows.append({
            "group":     exp["group"],
            "run_name":  exp["run_name"],
            "done":      "1" if done else "0",
            "macro_auc": macro_auc,
        })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "run_name", "done", "macro_auc"])
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root",   type=str, default=None)
    parser.add_argument("--index_path",  type=str, default=None)
    parser.add_argument("--runs_dir",    type=str, default=None)
    parser.add_argument("--group",       type=str, default="all",
                        help="all | A | B | C | D (can combine: 'AB')")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Print run list without executing")
    parser.add_argument("--resume_csv",  type=str, default=None)
    args = parser.parse_args()

    # Resolve paths from env or CLI
    data_root  = args.data_root  or os.getenv("EZNX_DATA_REAL",  "")
    index_path = args.index_path or os.getenv("EZNX_INDEX_PATH", "")
    runs_dir   = args.runs_dir   or os.getenv("EZNX_RUNS_DIR",   str(PROJECT_ROOT / "runs"))

    if not data_root:
        sys.exit("ERROR: --data_root (or EZNX_DATA_REAL env) is required.")
    if not index_path:
        sys.exit("ERROR: --index_path (or EZNX_INDEX_PATH env) is required.")

    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    resume_csv = Path(args.resume_csv or (Path(runs_dir) / "progress.csv"))

    experiments = build_experiment_list(data_root, index_path, runs_dir, args.group)
    n_total     = len(experiments)

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'GROUP':<6} {'RUN NAME':<70} {'DONE'}")
        print("-" * 90)
        groups_seen: Dict[str, int] = {}
        for exp in experiments:
            g  = exp["group"]
            done = "✓" if Path(exp["result_file"]).exists() else "·"
            groups_seen[g] = groups_seen.get(g, 0) + 1
            print(f"{g:<6} {exp['run_name']:<70} {done}")
        print("-" * 90)
        print(f"\nTotal: {n_total} runs")
        for g, cnt in sorted(groups_seen.items()):
            print(f"  Group {g}: {cnt}")
        write_progress_csv(experiments, resume_csv)
        print(f"\nProgress CSV: {resume_csv}")
        return

    # ── Execute ──────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"EZNX-ATLAS-A  Scientific Reports  —  {n_total} experiments")
    print(f"  data_root:  {data_root}")
    print(f"  index_path: {index_path}")
    print(f"  runs_dir:   {runs_dir}")
    print(f"  Started:    {datetime.now().isoformat()}")
    print("=" * 80)

    n_done = sum(Path(e["result_file"]).exists() for e in experiments)
    print(f"  Already complete: {n_done}/{n_total}")

    campaign_start = time.time()
    for i, exp in enumerate(experiments, 1):
        if Path(exp["result_file"]).exists():
            print(f"[{i:3d}/{n_total}] SKIP  {exp['run_name']}")
            continue

        print(f"\n[{i:3d}/{n_total}] RUN   {exp['run_name']}  "
              f"({datetime.now().strftime('%H:%M:%S')})")

        t0 = time.time()
        ret = subprocess.run(exp["cmd_args"], check=False)
        elapsed = time.time() - t0

        status = "OK" if ret.returncode == 0 else f"FAIL(rc={ret.returncode})"
        print(f"          → {status}  {elapsed/60:.1f} min")

        # Update CSV after every run so progress survives interruption
        write_progress_csv(experiments, resume_csv)

    campaign_elapsed = (time.time() - campaign_start) / 60
    n_done_final = sum(Path(e["result_file"]).exists() for e in experiments)

    print("\n" + "=" * 80)
    print(f"Campaign complete: {n_done_final}/{n_total} runs succeeded")
    print(f"Total wall time:   {campaign_elapsed:.1f} min")
    print(f"Progress CSV:      {resume_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
