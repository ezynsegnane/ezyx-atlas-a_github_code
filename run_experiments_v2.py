"""
run_experiments_v2.py — Orchestrator for the EZNX-ATLAS-A v5 Scientific Reports campaign.

Experiment taxonomy (270 descriptors / 250 unique GPU runs)
────────────────────────────────────────────────────────────
  [A] Primary confirmatory         60 desc /  60 unique  3 variants × 20 seeds
  [B] Fusion capacity sensitivity  40 desc /  30 unique  meta_hid {32,64,128,256} × 10 seeds
  [C] LAUC-BCE balance sensitivity 50 desc /  40 unique  lauc {0,0.04,0.08,0.12,0.16} × 10 seeds
  [D] Augmentation sensitivity     20 desc /  20 unique  noaug × demo+anthro × 20 seeds
  [E] Architecture ablations       80 desc /  80 unique  8 modes × demo+anthro × 10 seeds
  [F] Multi-split sensitivity      20 desc /  20 unique  4 splits × demo+anthro × 5 seeds

Shared runs (auto-skipped by auto-resume):
  Group B meta_hid=128 × seeds 2024–2033 = Group A demo+anthro seeds 2024–2033 (10 runs)
  Group C lauc=0.08   × seeds 2024–2033 = Group A demo+anthro seeds 2024–2033 (10 runs)

Notes on Group B:
  meta_hid controls BOTH the fusion projection dimension (130→meta_hid) AND the gate hidden
  dimension. These roles are architecturally entangled. Group B measures sensitivity to
  fusion+gate capacity jointly. See analysis_plan.md for the full pre-declaration.

Notes on Group C:
  fused_w = max(0.0, 0.60 − lauc_w). Varying lauc_w simultaneously changes fused_w.
  Group C is labeled EXPLORATORY. See analysis_plan.md.

Notes on Group E:
  E8 (trainmask): standard arch trained with x_meta=zeros, mpm=zeros. Evaluation is standard.
  arch_mode=standard is Group A — NOT re-trained in Group E.

Notes on Group F:
  Fold selection rule (pre-declared): two early splits (2,3) and two mid-range splits (7,8).
  val = test + 1 in all cases. Target model: demo+anthro (pre-specified, independent of Group A).

All runs are idempotent: results_{run_name}.json triggers auto-resume in the training script.

Usage
─────
  python run_experiments_v2.py [options]

  --group      A | B | C | D | E | F | all  (default: all)
  --dry_run    Print the full run list without executing
  --data_root  Path to PTB-XL 1.0.3
  --index_path Path to index_complete.parquet
  --runs_dir   Output directory for run artefacts
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
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

# ── Seed ranges ───────────────────────────────────────────────────────────────
SEEDS_20 = list(range(2024, 2044))   # Groups A, C (10), D
SEEDS_10 = list(range(2024, 2034))   # Groups B, C, E
SEEDS_5  = list(range(2024, 2029))   # Group F

# ── Group-A taxonomy ──────────────────────────────────────────────────────────
VARIANTS = ["none", "demo", "demo+anthro"]

# ── Group-B: fusion capacity ──────────────────────────────────────────────────
META_HIDS = [32, 64, 128, 256]      # 128 shared with Group A (seeds 2024-2033)

# ── Group-C: LAUC-BCE balance ─────────────────────────────────────────────────
LAUC_WS = [0.0, 0.04, 0.08, 0.12, 0.16]   # 0.08 shared with Group A (seeds 2024-2033)

# ── Group-E: architecture ablations ──────────────────────────────────────────
# E1-E7 use arch_mode; E8 uses meta_mask_mode=trainmask with arch_mode=standard
ARCH_MODES = [
    "meta_only",    # E1: only metadata path drives fused head
    "concat",       # E2: plain concatenation, no gate
    "additive",     # E3: metadata additively injected into ECG repr, no gate
    "no_glu",       # E4: gate with ReLU (unbounded) instead of Sigmoid
    "no_residual",  # E5: ts_meta_residual injection removed
    "no_q_meta",    # E6: meta_quality = 1.0 always
    "no_dropout",   # E7: meta_dropout_p = 0.0
]

# ── Group-F: multi-split sensitivity ─────────────────────────────────────────
# Pre-declared rule: two early splits (2,3) and two mid-range splits (7,8).
# val = test + 1 in all cases.
GROUP_F_SPLITS = [
    {"test_fold": 2, "val_fold": 3},
    {"test_fold": 3, "val_fold": 4},
    {"test_fold": 7, "val_fold": 8},
    {"test_fold": 8, "val_fold": 9},
]

TRAINING_SCRIPT = PROJECT_ROOT / "atlas_a_v5_multiseed_v2.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Run name mirror (must match make_run_name() in atlas_a_v5_multiseed_v2.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_name(
    variant: str,
    seed: int,
    meta_hid: int = 128,
    lauc_weight: float = 0.08,
    no_aug: bool = False,
    arch_mode: str = "standard",
    meta_mask_mode: str = "real",
    test_fold: int = 10,
    val_fold: int = 9,
) -> str:
    parts = [f"ATLAS_A_v5_{variant}"]
    if meta_hid != 128:
        parts.append(f"metaH{meta_hid}")
    if abs(lauc_weight - 0.08) > 1e-6:
        parts.append(f"lauc{lauc_weight:g}")
    if no_aug:
        parts.append("noaug")
    if arch_mode != "standard":
        parts.append(arch_mode)
    if meta_mask_mode == "trainmask":
        parts.append("trainmask")
    if test_fold != 10 or val_fold != 9:
        parts.append(f"tf{test_fold}_vf{val_fold}")
    parts.append(f"seed{seed}")
    return "_".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment list builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_experiment_list(
    data_root: str,
    index_path: str,
    runs_dir: str,
    groups: str = "all",
) -> List[Dict[str, Any]]:
    """
    Return ordered list of experiment descriptors (270 total).
    Each descriptor has: group, run_name, args, result_file.
    """
    experiments: List[Dict[str, Any]] = []

    def _add(
        group: str,
        variant: str,
        seed: int,
        meta_hid: int = 128,
        lauc_weight: float = 0.08,
        no_aug: bool = False,
        arch_mode: str = "standard",
        meta_mask_mode: str = "real",
        test_fold: int = 10,
        val_fold: int = 9,
    ) -> None:
        name = _run_name(
            variant, seed, meta_hid, lauc_weight, no_aug,
            arch_mode, meta_mask_mode, test_fold, val_fold
        )
        result_file = Path(runs_dir) / name / f"results_{name}.json"
        cmd_args = [
            "--variant",        variant,
            "--seed",           str(seed),
            "--meta_hid",       str(meta_hid),
            "--lauc_weight",    str(lauc_weight),
            "--arch_mode",      arch_mode,
            "--meta_mask_mode", meta_mask_mode,
            "--test_fold",      str(test_fold),
            "--val_fold",       str(val_fold),
            "--data_root",      data_root,
            "--index_path",     index_path,
            "--runs_dir",       runs_dir,
        ]
        if no_aug:
            cmd_args.append("--no_aug")
        experiments.append({
            "group":       group,
            "run_name":    name,
            "result_file": result_file,
            "cmd_args":    cmd_args,
        })

    run_all   = (groups == "all")
    run_group = groups.upper() if not run_all else None

    # ── Group A: Primary confirmatory (60 descriptors, 60 unique) ─────────────
    if run_all or run_group == "A":
        for v in VARIANTS:
            for s in SEEDS_20:
                _add("A", variant=v, seed=s)

    # ── Group B: Fusion capacity sensitivity (40 desc, 30 unique) ─────────────
    # meta_hid=128 × seeds 2024–2033 → shared with Group A (auto-skipped)
    if run_all or run_group == "B":
        for mh in META_HIDS:
            for s in SEEDS_10:
                _add("B", variant="demo+anthro", seed=s, meta_hid=mh)

    # ── Group C: LAUC-BCE balance sensitivity (50 desc, 40 unique) ────────────
    # lauc=0.08 × seeds 2024–2033 → shared with Group A (auto-skipped)
    if run_all or run_group == "C":
        for lw in LAUC_WS:
            for s in SEEDS_10:
                _add("C", variant="demo+anthro", seed=s, lauc_weight=lw)

    # ── Group D: Augmentation sensitivity (20 desc, 20 unique) ────────────────
    if run_all or run_group == "D":
        for s in SEEDS_20:
            _add("D", variant="demo+anthro", seed=s, no_aug=True)

    # ── Group E: Architecture ablations (80 desc, 80 unique) ─────────────────
    if run_all or run_group == "E":
        # E1–E7: arch_mode ablations (7 modes × 10 seeds)
        for am in ARCH_MODES:
            for s in SEEDS_10:
                _add("E", variant="demo+anthro", seed=s, arch_mode=am)
        # E8: trainmask (standard arch + zero metadata training) × 10 seeds
        for s in SEEDS_10:
            _add("E", variant="demo+anthro", seed=s, meta_mask_mode="trainmask")

    # ── Group F: Multi-split sensitivity (20 desc, 20 unique) ────────────────
    if run_all or run_group == "F":
        for sp in GROUP_F_SPLITS:
            for s in SEEDS_5:
                _add(
                    "F", variant="demo+anthro", seed=s,
                    test_fold=sp["test_fold"], val_fold=sp["val_fold"]
                )

    return experiments


# ═══════════════════════════════════════════════════════════════════════════════
# Progress tracking
# ═══════════════════════════════════════════════════════════════════════════════

def load_progress(csv_path: Path) -> Dict[str, Dict]:
    """Load progress CSV → dict keyed by run_name."""
    progress = {}
    if not csv_path.exists():
        return progress
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            progress[row["run_name"]] = row
    return progress


def append_progress(csv_path: Path, row: Dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def is_valid_result_file(result_path: Path) -> bool:
    """Return True only for a readable results JSON with expected top-level keys."""
    if not result_path.exists():
        return False
    try:
        with open(result_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("metadata"), dict)
        and isinstance(payload.get("test"), dict)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group",      type=str, default="all",
                        help="Run a single group: A|B|C|D|E|F (default: all)")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print run list without executing")
    parser.add_argument("--data_root",  type=str,
                        default=os.getenv("EZNX_DATA_REAL",
                                          str(PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3")))
    parser.add_argument("--index_path", type=str,
                        default=os.getenv("EZNX_INDEX_PATH",
                                          str(PROJECT_ROOT / "data" / "index_complete.parquet")))
    parser.add_argument("--runs_dir",   type=str,
                        default=os.getenv("EZNX_RUNS_DIR",
                                          str(PROJECT_ROOT / "runs")))
    args = parser.parse_args()

    if args.group.lower() not in {"a", "b", "c", "d", "e", "f", "all"}:
        parser.error("--group must be one of A, B, C, D, E, F, all")

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / "progress.csv"

    experiments = build_experiment_list(
        data_root=args.data_root,
        index_path=args.index_path,
        runs_dir=str(runs_dir),
        groups=args.group,
    )

    # Summary
    from collections import Counter
    group_counts = Counter(e["group"] for e in experiments)
    print("=" * 70)
    print("EZNX-ATLAS-A v5 — Experiment Campaign")
    print("=" * 70)
    print(f"  Total descriptors: {len(experiments)}")
    for g in sorted(group_counts):
        print(f"    Group {g}: {group_counts[g]} descriptors")
    print(f"  Output dir: {runs_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Full run list:\n")
        for i, exp in enumerate(experiments, 1):
            done = "✓" if is_valid_result_file(exp["result_file"]) else ("!" if exp["result_file"].exists() else " ")
            print(f"  [{done}] {i:3d}/{len(experiments)}  [{exp['group']}]  {exp['run_name']}")
        print(f"\nTotal: {len(experiments)} descriptors")
        skippable = sum(1 for e in experiments if is_valid_result_file(e["result_file"]))
        print(f"Already done: {skippable} | Remaining: {len(experiments) - skippable}")
        return

    # Execute
    n_total    = len(experiments)
    n_done     = 0
    n_skipped  = 0
    n_failed   = 0
    t_campaign = time.time()

    for i, exp in enumerate(experiments, 1):
        run_name    = exp["run_name"]
        result_file = exp["result_file"]
        group       = exp["group"]

        print(f"\n[{i:3d}/{n_total}] [{group}] {run_name}")

        # Auto-resume check (mirrored from training script)
        if is_valid_result_file(result_file):
            print(f"   SKIP — already complete: {result_file.name}")
            n_skipped += 1
            continue
        if result_file.exists():
            print(f"   STALE — unreadable/incomplete results file detected, re-running: {result_file.name}")

        cmd = [sys.executable, str(TRAINING_SCRIPT)] + exp["cmd_args"]
        t0  = time.time()

        try:
            ret = subprocess.run(cmd, check=False)
            elapsed = time.time() - t0
            status  = "done" if ret.returncode == 0 else f"FAILED(rc={ret.returncode})"
            if ret.returncode == 0:
                n_done += 1
            else:
                n_failed += 1
                print(f"   ERROR: training returned rc={ret.returncode}")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Campaign paused. Re-run to resume from last run.")
            sys.exit(1)
        except Exception as exc:
            elapsed = time.time() - t0
            status  = f"EXCEPTION({exc})"
            n_failed += 1
            print(f"   EXCEPTION: {exc}")

        append_progress(csv_path, {
            "timestamp":   datetime.now().isoformat(),
            "group":       group,
            "run_name":    run_name,
            "status":      status,
            "elapsed_s":   f"{elapsed:.0f}",
        })

    # Final summary
    wall = time.time() - t_campaign
    print("\n" + "=" * 70)
    print(f"Campaign complete in {wall/3600:.1f} h")
    print(f"  Done:    {n_done}")
    print(f"  Skipped: {n_skipped}")
    print(f"  Failed:  {n_failed}")
    print(f"  Progress: {csv_path}")
    print("=" * 70)

    if n_failed > 0:
        print(f"\nWARNING: {n_failed} run(s) failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
