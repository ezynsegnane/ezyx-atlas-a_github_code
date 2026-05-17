from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from verify_reproducibility import load_normalized_json, sha256_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVED_INDEX = PROJECT_ROOT / "reproducibility" / "archived_index" / "index_complete.parquet"
INITIAL_SNAPSHOT = PROJECT_ROOT / "published_snapshots" / "initial"
EXTENDED_SNAPSHOT = PROJECT_ROOT / "published_snapshots" / "extended"
VERIFY_SCRIPT = PROJECT_ROOT / "reproducibility" / "verify_reproducibility.py"

EXTENDED_EXPERIMENTS = {
    "H5H8": {
        "variant": "demo+anthro",
        "seed": 2029,
        "gate_hidden_dim": 1152,
        "lauc_weight": 0.08,
        "reference_name": "results_ext_H5H8_seed2029_glu1152_lauc0.08.json",
    },
    "H7-glu512": {
        "variant": "demo+anthro",
        "seed": 2026,
        "gate_hidden_dim": 512,
        "lauc_weight": 0.08,
        "reference_name": "results_ext_H7_glu512_seed2026_lauc0.08.json",
    },
    "H7-glu1152": {
        "variant": "demo+anthro",
        "seed": 2026,
        "gate_hidden_dim": 1152,
        "lauc_weight": 0.08,
        "reference_name": "results_ext_H7_glu1152_seed2026_lauc0.08.json",
    },
    "H7-glu2048": {
        "variant": "demo+anthro",
        "seed": 2026,
        "gate_hidden_dim": 2048,
        "lauc_weight": 0.08,
        "reference_name": "results_ext_H7_glu2048_seed2026_lauc0.08.json",
    },
    "M3": {
        "variant": "demo+anthro",
        "seed": 2026,
        "gate_hidden_dim": 1152,
        "lauc_weight": 0.00,
        "reference_name": "results_ext_M3_seed2026_glu1152_lauc0.00.json",
    },
}


def fixed_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return env


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def verify_published_inputs(python_exec: str, env: dict[str, str]) -> None:
    run_command(
        [python_exec, str(VERIFY_SCRIPT), "--mode", "both", "--verify-published"],
        cwd=PROJECT_ROOT,
        env=env,
    )


def compare_json(reference_path: Path, candidate_path: Path, label: str) -> None:
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference JSON missing for {label}: {reference_path}")
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate JSON missing for {label}: {candidate_path}")

    reference_sha = sha256_text(load_normalized_json(reference_path))
    candidate_sha = sha256_text(load_normalized_json(candidate_path))
    if reference_sha != candidate_sha:
        raise RuntimeError(
            f"Normalized JSON mismatch for {label}: "
            f"reference={reference_sha} candidate={candidate_sha}"
        )

    print(f"[ok] Normalized JSON match for {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab-friendly smoke test for the published reproducibility surface."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the PTB-XL 1.0.3 root directory.",
    )
    parser.add_argument(
        "--mode",
        choices=["infra", "initial", "extended", "both"],
        default="both",
        help="Whether to run only infrastructure checks or also one initial/extended reference run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used inside Colab.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "colab_runs_smoke",
        help="Directory where smoke-test outputs are written.",
    )
    parser.add_argument(
        "--initial-variant",
        choices=["none", "demo", "demo+anthro"],
        default="demo+anthro",
        help="Initial-study variant to replay for the smoke test.",
    )
    parser.add_argument(
        "--initial-seed",
        type=int,
        default=2024,
        help="Initial-study seed to replay for the smoke test.",
    )
    parser.add_argument(
        "--extended-id",
        choices=sorted(EXTENDED_EXPERIMENTS),
        default="H5H8",
        help="Complementary experiment to replay for the smoke test.",
    )
    return parser.parse_args()


def run_initial(args: argparse.Namespace, env: dict[str, str]) -> None:
    runs_dir = args.output_root / "initial"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "atlas_a_v5_multiseed.py",
        "--variant",
        args.initial_variant,
        "--seed",
        str(args.initial_seed),
        "--data_root",
        str(args.data_root),
        "--index_path",
        str(ARCHIVED_INDEX),
        "--runs_dir",
        str(runs_dir),
    ]
    run_command(cmd, cwd=INITIAL_SNAPSHOT, env=env)

    candidate_json = (
        runs_dir
        / f"ATLAS_A_v5_{args.initial_variant}_seed{args.initial_seed}"
        / f"results_{args.initial_variant}_seed{args.initial_seed}.json"
    )
    reference_json = (
        PROJECT_ROOT
        / "results"
        / "seed_json"
        / f"results_{args.initial_variant}_seed{args.initial_seed}.json"
    )
    compare_json(reference_json, candidate_json, f"initial {args.initial_variant} seed {args.initial_seed}")


def run_extended(args: argparse.Namespace, env: dict[str, str]) -> None:
    spec = EXTENDED_EXPERIMENTS[args.extended_id]
    runs_dir = args.output_root / "extended"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "atlas_a_v5_extended.py",
        "--variant",
        spec["variant"],
        "--seed",
        str(spec["seed"]),
        "--gate_hidden_dim",
        str(spec["gate_hidden_dim"]),
        "--lauc_weight",
        str(spec["lauc_weight"]),
        "--data_root",
        str(args.data_root),
        "--index_path",
        str(ARCHIVED_INDEX),
        "--runs_dir",
        str(runs_dir),
    ]
    run_command(cmd, cwd=EXTENDED_SNAPSHOT, env=env)

    run_tag = (
        f"ATLAS_A_v5_ext_{spec['variant']}_seed{spec['seed']}"
        f"_glu{spec['gate_hidden_dim']}"
        f"_lauc{spec['lauc_weight']:.2f}"
    )
    candidate_json = runs_dir / run_tag / f"results_ext_{run_tag}.json"
    reference_json = PROJECT_ROOT / "results" / "extended_json" / spec["reference_name"]
    compare_json(reference_json, candidate_json, f"extended {args.extended_id}")


def main() -> int:
    args = parse_args()
    env = fixed_env()

    if not ARCHIVED_INDEX.exists():
        raise FileNotFoundError(f"Archived index not found: {ARCHIVED_INDEX}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"PTB-XL root not found: {args.data_root}")

    verify_published_inputs(args.python, env)

    if args.mode in {"initial", "both"}:
        run_initial(args, env)

    if args.mode in {"extended", "both"}:
        run_extended(args, env)

    print("[done] Colab smoke test completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
