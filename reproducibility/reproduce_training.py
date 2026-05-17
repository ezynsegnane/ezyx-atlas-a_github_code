from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVED_INDEX = PROJECT_ROOT / "reproducibility" / "archived_index" / "index_complete.parquet"
INITIAL_SNAPSHOT = PROJECT_ROOT / "published_snapshots" / "initial"
EXTENDED_SNAPSHOT = PROJECT_ROOT / "published_snapshots" / "extended"
VERIFY_SCRIPT = PROJECT_ROOT / "reproducibility" / "verify_reproducibility.py"


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
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def verify_published_inputs(python_exec: str) -> None:
    run_command(
        [python_exec, str(VERIFY_SCRIPT), "--mode", "both", "--verify-published"],
        cwd=PROJECT_ROOT,
        env=fixed_env(),
    )


def verify_outputs(python_exec: str, mode: str, candidate_dir: Path) -> None:
    run_command(
        [
            python_exec,
            str(VERIFY_SCRIPT),
            "--mode",
            mode,
            "--candidate-dir",
            str(candidate_dir),
        ],
        cwd=PROJECT_ROOT,
        env=fixed_env(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the published frozen training snapshots against the archived index."
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "extended", "all"],
        default="all",
        help="Which published training surface to execute.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the PTB-XL 1.0.3 root directory.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used inside the frozen environment.",
    )
    parser.add_argument(
        "--initial-output-dir",
        type=Path,
        default=PROJECT_ROOT / "reproducibility" / "runs_initial",
        help="Where to write regenerated initial-run outputs.",
    )
    parser.add_argument(
        "--extended-output-dir",
        type=Path,
        default=PROJECT_ROOT / "reproducibility" / "runs_extended",
        help="Where to write regenerated extended-run outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume through to the orchestrators.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare regenerated JSON outputs against the published reference files after each run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = fixed_env()
    verify_published_inputs(args.python)

    if not ARCHIVED_INDEX.exists():
        raise FileNotFoundError(f"Archived index not found: {ARCHIVED_INDEX}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"PTB-XL root not found: {args.data_root}")

    modes = ["initial", "extended"] if args.mode == "all" else [args.mode]

    if "initial" in modes:
        args.initial_output_dir.mkdir(parents=True, exist_ok=True)
        initial_cmd = [
            args.python,
            "run_multiseed_experiments.py",
            "--data_root",
            str(args.data_root),
            "--index_path",
            str(ARCHIVED_INDEX),
            "--runs_dir",
            str(args.initial_output_dir),
        ]
        if args.resume:
            initial_cmd.append("--resume")
        run_command(initial_cmd, cwd=INITIAL_SNAPSHOT, env=env)
        if args.verify:
            verify_outputs(args.python, "initial", args.initial_output_dir)

    if "extended" in modes:
        args.extended_output_dir.mkdir(parents=True, exist_ok=True)
        extended_cmd = [
            args.python,
            "run_extended_experiments.py",
            "--data_root",
            str(args.data_root),
            "--index_path",
            str(ARCHIVED_INDEX),
            "--runs_dir",
            str(args.extended_output_dir),
        ]
        if args.resume:
            extended_cmd.append("--resume")
        run_command(extended_cmd, cwd=EXTENDED_SNAPSHOT, env=env)
        if args.verify:
            verify_outputs(args.python, "extended", args.extended_output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
