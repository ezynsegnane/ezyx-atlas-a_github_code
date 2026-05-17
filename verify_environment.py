"""
verify_environment.py -- Pre-flight environment check for the EZNX-ATLAS-A v5 campaign.

Checks:
  1. Python >= 3.11
  2. Required packages and minimum versions (including wfdb; xgboost optional)
  3. CUDA / GPU availability (warning only if absent)
  4. Data files: index_complete.parquet + PTB-XL ECG directory
  5. analysis_plan.md presence and SHA-256 hash (pre-declaration proof)
  6. Determinism environment variable (CUBLAS_WORKSPACE_CONFIG)

Exits with code 0 (all hard checks pass) or 1 (any hard check failed).
GPU and CUBLAS checks are warnings only (soft failures).

Note: All output uses ASCII-only characters for Windows cp1252 / Kaggle
console compatibility.

Usage
-----
  python verify_environment.py
      [--index_path /kaggle/working/index_complete.parquet]
      [--data_root  /kaggle/input/ptb-xl/1.0.3]
      [--expected_plan_sha256 <hash>]
"""

import argparse
import hashlib
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Minimum package versions ──────────────────────────────────────────────────
# Format: import_name -> (min_major, min_minor, min_patch)
# Note: min versions, not exact — environment may have newer compatible versions.
REQUIRED_PACKAGES = {
    "torch":      (2, 1, 0),
    "numpy":      (1, 24, 0),
    "pandas":     (2, 0, 0),
    "sklearn":    (1, 3, 0),
    "scipy":      (1, 11, 0),
    "matplotlib": (3, 7, 0),
    "pyarrow":    (12, 0, 0),
    "wfdb":       (4, 1, 0),   # ECG file I/O — required by eznx_loader_v2.py
}

# Optional packages: absence is reported as a warning but does not block training.
OPTIONAL_PACKAGES = {
    "xgboost":    (2, 0, 0),
}

# CUBLAS determinism env variable
CUBLAS_VAR   = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":16:8"

# Status symbols — ASCII only (cp1252 / Kaggle safe)
OK   = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"


def _check_python() -> bool:
    ok  = sys.version_info >= (3, 11)
    ver = (f"Python {sys.version_info.major}.{sys.version_info.minor}"
           f".{sys.version_info.micro}")
    sym = OK if ok else FAIL
    print(f"  {sym} {ver} {'(OK)' if ok else '(need >= 3.11)'}")
    return ok


def _parse_version(ver_str: str):
    """Parse version string like '2.1.0+cu121' -> (2, 1, 0)."""
    parts = ver_str.split("+")[0].split(".")
    try:
        return tuple(int(p) for p in parts[:3])
    except ValueError:
        return (0, 0, 0)


def _check_packages() -> bool:
    all_ok = True
    for pkg, min_ver in REQUIRED_PACKAGES.items():
        import_name = "sklearn" if pkg == "sklearn" else pkg
        try:
            mod     = importlib.import_module(import_name)
            ver_str = getattr(mod, "__version__", "0.0.0")
            ver     = _parse_version(ver_str)
            ok      = ver >= min_ver
            min_str = ".".join(map(str, min_ver))
            sym     = OK if ok else FAIL
            note    = "(OK)" if ok else f"(need >= {min_str})"
            print(f"  {sym} {pkg:<14}: {ver_str} {note}")
            all_ok = all_ok and ok
        except ImportError:
            print(f"  {FAIL} {pkg:<14}: NOT INSTALLED")
            all_ok = False
    for pkg, min_ver in OPTIONAL_PACKAGES.items():
        import_name = pkg
        try:
            mod     = importlib.import_module(import_name)
            ver_str = getattr(mod, "__version__", "0.0.0")
            ver     = _parse_version(ver_str)
            ok      = ver >= min_ver
            min_str = ".".join(map(str, min_ver))
            sym     = OK if ok else WARN
            note    = "(OK)" if ok else f"(optional; recommend >= {min_str})"
            print(f"  {sym} {pkg:<14}: {ver_str} {note}")
        except ImportError:
            print(f"  {WARN} {pkg:<14}: NOT INSTALLED (optional; XGBoost baseline will be skipped)")
    return all_ok


def _check_gpu() -> bool:
    """Soft check — warns but does not fail if GPU is absent."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            print(f"  {OK} GPU: {name} ({mem} GB)")
            return True
        else:
            print(f"  {WARN} GPU: CUDA not available -- training will run on CPU (very slow)")
            return False
    except ImportError:
        print(f"  {WARN} GPU check skipped (torch not installed)")
        return False


def _check_cublas() -> bool:
    """Soft check — training script sets this programmatically before torch import."""
    val = os.environ.get(CUBLAS_VAR, "")
    ok  = val == CUBLAS_VALUE
    sym = OK if ok else WARN
    note = "(OK)" if ok else f"(expected {CUBLAS_VALUE!r} -- set before torch import)"
    print(f"  {sym} {CUBLAS_VAR}={val!r} {note}")
    return ok


def _check_data(index_path: str, data_root: str) -> bool:
    all_ok = True

    # Index parquet
    p = Path(index_path)
    if p.exists():
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  {OK} Index:  {p} ({size_mb:.1f} MB)")
    else:
        print(f"  {FAIL} Index:  {p} -- FILE NOT FOUND")
        all_ok = False

    # PTB-XL directory + key files
    d = Path(data_root)
    if d.is_dir():
        for fname in ["ptbxl_database.csv", "scp_statements.csv"]:
            if (d / fname).exists():
                print(f"  {OK} {fname}")
            else:
                print(f"  {FAIL} {fname} -- NOT FOUND in {d}")
                all_ok = False
        # Count ECG .dat files (lazy, stop after first find)
        found_dat = any(True for _ in d.rglob("*.dat"))
        if found_dat:
            print(f"  {OK} PTB-XL ECG records (.dat) found under {d}")
        else:
            print(f"  {WARN} PTB-XL dir exists but no .dat files found under {d}")
    else:
        print(f"  {FAIL} PTB-XL: {d} -- DIRECTORY NOT FOUND")
        all_ok = False

    return all_ok


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_analysis_plan(expected_sha: str = "") -> bool:
    plan_path = PROJECT_ROOT / "analysis_plan.md"
    if not plan_path.exists():
        print(f"  {FAIL} analysis_plan.md: NOT FOUND at {plan_path}")
        return False

    sha = _sha256_file(plan_path)
    if expected_sha:
        ok  = sha == expected_sha
        sym = OK if ok else FAIL
        print(f"  {sym} analysis_plan.md SHA256={sha}")
        if not ok:
            print(f"    Expected: {expected_sha}")
        return ok
    else:
        print(f"  {OK} analysis_plan.md present  SHA256={sha}")
        return True


def main() -> None:
    # Ensure stdout can handle ASCII-only output on all platforms
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Pre-flight environment check for EZNX-ATLAS-A v5."
    )
    parser.add_argument("--index_path",
                        default=os.getenv("EZNX_INDEX_PATH",
                                          str(PROJECT_ROOT / "data" /
                                              "index_complete.parquet")))
    parser.add_argument("--data_root",
                        default=os.getenv("EZNX_DATA_REAL",
                                          str(PROJECT_ROOT / "data" /
                                              "ptb-xl" / "1.0.3")))
    parser.add_argument("--expected_plan_sha256", default="",
                        help="Expected SHA-256 of analysis_plan.md (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("EZNX-ATLAS-A v5 -- Environment Verification")
    print("=" * 60)

    results: dict = {}

    print("\n[1] Python version")
    results["python"] = _check_python()

    print("\n[2] Required packages (minimum versions)")
    results["packages"] = _check_packages()

    print("\n[3] GPU / CUDA  (warning only)")
    results["gpu"] = _check_gpu()

    print("\n[4] Determinism env variable  (warning only)")
    results["cublas"] = _check_cublas()

    print("\n[5] Data files")
    results["data"] = _check_data(args.index_path, args.data_root)

    print("\n[6] Pre-declaration proof (analysis_plan.md)")
    results["plan"] = _check_analysis_plan(args.expected_plan_sha256)

    # Summary
    print("\n" + "=" * 60)
    hard_checks = ["python", "packages", "data", "plan"]
    soft_checks = ["gpu", "cublas"]
    all_hard_ok = all(results[k] for k in hard_checks)

    print("SUMMARY:")
    for k, v in results.items():
        if k in soft_checks:
            sym = OK if v else WARN
        else:
            sym = OK if v else FAIL
        print(f"  {sym} {k}")

    if all_hard_ok:
        print("\n" + OK + " All critical checks passed -- environment ready.")
        sys.exit(0)
    else:
        failed = [k for k in hard_checks if not results[k]]
        print(f"\n{FAIL} {len(failed)} critical check(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
