#!/usr/bin/env python3
"""
Build the PTB-XL metadata index used by the EZNX-ATLAS-A training pipeline.

This script faithfully reproduces the two-step process that produced the
working indices for this study:

  Step 1  (from notebook metadata_train_evaluated.ipynb):
    Reads ptbxl_database.csv, engineers metadata features (z-scores,
    availability masks, missingness indicators), and writes
    index_mm_core.parquet.

  Step 2  (from script fix_index.py):
    Merges index_mm_core.parquet with ptbxl_database.csv to add scp_codes,
    filename_lr, and filename_hr, then writes the final index_complete.parquet
    that eznx_loader_v2.py reads at training time.

NOTE: The previous version of this file (index_construction.py before v2.3.6)
was an exploratory prototype that only performed Step 1 and wrote
index_mm_core.parquet using filename_hr only, without scp_codes or filename_lr.
That prototype was INCOMPATIBLE with the training loader.  This version is the
correct, complete pipeline.

Usage
-----
    python index_construction.py [--data-root PATH] [--out-dir PATH]

Environment overrides
---------------------
    EZNX_DATA_REAL   Path to the extracted PTB-XL 1.0.3 directory
    EZNX_INDEX_OUT   Directory where output files are written
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATA_ROOT = Path(
    os.environ.get("EZNX_DATA_REAL", str(PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3"))
)
DEFAULT_OUT_DIR = Path(
    os.environ.get("EZNX_INDEX_OUT", str(PROJECT_ROOT / "data"))
)

META_FEATURES = [
    "age_z", "sex01",
    "height_z", "weight_z", "bmi_z",
    "miss__height", "miss__weight", "miss__bmi",
]
MASK_FEATURES = [
    "mask__age", "mask__sex",
    "mask__height", "mask__weight", "mask__bmi",
    "mask__miss_height", "mask__miss_weight", "mask__miss_bmi",
]


def clean_range(s: pd.Series, lo=None, hi=None) -> pd.Series:
    s = s.copy()
    if lo is not None:
        s = s.where(s >= lo, np.nan)
    if hi is not None:
        s = s.where(s <= hi, np.nan)
    return s


def norm_sex(x) -> int:
    """Encode PTB-XL sex in {0, 1}; return 0 for missing.
    In the public PTB-XL corpus sex is always observed (mask__sex = 1.0
    across all 21 799 records), so the fallback is never triggered."""
    if pd.isna(x):
        return 0
    try:
        xv = int(x)
        return xv if xv in (0, 1) else 0
    except Exception:
        return 0


def build_mm_core(df: pd.DataFrame, data_root: Path, out_dir: Path) -> pd.DataFrame:
    """Step 1 — feature engineering → index_mm_core.parquet.

    Reproduces the notebook cells in metadata_train_evaluated.ipynb.
    """
    df = df.copy()

    # ── Conservative range cleaning ──────────────────────────────────────────
    df["age"]    = clean_range(df["age"],    lo=0,   hi=120)
    df["height"] = clean_range(df["height"], lo=120, hi=210)
    df["weight"] = clean_range(df["weight"], lo=30,  hi=250)
    h_m = df["height"] / 100.0
    df["bmi_raw"] = df["weight"] / (h_m * h_m)
    df["bmi_raw"] = clean_range(df["bmi_raw"], lo=10, hi=60)

    print("NaN rates after cleaning:")
    print(df[["age", "height", "weight", "bmi_raw"]].isna().mean())

    # ── Sex encoding ──────────────────────────────────────────────────────────
    df["sex01"] = df["sex"].apply(norm_sex).astype(int)

    # ── Availability masks (before imputation) ────────────────────────────────
    df["mask__age"]    = df["age"].notna().astype(int)
    df["mask__sex"]    = df["sex"].notna().astype(int)
    df["mask__height"] = df["height"].notna().astype(int)
    df["mask__weight"] = df["weight"].notna().astype(int)
    df["mask__bmi"]    = df["bmi_raw"].notna().astype(int)

    df["miss__height"] = (1 - df["mask__height"]).astype(int)
    df["miss__weight"] = (1 - df["mask__weight"]).astype(int)
    df["miss__bmi"]    = (1 - df["mask__bmi"]).astype(int)

    df["meta_present_any"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) > 0
    ).astype(int)
    df["meta_present_strict"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) >= 2
    ).astype(int)

    print("\nRates:")
    print(df[["meta_present_any", "meta_present_strict",
              "mask__height", "mask__weight", "mask__bmi",
              "mask__age", "mask__sex"]].mean())

    # ── Imputation: train-split medians ───────────────────────────────────────
    train_mask = df["strat_fold"].between(1, 8)
    train_df   = df.loc[train_mask]

    impute_medians = {
        "age":     float(train_df["age"].median(skipna=True)),
        "height":  float(train_df["height"].median(skipna=True)),
        "weight":  float(train_df["weight"].median(skipna=True)),
        "bmi_raw": float(train_df["bmi_raw"].median(skipna=True)),
    }
    print("\nImpute medians (TRAIN):", impute_medians)

    for col in ["age", "height", "weight", "bmi_raw"]:
        df[col + "_imp"] = df[col].fillna(impute_medians[col])

    # ── Z-score normalisation: train mean / std ───────────────────────────────
    scaler: dict[str, dict[str, float]] = {}
    for col in ["age_imp", "height_imp", "weight_imp", "bmi_raw_imp"]:
        mu = float(df.loc[train_mask, col].mean())
        sd = float(df.loc[train_mask, col].std(ddof=0)) or 1.0
        scaler[col] = {"mean": mu, "std": sd}
    print("Scaler (TRAIN):", scaler)

    df["age_z"]    = (df["age_imp"]     - scaler["age_imp"]["mean"])     / scaler["age_imp"]["std"]
    df["height_z"] = (df["height_imp"]  - scaler["height_imp"]["mean"])  / scaler["height_imp"]["std"]
    df["weight_z"] = (df["weight_imp"]  - scaler["weight_imp"]["mean"])  / scaler["weight_imp"]["std"]
    df["bmi_z"]    = (df["bmi_raw_imp"] - scaler["bmi_raw_imp"]["mean"]) / scaler["bmi_raw_imp"]["std"]

    # ── Masks for the missingness-indicator features ───────────────────────────
    df["mask__miss_height"] = df["mask__height"].astype(int)
    df["mask__miss_weight"] = df["mask__weight"].astype(int)
    df["mask__miss_bmi"]    = df["mask__bmi"].astype(int)

    # ── Assemble index_mm_core ────────────────────────────────────────────────
    idx = df[["ecg_id", "patient_id", "strat_fold", "filename_hr"]].copy()
    idx["hea_path"] = idx["filename_hr"].apply(
        lambda p: str((data_root / p).with_suffix(".hea"))
    )

    meta_core_cols = (
        ["ecg_id", "patient_id", "strat_fold"]
        + META_FEATURES + MASK_FEATURES
        + ["meta_present_any", "meta_present_strict"]
    )
    meta_core = df[meta_core_cols].copy()

    mm = idx.merge(meta_core, on=["ecg_id", "patient_id", "strat_fold"], how="left")

    mm_out = out_dir / "index_mm_core.parquet"
    mm.to_parquet(mm_out, index=False)
    print(f"\nindex_mm_core.parquet -> {mm_out}  shape = {mm.shape}")

    # Quick sanity checks
    print("Any NaN in META_FEATURES?:", mm[META_FEATURES].isna().any().any())
    print("Any NaN in MASK_FEATURES?:", mm[MASK_FEATURES].isna().any().any())
    print("Split counts:")
    print(mm["strat_fold"].value_counts().sort_index())

    return mm


def build_complete(mm: pd.DataFrame, data_root: Path, out_dir: Path) -> pd.DataFrame:
    """Step 2 — merge with official PTB-XL columns → index_complete.parquet.

    Reproduces fix_index.py: loads scp_codes, filename_hr, filename_lr from
    ptbxl_database.csv and merges them onto the feature frame by ecg_id.
    """
    db_path = data_root / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL database not found: {db_path}")

    official_cols = [
        "ecg_id", "patient_id", "strat_fold",
        "scp_codes", "filename_hr", "filename_lr",
    ]
    official_df = pd.read_csv(db_path, usecols=official_cols)
    print(f"\nLoaded official PTB-XL database: {len(official_df)} rows "
          f"(includes patient_id, scp_codes, filename_hr, filename_lr)")

    # Drop columns from mm that will come from official_df to avoid duplicates
    drop_cols = [c for c in ["patient_id", "strat_fold", "scp_codes",
                              "filename_hr", "filename_lr"]
                 if c in mm.columns]
    if drop_cols:
        print(f"Dropping redundant columns before merge: {drop_cols}")
    mm_clean = mm.drop(columns=drop_cols)

    merged = pd.merge(mm_clean, official_df, on="ecg_id", how="inner")
    print(f"Merged index: {len(merged)} rows, {len(merged.columns)} columns")

    # Integrity check — columns required by eznx_loader_v2.py
    required = ["filename_lr", "filename_hr", "scp_codes", "patient_id", "strat_fold"]
    missing  = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(
            f"CRITICAL: columns required by the training loader are missing: {missing}"
        )

    out_path = out_dir / "index_complete.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"index_complete.parquet -> {out_path}")
    print("Done.")
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help="Path to the extracted PTB-XL 1.0.3 directory.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory where index_mm_core.parquet and index_complete.parquet are written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"PTB-XL root not found: '{args.data_root}'.\n"
            "Set the EZNX_DATA_REAL environment variable or pass --data-root."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw PTB-XL database ──────────────────────────────────────────────
    db_path  = args.data_root / "ptbxl_database.csv"
    raw_cols = ["ecg_id", "patient_id", "strat_fold",
                "filename_hr", "age", "sex", "height", "weight"]
    df = pd.read_csv(db_path, usecols=raw_cols)
    df["strat_fold"] = df["strat_fold"].astype(int)

    print(f"Loaded ptbxl_database.csv: {df.shape}")
    print(f"Train n = {int(df['strat_fold'].between(1, 8).sum())}")
    print(f"Val   n = {int(df['strat_fold'].eq(9).sum())}")
    print(f"Test  n = {int(df['strat_fold'].eq(10).sum())}")

    # ── Step 1: feature engineering → index_mm_core.parquet ──────────────────
    mm = build_mm_core(df, args.data_root, args.out_dir)

    # ── Step 2: add PTB-XL labels + paths → index_complete.parquet ───────────
    build_complete(mm, args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
