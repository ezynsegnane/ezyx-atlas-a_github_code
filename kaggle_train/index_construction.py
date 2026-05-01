import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "ptb-xl" / "1.0.3"

DATA = Path(os.getenv("EZNX_DATA_REAL", DEFAULT_DATA))
OUT_DIR = Path(os.getenv("EZNX_INDEX_OUT_DIR", PROJECT_ROOT / "data"))

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


def preview_frame(df: pd.DataFrame, n: int = 3) -> None:
    print(df.head(n).to_string(index=False))


def clean_range(s, lo=None, hi=None):
    s = s.copy()
    if lo is not None:
        s = s.where(s >= lo, np.nan)
    if hi is not None:
        s = s.where(s <= hi, np.nan)
    return s


def norm_sex(x):
    if pd.isna(x):
        raise ValueError(
            "PTB-XL sex is expected to be observed in this release; "
            "refuse to silently impute a missing value."
        )
    try:
        xv = int(x)
    except Exception as exc:
        raise ValueError(f"Unexpected sex value: {x!r}") from exc
    if xv not in (0, 1):
        raise ValueError(f"Unexpected sex code: {xv!r}")
    return xv


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"PTB-XL root not found at '{DATA}'. Set EZNX_DATA_REAL to the extracted 1.0.3 folder."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "ptbxl_database.csv")
    keep_cols = ["ecg_id", "patient_id", "strat_fold", "filename_hr", "age", "sex", "height", "weight"]
    df = df[keep_cols].copy()

    print("DATA =", DATA)
    print("OUT_DIR =", OUT_DIR)
    print("Shape df:", df.shape)
    preview_frame(df)

    df["strat_fold"] = df["strat_fold"].astype(int)

    train_mask = df["strat_fold"].between(1, 8)
    val_mask = df["strat_fold"].eq(9)
    test_mask = df["strat_fold"].eq(10)

    print("Train n =", int(train_mask.sum()))
    print("Val n   =", int(val_mask.sum()))
    print("Test n  =", int(test_mask.sum()))

    df["age"] = clean_range(df["age"], lo=0, hi=120)
    df["height"] = clean_range(df["height"], lo=120, hi=210)
    df["weight"] = clean_range(df["weight"], lo=30, hi=250)

    h_m = df["height"] / 100.0
    df["bmi_raw"] = df["weight"] / (h_m * h_m)
    df["bmi_raw"] = clean_range(df["bmi_raw"], lo=10, hi=60)

    print("NaN rates after cleaning:")
    print(df[["age", "height", "weight", "bmi_raw"]].isna().mean())

    df["sex01"] = df["sex"].apply(norm_sex).astype(int)

    df["mask__age"] = df["age"].notna().astype(int)
    df["mask__sex"] = df["sex"].notna().astype(int)
    df["mask__height"] = df["height"].notna().astype(int)
    df["mask__weight"] = df["weight"].notna().astype(int)
    df["mask__bmi"] = df["bmi_raw"].notna().astype(int)

    df["miss__height"] = (1 - df["mask__height"]).astype(int)
    df["miss__weight"] = (1 - df["mask__weight"]).astype(int)
    df["miss__bmi"] = (1 - df["mask__bmi"]).astype(int)

    df["meta_present_any"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) > 0
    ).astype(int)
    df["meta_present_strict"] = (
        (df["mask__height"] + df["mask__weight"] + df["mask__bmi"]) >= 2
    ).astype(int)

    print("Rates:")
    print(
        df[
            [
                "meta_present_any",
                "meta_present_strict",
                "mask__height",
                "mask__weight",
                "mask__bmi",
                "mask__age",
                "mask__sex",
            ]
        ].mean()
    )

    print("\nCounts meta_present_any:")
    print(df["meta_present_any"].value_counts())

    print("\nCounts meta_present_strict:")
    print(df["meta_present_strict"].value_counts())

    train_df = df.loc[train_mask].copy()
    impute_medians = {
        "age": float(train_df["age"].median(skipna=True)),
        "height": float(train_df["height"].median(skipna=True)),
        "weight": float(train_df["weight"].median(skipna=True)),
        "bmi_raw": float(train_df["bmi_raw"].median(skipna=True)),
    }
    print("Impute medians (TRAIN):", impute_medians)

    for col in ["age", "height", "weight", "bmi_raw"]:
        df[col + "_imp"] = df[col].fillna(impute_medians[col])

    scaler = {}
    for col in ["age_imp", "height_imp", "weight_imp", "bmi_raw_imp"]:
        mu = float(df.loc[train_mask, col].mean())
        sd = float(df.loc[train_mask, col].std(ddof=0)) or 1.0
        scaler[col] = {"mean": mu, "std": sd}
    print("Scaler (TRAIN):", scaler)

    df["age_z"] = (df["age_imp"] - scaler["age_imp"]["mean"]) / scaler["age_imp"]["std"]
    df["height_z"] = (df["height_imp"] - scaler["height_imp"]["mean"]) / scaler["height_imp"]["std"]
    df["weight_z"] = (df["weight_imp"] - scaler["weight_imp"]["mean"]) / scaler["weight_imp"]["std"]
    df["bmi_z"] = (df["bmi_raw_imp"] - scaler["bmi_raw_imp"]["mean"]) / scaler["bmi_raw_imp"]["std"]

    print("BMI std (TRAIN):", scaler["bmi_raw_imp"]["std"])
    preview_frame(
        df[
            [
                "ecg_id",
                "age_z",
                "sex01",
                "height_z",
                "weight_z",
                "bmi_z",
                "mask__age",
                "mask__sex",
                "mask__height",
                "mask__weight",
                "mask__bmi",
            ]
        ]
    )

    df["mask__miss_height"] = df["mask__height"].astype(int)
    df["mask__miss_weight"] = df["mask__weight"].astype(int)
    df["mask__miss_bmi"] = df["mask__bmi"].astype(int)

    assert len(META_FEATURES) == len(MASK_FEATURES)

    x_meta = df[META_FEATURES].to_numpy(dtype=np.float32)
    m_meta = df[MASK_FEATURES].to_numpy(dtype=np.float32)

    print("meta_dim =", len(META_FEATURES))
    print("X_meta shape:", x_meta.shape)
    print("M_meta shape:", m_meta.shape)

    meta_core_cols = ["ecg_id", "patient_id", "strat_fold"] + META_FEATURES + MASK_FEATURES + [
        "meta_present_any",
        "meta_present_strict",
    ]
    meta_core = df[meta_core_cols].copy()
    meta_core_out = OUT_DIR / "meta_core.parquet"
    meta_core.to_parquet(meta_core_out, index=False)
    print("meta_core.parquet ->", meta_core_out)

    idx = df[["ecg_id", "patient_id", "strat_fold", "filename_hr"]].copy()
    idx["hea_path"] = idx["filename_hr"].apply(lambda p: str((DATA / p).with_suffix(".hea")))

    mm = idx.merge(meta_core, on=["ecg_id", "patient_id", "strat_fold"], how="left")
    mm_out = OUT_DIR / "index_mm_core.parquet"
    mm.to_parquet(mm_out, index=False)

    print("index_mm_core.parquet ->", mm_out, "shape =", mm.shape)
    preview_frame(mm)

    sample_paths = mm["hea_path"].head(5).tolist()
    print("hea_path samples:", sample_paths)
    for path_str in sample_paths:
        print(path_str, "->", Path(path_str).exists())

    print("\nSplit counts (from mm):")
    print(mm["strat_fold"].value_counts().sort_index().head(12))

    cols_rate = [
        "mask__age",
        "mask__sex",
        "mask__height",
        "mask__weight",
        "mask__bmi",
        "meta_present_any",
        "meta_present_strict",
    ]
    print("\nRates sanity:")
    print(mm[cols_rate].mean())

    mm = pd.read_parquet(mm_out)
    print("Any NaN in META_FEATURES?:", mm[META_FEATURES].isna().any().any())
    print("Any NaN in MASK_FEATURES?:", mm[MASK_FEATURES].isna().any().any())
    print("Unique mask values:", {c: sorted(mm[c].unique().tolist()) for c in MASK_FEATURES})

    check = (
        (mm["miss__height"] == 1).astype(int).eq((mm["mask__height"] == 0).astype(int)).mean(),
        (mm["miss__weight"] == 1).astype(int).eq((mm["mask__weight"] == 0).astype(int)).mean(),
        (mm["miss__bmi"] == 1).astype(int).eq((mm["mask__bmi"] == 0).astype(int)).mean(),
    )
    print("Consistency miss__* vs mask__* (should be ~1.0):", check)

    print("\nDone.")
    print(f"Validation split size check: val={int(val_mask.sum())}, test={int(test_mask.sum())}")


if __name__ == "__main__":
    main()
