# Statistical Analysis Protocol Note

This note documents the analysis family that was executed for the archived
release bundled with this package. It is a package-local documentation artifact,
not a preregistration or an external time-stamped analysis plan.

## Experimental units

- Variants: `none`, `demo`, `demo+anthro`
- Random seeds: `10`
- Pairwise contrasts: `3`
  - `demo - none`
  - `demo+anthro - demo`
  - `demo+anthro - none`

## Primary archived metrics

- Macro-AUC
- Macro-`F1*` at validation-selected class thresholds
- Per-class AUC for `NORM`, `MI`, `STTC`, `CD`, `HYP`
- Per-class `F1*` for `NORM`, `MI`, `STTC`, `CD`, `HYP`

## Statistical family

The manuscript-wide confirmatory family contains `36` tests:

- `3` variant contrasts
- multiplied by:
  - `1` macro-AUC
  - `1` macro-`F1*`
  - `5` per-class AUC values
  - `5` per-class `F1*` values

## Paired inference

- Exact two-sided paired Wilcoxon signed-rank tests on seed-matched contrasts.
- Seed-level percentile bootstrap confidence intervals (`10,000` resamples).
- Effect sizes:
  - Cohen's `d_z`
  - Hedges-corrected `g_z`
- Multiplicity control:
  - Benjamini-Hochberg FDR at `q = 0.05`

## Selection dependencies

- Blend weights are selected on validation fold 9.
- Class-specific `F1` thresholds are selected on validation fold 9.
- Near-tied checkpoints are broken toward larger validation `Delta_meta`.

All inferential statements in the manuscript should therefore be read
conditionally on this validation-based model-selection protocol.
