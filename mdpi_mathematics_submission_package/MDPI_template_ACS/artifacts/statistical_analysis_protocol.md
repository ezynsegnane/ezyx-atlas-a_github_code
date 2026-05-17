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

## Post-hoc power analysis (cited in §5.4)

Power was estimated via the Pitman-efficiency normal approximation for
paired Wilcoxon signed-rank tests: Wilcoxon power ≥ (3/π) × t-test power ≈
0.955 × t-test power (for large-sample normal-score approximation). For n = 10
paired differences, the t-test critical value is t_{0.025, 9} = 2.262 and
the non-centrality parameter is δ = d_z × √n.

| Contrast                     | Cohen d_z | δ = d_z × √10 | Power_t (normal approx) | Wilcoxon LB (≥ 3/π × Power_t) |
|------------------------------|-----------|---------------|-------------------------|-------------------------------|
| demo+anthro vs none (primary)| 1.77      | 5.596         | 0.9999                  | ≥ 0.9548                      |
| demo+anthro vs demo (secondary)| 1.27    | 4.031         | 0.9808                  | ≥ 0.9366                      |
| demo vs none (null-like)     | 0.30      | 0.944         | 0.1567                  | ≥ 0.1496                      |

Values are computed by `compute_post_hoc_power()` in `analyze_multiseed_results.py`
and stored under `power_analysis` in `statistical_analysis_full.json`.

The Wilcoxon column reports the **Pitman lower bound** (3/π ≈ 0.9549 times the
paired t-test power), not the exact Wilcoxon power. For the primary contrast
(d_z = 1.77, n = 10) the actual Wilcoxon power is expected to substantially
exceed 0.9548: at this effect size, nearly all 10 paired differences are positive,
and the Wilcoxon test rejects with probability > 0.99 in practice. The lower
bound of ≥ 0.9548 is therefore conservative. The secondary and null-like
contrasts follow the same pattern but with smaller effects. Effect sizes (d_z)
are drawn from the archived `statistical_analysis_full.json` (macro-AUC column
of the primary paired statistics table).

## Extended sensitivity experiments (§7.8, cited in Limitation 3)

Five additional single-seed runs were executed after the main 30-run release
to address reviewer requests H5, H7, H8, and M3. Their raw JSON results are
archived in `artifacts/extended_runs/`. These runs are deliberately single-seed
exploratory experiments and carry no confirmatory statistical weight.
