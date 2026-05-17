# Strict CPU Reproducibility

This directory turns the repository into a checksum-verified, CPU-only
reproduction package for:
- the initial 30-run training sweep;
- the five complementary retraining runs.

## Published inputs

- Archived working index:
  [archived_index/index_complete.parquet](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/reproducibility/archived_index/index_complete.parquet)
- Initial code snapshot:
  [published_snapshots/initial](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/published_snapshots/initial)
- Extended code snapshot:
  [published_snapshots/extended](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/published_snapshots/extended)
- Initial reference outputs:
  [results/seed_json](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/results/seed_json)
- Extended reference outputs:
  [results/extended_json](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/results/extended_json)

## Verification

Check that the published snapshots and archived index match their manifests:

```bash
python reproducibility/verify_reproducibility.py --mode both --verify-published
```

Compare newly generated runs against the published JSON outputs:

```bash
python reproducibility/verify_reproducibility.py \
  --mode initial \
  --candidate-dir /path/to/runs_initial

python reproducibility/verify_reproducibility.py \
  --mode extended \
  --candidate-dir /path/to/runs_extended
```

The output comparison normalizes the JSON payloads before hashing and ignores
the volatile `metadata.timestamp` field emitted by the training scripts.

## Re-run training

The wrapper below always uses:
- the published snapshots, not the live mutable scripts;
- the archived `index_complete.parquet`;
- a CPU-only, thread-pinned environment.

```bash
python reproducibility/reproduce_training.py \
  --mode initial \
  --data-root /path/to/ptb-xl/1.0.3 \
  --verify

python reproducibility/reproduce_training.py \
  --mode extended \
  --data-root /path/to/ptb-xl/1.0.3 \
  --verify
```

## Docker

Build the frozen CPU image:

```bash
docker build -f reproducibility/Dockerfile.cpu -t eznx-atlas-a:cpu-repro .
```

Run the initial training inside Docker:

```bash
docker run --rm \
  -v /path/to/ptb-xl/1.0.3:/data/ptb-xl:ro \
  -v /path/to/output:/workspace/repro_outputs \
  eznx-atlas-a:cpu-repro \
  python reproducibility/reproduce_training.py \
    --mode initial \
    --data-root /data/ptb-xl \
    --initial-output-dir /workspace/repro_outputs/initial \
    --verify
```

Run the extended retraining inside Docker:

```bash
docker run --rm \
  -v /path/to/ptb-xl/1.0.3:/data/ptb-xl:ro \
  -v /path/to/output:/workspace/repro_outputs \
  eznx-atlas-a:cpu-repro \
  python reproducibility/reproduce_training.py \
    --mode extended \
    --data-root /data/ptb-xl \
    --extended-output-dir /workspace/repro_outputs/extended \
    --verify
```
