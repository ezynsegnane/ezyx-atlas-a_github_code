# Google Colab Smoke Test

This folder provides a Colab-friendly way to check that the published
reproducibility surface works from a fresh hosted notebook.

## What Colab is good for here

- Cloning the GitHub repository from scratch.
- Installing the pinned CPU dependencies.
- Downloading PTB-XL 1.0.3 into the runtime.
- Verifying the published snapshots and archived index.
- Replaying one initial run and one complementary run, then comparing the
  regenerated JSON files to the archived references after normalization.

## What Colab is not ideal for here

- The full 30-run initial sweep plus the 5 complementary runs on the free tier.
- Long unattended CPU workloads when the notebook is idle.
- Strict Docker-equivalent guarantees.

Colab notebooks run on virtual machines that are deleted when idle and have a
maximum lifetime. Google also states that usage limits and maximum runtime
length fluctuate over time, and that the free tier commonly runs notebooks for
at most 12 hours.

Official references:
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)

## Recommended test strategy

1. Open the notebook in this folder from GitHub inside Colab.
2. Set the runtime to `CPU`.
3. Run the notebook once in `infra` mode to confirm install + published checks.
4. Run it again in `both` mode to replay one initial run and one complementary
   run.
5. If both normalized JSON comparisons pass, the hosted Colab path is working.

For the full study rerun, prefer the Docker flow in
[reproducibility/README.md](/Users/hp/Documents/Playground/ezyx-atlas-a_gihub/reproducibility/README.md).
