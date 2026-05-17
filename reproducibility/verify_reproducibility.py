from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PROJECT_ROOT / "reproducibility" / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(mode: str) -> dict:
    manifest_path = MANIFEST_DIR / f"{mode}_inputs.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_inputs(manifest: dict) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    snapshot_dir = PROJECT_ROOT / manifest["snapshot_dir"]
    index_path = PROJECT_ROOT / manifest["index_path"]

    if not snapshot_dir.exists():
        return False, [f"Missing snapshot directory: {snapshot_dir}"]
    if not index_path.exists():
        return False, [f"Missing archived index: {index_path}"]

    actual_index = sha256_file(index_path)
    if actual_index != manifest["index_sha256"]:
        ok = False
        messages.append(
            f"Index checksum mismatch: expected {manifest['index_sha256']}, got {actual_index}"
        )
    else:
        messages.append(f"OK index checksum: {index_path}")

    for rel_name, expected_sha in sorted(manifest["snapshot_files"].items()):
        path = snapshot_dir / rel_name
        if not path.exists():
            ok = False
            messages.append(f"Missing snapshot file: {path}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            ok = False
            messages.append(
                f"Checksum mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        else:
            messages.append(f"OK snapshot checksum: {path}")

    return ok, messages


def collect_reference_jsons(reference_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(reference_dir.rglob("*.json")):
        files[path.name] = path
    return files


def load_normalized_json(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    normalized = deepcopy(payload)
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("timestamp", None)

    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_candidate_outputs(manifest: dict, candidate_dir: Path) -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    reference_dir = PROJECT_ROOT / manifest["reference_outputs_dir"]
    if not reference_dir.exists():
        return False, [f"Missing reference outputs directory: {reference_dir}"]
    if not candidate_dir.exists():
        return False, [f"Missing candidate outputs directory: {candidate_dir}"]

    reference_files = collect_reference_jsons(reference_dir)
    candidate_files = collect_reference_jsons(candidate_dir)

    if len(reference_files) != manifest["expected_output_count"]:
        ok = False
        messages.append(
            f"Reference output count mismatch for {reference_dir}: "
            f"expected {manifest['expected_output_count']}, got {len(reference_files)}"
        )
    else:
        messages.append(
            f"OK reference output count: {reference_dir} ({len(reference_files)} files)"
        )

    for name, reference_path in reference_files.items():
        candidate_path = candidate_files.get(name)
        if candidate_path is None:
            ok = False
            messages.append(f"Missing candidate output: {name}")
            continue
        reference_sha = sha256_text(load_normalized_json(reference_path))
        candidate_sha = sha256_text(load_normalized_json(candidate_path))
        if reference_sha != candidate_sha:
            ok = False
            messages.append(
                f"Normalized output checksum mismatch for {name}: "
                f"reference {reference_sha}, candidate {candidate_sha}"
            )
        else:
            messages.append(f"OK normalized output checksum: {name}")

    extra_files = sorted(set(candidate_files) - set(reference_files))
    if extra_files:
        messages.append(
            "Extra candidate JSON files ignored: " + ", ".join(extra_files)
        )

    return ok, messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify frozen reproducibility inputs and compare regenerated outputs."
    )
    parser.add_argument(
        "--mode",
        choices=["initial", "extended", "both"],
        default="both",
        help="Which published reproduction surface to verify.",
    )
    parser.add_argument(
        "--verify-published",
        action="store_true",
        help="Verify published snapshots and archived index against the manifests.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help="Directory containing regenerated JSON outputs to compare with the published reference set.",
    )
    return parser.parse_args()


def run_mode(mode: str, verify_published: bool, candidate_dir: Path | None) -> bool:
    manifest = load_manifest(mode)
    ok = True

    print(f"[{mode}]")
    if verify_published:
        inputs_ok, messages = verify_inputs(manifest)
        ok = ok and inputs_ok
        for message in messages:
            print(f"  {message}")

    if candidate_dir is not None:
        outputs_ok, messages = verify_candidate_outputs(manifest, candidate_dir)
        ok = ok and outputs_ok
        for message in messages:
            print(f"  {message}")

    if not verify_published and candidate_dir is None:
        print("  Nothing to verify. Use --verify-published and/or --candidate-dir.")
        ok = False

    return ok


def main() -> int:
    args = parse_args()
    modes = ["initial", "extended"] if args.mode == "both" else [args.mode]
    overall_ok = True

    for mode in modes:
        overall_ok = run_mode(mode, args.verify_published, args.candidate_dir) and overall_ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
