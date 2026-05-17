import hashlib
import pathlib

root = pathlib.Path(r"C:\Users\hp\Documents\Playground\ezyx-atlas-a_gihub\mdpi_mathematics_submission_package\MDPI_template_ACS")
checksums_file = root / "CHECKSUMS.sha256"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

ok = 0
fail = 0
for line in checksums_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    expected, rel = line.split("  ", 1)
    p = root / rel
    if not p.exists():
        print(f"MISSING: {rel}")
        fail += 1
        continue
    actual = sha256(p)
    if actual != expected:
        print(f"FAIL: {rel}")
        fail += 1
    else:
        ok += 1

print(f"{ok} OK, {fail} FAILED")
