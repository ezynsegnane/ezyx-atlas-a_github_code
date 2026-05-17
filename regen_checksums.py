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

entries = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p.name != "CHECKSUMS.sha256":
        rel = p.relative_to(root).as_posix()
        entries.append(f"{sha256(p)}  {rel}")

checksums_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
print(f"Written {len(entries)} entries to CHECKSUMS.sha256")
