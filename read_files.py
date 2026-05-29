from pathlib import Path
import base64, json, hashlib

include_ext = {".py", ".yaml", ".yml", ".toml", ".txt", ".md"}
skip_parts = {"__pycache__", ".git", ".idea"}

files = []
for p in Path(".").rglob("*"):
    if not p.is_file():
        continue
    if any(part in skip_parts for part in p.parts):
        continue
    if p.suffix not in include_ext:
        continue

    data = p.read_bytes()
    files.append({
        "path": p.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_b64": base64.b64encode(data).decode("ascii"),
    })

Path("MFRP_CODE_BUNDLE.txt").write_text(
    json.dumps(files, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"bundled {len(files)} files")