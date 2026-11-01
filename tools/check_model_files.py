"""Check expected model bytes without downloading or deserializing checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "models/model-manifest.json").read_text())
    ok = True
    for name, expected in manifest.items():
        path = ROOT / "models" / name
        status = "MISSING"
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            status = "OK" if actual == expected else "HASH_MISMATCH"
        print(f"{name}: {status}")
        ok = ok and status == "OK"
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
