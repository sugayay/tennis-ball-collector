"""Run selected frozen cases sequentially and preserve every verdict."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

from audit_full_basket import audit

ROOT = Path(__file__).resolve().parents[1]
CASES = sorted(p.stem for p in (ROOT / "cases").glob("*.json"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def frozen_inputs(excluded_output):
    result = {}
    pending = [ROOT]
    while pending:
        directory = pending.pop()
        for p in directory.iterdir():
            if p == excluded_output or p.name in (".runtime", "__pycache__", "runs", "evidence", "仕事") or p.is_symlink():
                continue
            if p.is_dir():
                pending.append(p)
            elif p.suffix in (".py", ".json", ".urdf", ".obj", ".pt"):
                result[p.relative_to(ROOT).as_posix()] = digest(p)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=CASES)
    group.add_argument("--all-cases", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("amdgpu", "cpu"), default="amdgpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new one.")
    expected = read(ROOT / "models/model-manifest.json")
    if not all((ROOT / "models" / name).is_file() and digest(ROOT / "models" / name) == h for name, h in expected.items()):
        parser.error("A bundled model is missing or its SHA-256 does not match.")
    selected = CASES if args.all_cases else [args.case or "02_diagonal"]
    inputs = frozen_inputs(output)
    output.mkdir(parents=True)
    plan = {"version": "v85", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cases": selected, "backend": args.backend, "video": not args.no_video,
            "source_sha256": inputs, "attempts_per_case": 1}
    write(output / "run-plan.json", plan)
    summary = {"version": "v85", "status": "running", "verified": False, "completed": [], "active": None}
    for case in selected:
        summary["active"] = case
        write(output / "suite-summary.json", summary)
        target = output / case
        cmd = [sys.executable, "-B", str(ROOT / "tools/offline_runtime.py"), "--source", str(ROOT),
               "--case", str(ROOT / "cases" / f"{case}.json"), "--output", str(target), "--backend", args.backend]
        if not args.no_video:
            cmd.append("--video")
        started = time.monotonic()
        print("START " + case, flush=True)
        row = {"case": case, "verified": False, "held": None, "total": 4}
        try:
            with (output / f"{case}-process.log").open("x", encoding="utf-8") as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            row["returncode"] = process.returncode
            native = read(target / "mission_check.json")
            if "simulated_seconds" not in native:
                raise RuntimeError("Simulation did not complete.")
            independent = audit(target)
            write(target / "full-basket-independent-audit.json", independent)
            row.update(held=len({i for x in independent["final"].values() for i in x["ids"]}),
                       simulated_seconds=native["simulated_seconds"],
                       native_verified=native["verified"], independent_verified=independent["verified"],
                       failed_checks=[k for k, v in native["checks"].items() if not v],
                       failed_independent_checks=[k for k, v in independent["checks"].items() if not v])
            row["verified"] = process.returncode == 0 and native["verified"] and independent["verified"]
        except Exception as error:
            row["execution_error"] = type(error).__name__ + ": " + str(error)
        row.update(wall_seconds=time.monotonic() - started, source_unchanged=inputs == frozen_inputs(output))
        row["verified"] = row["verified"] and row["source_unchanged"]
        summary["completed"].append(row)
        write(output / "suite-summary.json", summary)
        print(json.dumps(row), flush=True)
    summary.update(status="complete", active=None, verified=all(x["verified"] for x in summary["completed"]))
    write(output / "suite-summary.json", summary)
    return 0 if summary["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
