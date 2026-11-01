"""Unchanged frozen-source execution with an explicit, shared offline runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--backend", choices=("cpu", "amdgpu"), default="cpu")
parser.add_argument("--video", action="store_true")
args = parser.parse_args()
assert not args.output.exists(), args.output
args.output.mkdir(parents=True)
source = args.source.resolve()
entry = source / "fleet_mission.py"
assert os.environ.get("GALLIUM_DRIVER") == "llvmpipe"

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
assert torch.cuda.is_available(), "Existing ROCm inference device is required"
import genesis as gs

original_build = gs.Scene.build
original_step = gs.Scene.step
pace = {}
rate = 0.5


def build(scene, *a, **kw):
    result = original_build(scene, *a, **kw)
    assert gs.backend == getattr(gs, args.backend)
    from OpenGL import GL
    context = scene.visualizer.rasterizer._renderer
    context.make_current()
    try:
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
    finally:
        context.make_uncurrent()
    assert "llvmpipe" in renderer.lower() or "softpipe" in renderer.lower(), renderer
    metadata = {
        "physics_backend": args.backend, "precision": "32",
        "renderer": renderer, "inference_device": "cuda:0 (existing AMD ROCm)",
        "maximum_simulation_seconds_per_wall_second": rate,
        "pacing": "Constant offline clock cap. Does not inspect or wait for inference results; no real-time claim.",
        "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "seed": 13,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "genesis": getattr(gs, "__version__", "unknown"),
        "device": torch.cuda.get_device_name(0),
        "physics_substeps": json.loads((source / "vision_config.json").read_text())["physics_substeps"],
    }
    (args.output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pace[id(scene)] = time.monotonic()
    return result


def step(scene, *a, **kw):
    result = original_step(scene, *a, **kw)
    deadline = pace[id(scene)] + 1 / 120 / rate
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    pace[id(scene)] = max(deadline, time.monotonic())
    return result


gs.Scene.build = build
gs.Scene.step = step
sys.path.insert(0, str(source))
sys.argv = [str(entry), "--backend", args.backend, "--scenario", "straight",
            "--robot-variant", "lift", "--case-config", str(args.case),
            "--output-dir", str(args.output)]
if not args.video:
    sys.argv.append("--no-video")
runpy.run_path(str(entry), run_name="__main__")
