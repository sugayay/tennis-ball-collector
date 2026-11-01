#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export QD_PERFDISPATCH_FORCE=func_solve_body:func_solve_body_monolith
export YOLO_CONFIG_DIR="$task_dir/.runtime/yolo-config"
export YOLO_AUTOINSTALL=false
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
exec "${PHYSICAL_AI_PYTHON:-python3}" -B "$task_dir/tools/run_cases.py" "$@"
