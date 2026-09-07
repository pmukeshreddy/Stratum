"""Bounded machine metadata without credentials or arbitrary environment dumps."""

import os
import platform
import shutil
import subprocess
from functools import lru_cache

import psutil


@lru_cache(maxsize=1)
def metadata():
    versions = {}
    for name in ("git", "clang", "gcc", "rustc", "go", "node", "nvcc", "nvidia-smi"):
        path = shutil.which(name)
        if not path:
            continue
        args = [path, "version"] if name == "go" else [path, "--version"]
        if name == "nvidia-smi":
            args = [path, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        try:
            result = subprocess.run(args, capture_output=True, timeout=2, check=False)
            versions[name] = (result.stdout + result.stderr).decode(errors="replace")[:500]
        except (OSError, subprocess.TimeoutExpired):
            versions[name] = "version query failed"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_logical": os.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "versions": versions,
        "environment": {
            k: os.environ[k]
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
            if k in os.environ
        },
    }
