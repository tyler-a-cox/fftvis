"""Run the GPU engine against the CPU engine with cupy/cufinufft shimmed out.

The real acceptance test for the GPU backend is ``test_cpu_vs_gpu.py``, but it
skips without a CUDA device. This runs the same comparisons with cupy and
cufinufft replaced by numpy and finufft, which covers the port's *logic* --
axis flips, einsums, reshapes, accumulation indexing -- on CPU-only CI.

The shim replaces ``sys.modules`` entries, so it runs in a subprocess.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "_gpu_shim_check.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="shim script missing")
def test_gpu_engine_matches_cpu_engine_under_shim():
    """Every GPU code path reproduces the CPU engine to NUFFT accuracy."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(
            "GPU shim comparison failed:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    assert "ALL PASS" in proc.stdout, proc.stdout
