"""Importing the plugin must not acquire a CUDA context."""

import os
import subprocess
import sys


def test_registration_without_gpu():
    code = """
import sys
from phono3py.conductivity.factory import _REGISTRY
assert {"gpu-rta", "gpu-lbte", "wte-gpu-rta", "wte-gpu-lbte"} <= _REGISTRY.keys()
assert "cupy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
