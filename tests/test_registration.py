"""Importing the plugin must not acquire a CUDA context."""

import os
import subprocess
import sys


def test_registration_without_gpu():
    code = """
import sys
from phono3py.conductivity.factory import _REGISTRY
from phono3py.phonon3.interaction import Interaction
import phono3py_gpu
from phono3py_gpu.initialization import all_shortest, reference_all_shortest
assert {"gpu-rta", "gpu-lbte", "wte-gpu-rta", "wte-gpu-lbte"} <= _REGISTRY.keys()
phono3py_gpu.register()
phono3py_gpu.register()
assert Interaction._get_all_shortest is all_shortest
assert reference_all_shortest is not all_shortest
assert "cupy" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
    )
