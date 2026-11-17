"""GPU transport plugin; registration does not initialize CUDA."""

__version__ = "0.1.0"


def register():
    """Register GPU BTE and WTE calculator factories with phono3py."""
    from phono3py.conductivity.factory import _REGISTRY

    from phono3py_gpu.factory import make_lbte, make_rta

    _REGISTRY["gpu-rta"] = make_rta
    _REGISTRY["gpu-lbte"] = make_lbte
    _REGISTRY["wte-gpu-rta"] = lambda pp, config: make_rta(pp, config, wte=True)
    _REGISTRY["wte-gpu-lbte"] = lambda pp, config: make_lbte(pp, config, wte=True)
