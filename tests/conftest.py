"""Independent upstream structures and force constants for GPU validation."""

from pathlib import Path

import phono3py
import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def si():
    return phono3py.load(
        DATA / "phono3py_si_pbesol.yaml",
        forces_fc3_filename=DATA / "FORCES_FC3_si_pbesol",
        make_r0_average=True,
        log_level=0,
        lang="C",
    )


@pytest.fixture(scope="session")
def nacl():
    return phono3py.load(
        DATA / "phono3py_params_NaCl222.yaml.xz", make_r0_average=True, log_level=0
    )


@pytest.fixture(scope="session")
def aln():
    # phono3py f0a2fdd0 test fixture: wurtzite, NAC, 3x3x2 supercell.
    return phono3py.load(
        DATA / "phono3py_params_AlN332.yaml.xz", make_r0_average=True, log_level=0
    )


@pytest.fixture(scope="session")
def si_full():
    return phono3py.load(
        DATA / "phono3py_si_pbesol.yaml",
        forces_fc3_filename=DATA / "FORCES_FC3_si_pbesol",
        is_compact_fc=False,
        make_r0_average=True,
        log_level=0,
    )
