"""Kappa solvers for the Wigner transport equation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.collision_matrix_kernel import CollisionMatrixKernel
from phono3py.conductivity.grid_point_data import GridPointAggregates
from phono3py.conductivity.lbte_collision_solver import LBTECollisionResult
from phono3py.conductivity.utils import (
    log_kappa_header,
    log_kappa_row,
    log_sigma_header,
)
from phonopy.physical_units import get_physical_units

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.kappa import effective_gamma

# Threshold in THz below which two modes are considered degenerate and treated
# as a population (diagonal) term instead of a coherence (off-diagonal) term.
DEGENERATE_FREQUENCY_THRESHOLD_THZ = 1e-4


def _get_conversion_factor_WTE(volume: float) -> float:
    """Return unit conversion factor for Wigner transport equation kappa.

    Parameters
    ----------
    volume : float
        Primitive-cell volume in Angstrom^3.

    Returns
    -------
    float
        Conversion factor in W/(m*K).

    """
    u = get_physical_units()
    return (u.THz * u.Angstrom) ** 2 * u.EV * u.Hbar / (volume * u.Angstrom**3)


def pair_contributions(f, g, cv, vv, cutoff, conversion):
    """WTE mode pairs; g is a half width in THz, as in phono3py.gamma."""
    cp = cupy()
    f, g, cv, vv = (cp.asarray(x) for x in (f, g, cv, vv))
    omega = f * get_physical_units().THzToEv
    width = g * 2 * get_physical_units().THzToEv
    total = width[..., :, None] + width[..., None, :]
    delta = omega[:, None] - omega[None, :]
    denom = delta**2 + total**2 / 4
    c = cv / cp.where(omega > 0, omega, 1)
    pref = (omega[:, None] + omega[None, :]) * (c[..., :, None] + c[..., None, :]) / 4
    valid = (f[:, None] > cutoff) & (f[None, :] > cutoff) & (denom != 0)
    scalar = cp.where(valid, pref * total / 2 / cp.where(denom != 0, denom, 1), 0)
    return scalar[..., None] * vv * conversion


class GPUWignerRTAKappaSolver:
    """Kappa solver for the Wigner transport equation (WTE).

    Decomposes kappa into a population term (kappa_P_RTA) and a coherence
    term (kappa_C).  The total kappa is kappa_TOT_RTA = kappa_P_RTA + kappa_C.

    The P and C arrays are allocated in ``finalize()``.

    Parameters
    ----------
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    volume : float
        Primitive-cell volume in Angstrom^3.
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        volume: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._conversion_factor_WTE = _get_conversion_factor_WTE(volume)
        self._log_level = log_level

        self._kappa_P: NDArray[np.double] | None = None
        self._mode_kappa_P: NDArray[np.double] | None = None
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.double] | None = None
        self.velocity_operator: NDArray[np.cdouble] | None = None

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute mode kappa (P + C) and kappa from aggregated data."""
        assert aggregates.vm_by_vm is not None

        num_sigma, num_temp, num_gp, num_band = aggregates.gamma.shape
        self._mode_kappa_P = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, 6), dtype="double", order="C"
        )
        self._mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )

        self._compute_mode_kappa(aggregates)

        num_mesh_points = int(np.prod(self._kappa_settings.mesh_numbers))
        self._kappa_P = np.sum(self._mode_kappa_P, axis=(2, 3)) / num_mesh_points
        self._kappa_C = np.sum(self._mode_kappa_C, axis=(2, 3, 4)) / num_mesh_points

    def _compute_mode_kappa(self, aggregates: GridPointAggregates) -> None:
        cp = cupy()
        gamma = effective_gamma(aggregates)
        for i, gp in enumerate(self._kappa_settings.grid_points):
            f = cp.asarray(self._frequencies[gp])
            pair = pair_contributions(
                f,
                gamma[:, :, i],
                aggregates.mode_heat_capacities[:, i],
                aggregates.vm_by_vm[i],
                self._kappa_settings.cutoff_frequency,
                self._conversion_factor_WTE,
            ).real
            population = (
                cp.abs(f[:, None] - f[None, :]) < DEGENERATE_FREQUENCY_THRESHOLD_THZ
            )
            p = pair * population[..., None]
            self._mode_kappa_P[:, :, i] = host((p.sum(axis=-2) + p.sum(axis=-3)) / 2)
            self._mode_kappa_C[:, :, i] = host(pair * (~population)[..., None])

    @property
    def kappa(self) -> NDArray[np.double] | None:
        """Return total Wigner kappa (kappa_P + kappa_C)."""
        if self._kappa_P is None or self._kappa_C is None:
            return None

        return self._kappa_P + self._kappa_C

    @property
    def kappa_TOT_RTA(self) -> NDArray[np.double] | None:
        """Return total Wigner kappa (same as ``kappa``)."""
        return self.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double] | None:
        """Return population kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_P

    @property
    def mode_kappa(self) -> NDArray[np.double] | None:
        """Return population mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._mode_kappa_P

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_C

    @property
    def mode_kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_C

    # ------------------------------------------------------------------
    # Properties — per-grid-point data
    # ------------------------------------------------------------------

    def get_extra_grid_point_output(self) -> dict[str, Any]:
        """Return per-grid-point extra arrays for HDF5 output.

        Returns the full velocity operator array. The caller is responsible
        for slicing by grid-point index.

        """
        return {"velocity_operator": self.velocity_operator}

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print K_P, K_C, K_T rows for the Wigner-RTA conductivity."""
        if not self._log_level:
            return

        kappa_tot = self.kappa
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )

        for i, sigma in enumerate(self._kappa_settings.sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_P\t", t, self._kappa_P[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_C\t", t, self._kappa_C[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_T\t", t, kappa_tot[i, j], ipm, num_phonon_modes)
            print("", flush=True)

    def get_extra_kappa_output(self) -> dict[str, Any]:
        """Return Wigner-specific extra arrays keyed by HDF5 dataset name.

        Sigma-dependent values have sigma as their first axis; the caller
        slices them with ``value[i_sigma]``.  Sigma-independent values
        (e.g. velocity_operator) are written as-is for every sigma.

        """
        return {
            "kappa_TOT_RTA": self.kappa_TOT_RTA,
            "kappa_P_RTA": self._kappa_P,
            "kappa_C": self._kappa_C,
            "mode_kappa": self._mode_kappa_P,
            "mode_kappa_P_RTA": self._mode_kappa_P,
            "mode_kappa_C": self._mode_kappa_C,
            "velocity_operator": self.velocity_operator,
        }


class GPUWignerLBTEKappaSolver:
    """LBTE kappa solver with added Wigner coherence (C) term.

    Composes a CollisionMatrixKernel for the standard LBTE solve (P-term) and
    adds the coherence (C) term from the stored velocity operator outer
    products and linewidths.

    Stage 1 (per-grid-point): store() delegates collision data to the solver.

    Stage 2 (global): finalize() calls solver.solve() for the P-term kappa,
    then computes the C-term from the stored outer products and linewidths.

    Parameters
    ----------
    solver : CollisionMatrixKernel
        Shared solver for the P-term (standard LBTE).
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    volume : float
        Primitive-cell volume in Angstrom^3.
    is_reducible_collision_matrix : bool, optional
        When True the C-term is not computed (not implemented).  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixKernel,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        volume: float,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._conversion_factor_WTE = _get_conversion_factor_WTE(volume)
        self._is_reducible = is_reducible_collision_matrix
        self._log_level = log_level

        self.velocity_operator: NDArray[np.cdouble] | None = None

        # C-term output arrays.
        num_sigma = len(kappa_settings.sigmas)
        num_temp = len(kappa_settings.temperatures)
        num_ir = len(kappa_settings.grid_points)
        num_band0 = len(kappa_settings.band_indices)
        num_band = frequencies.shape[1]
        self._mode_kappa_C: NDArray[np.cdouble] | None = np.zeros(
            (num_sigma, num_temp, num_ir, num_band0, num_band, 6), dtype="complex128"
        )
        self._kappa_C: NDArray[np.double] | None = np.zeros(
            (num_sigma, num_temp, 6), dtype="double"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
    ) -> None:
        """Store collision matrix row for this grid point.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionSolver.compute().

        """
        self._solver.store(i_gp, collision_result)

    def finalize(
        self,
        aggregates: GridPointAggregates,
    ) -> None:
        """Finalize P-term via solver, then compute C-term.

        The standard LBTE kappa table is suppressed (replaced by the Wigner
        table printed at the end of this method) when log_level > 0.

        """
        if self._is_reducible:
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True"
            )
            self._mode_kappa_C = None
            self._kappa_C = None
        prev_sigma = -1
        for i_sigma, i_temp in self._solver.solve_iter(aggregates):
            self._compute_coherence_kappa_at(aggregates, i_sigma, i_temp)
            if self._log_level:
                if i_sigma != prev_sigma:
                    log_sigma_header(self._kappa_settings.sigmas[i_sigma])
                    prev_sigma = i_sigma
                self._log_wigner_kappa_at(i_sigma, i_temp)

    # ------------------------------------------------------------------
    # Properties — delegated to solver (LBTE P-term results)
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixKernel:
        """Return the underlying CollisionMatrixKernel."""
        return self._solver

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa

    # ------------------------------------------------------------------
    # Properties — Wigner-specific aliases
    # ------------------------------------------------------------------

    @property
    def kappa_P_exact(self) -> NDArray[np.double]:
        """Return LBTE kappa (P-exact term) — alias for kappa."""
        return self._solver.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return RTA kappa (P-RTA term) — alias for kappa_RTA."""
        return self._solver.kappa_RTA

    @property
    def mode_kappa_P_exact(self) -> NDArray[np.double]:
        """Return mode LBTE kappa (P-exact term) — alias for mode_kappa."""
        return self._solver.mode_kappa

    @property
    def mode_kappa_P_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa (P-RTA term) — alias for mode_kappa_RTA."""
        return self._solver.mode_kappa_RTA

    # ------------------------------------------------------------------
    # Properties — C-term
    # ------------------------------------------------------------------

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_C

    @property
    def mode_kappa_C(self) -> NDArray[np.cdouble] | None:
        """Return per-mode coherence kappa (complex)."""
        return self._mode_kappa_C

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa."""
        return self._solver.mode_kappa

    def get_extra_kappa_output(self) -> dict[str, NDArray | None]:
        """Return Wigner-specific LBTE kappa arrays keyed by HDF5 dataset name.

        Keys
        ----
        kappa_TOT_exact   : (num_sigma, num_temp, 6)
        kappa_TOT_RTA     : (num_sigma, num_temp, 6)
        kappa_P_exact     : (num_sigma, num_temp, 6)
        kappa_P_RTA       : (num_sigma, num_temp, 6)
        kappa_C           : (num_sigma, num_temp, 6) or None
        mode_kappa_P_exact : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_P_RTA  : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_C      : (num_sigma, num_temp, num_gp, num_band0, num_band, 6)
                            or None

        """
        kappa_P_exact = self._solver.kappa
        kappa_P_RTA = self._solver.kappa_RTA
        kappa_C = self._kappa_C
        if kappa_C is not None:
            kappa_TOT_exact: NDArray[np.double] | None = kappa_P_exact + kappa_C
            kappa_TOT_RTA: NDArray[np.double] | None = kappa_P_RTA + kappa_C
        else:
            kappa_TOT_exact = kappa_P_exact
            kappa_TOT_RTA = kappa_P_RTA
        mode_kappa_C = self._mode_kappa_C
        return {
            "kappa_TOT_exact": kappa_TOT_exact,
            "kappa_TOT_RTA": kappa_TOT_RTA,
            "kappa_P_exact": kappa_P_exact,
            "kappa_P_RTA": kappa_P_RTA,
            "kappa_C": kappa_C,
            "mode_kappa": self._solver.mode_kappa,
            "mode_kappa_P_exact": self._solver.mode_kappa,
            "mode_kappa_P_RTA": self._solver.mode_kappa_RTA,
            "mode_kappa_C": None if mode_kappa_C is None else mode_kappa_C.real,
            "velocity_operator": self.velocity_operator,
        }

    # ------------------------------------------------------------------
    # Private: C-term computation
    # ------------------------------------------------------------------

    def _compute_coherence_kappa_at(
        self,
        aggregates: GridPointAggregates,
        i_sigma: int,
        i_temp: int,
    ) -> None:
        cp = cupy()
        if self._mode_kappa_C is None:
            return
        total = cp.zeros(6)
        for i, gp in enumerate(self._kappa_settings.grid_points):
            f = cp.asarray(self._frequencies[gp])
            g = self._solver.get_main_diagonal(i, i_sigma, i_temp)
            pair = pair_contributions(
                f,
                g,
                aggregates.mode_heat_capacities[i_temp, i],
                aggregates.vm_by_vm[i],
                self._kappa_settings.cutoff_frequency,
                self._conversion_factor_WTE,
            )
            pair *= (
                cp.abs(f[:, None] - f[None, :]) > DEGENERATE_FREQUENCY_THRESHOLD_THZ
            )[..., None]
            self._mode_kappa_C[i_sigma, i_temp, i] = host(pair)
            total += pair.sum(axis=(0, 1)).real
        self._kappa_C[i_sigma, i_temp] = host(
            total / int(np.prod(self._kappa_settings.mesh_numbers))
        )

    def _log_wigner_kappa_at(self, i_sigma: int, i_temp: int) -> None:
        """Print Wigner LBTE kappa for one (sigma, temperature) pair."""
        t = self._kappa_settings.temperatures[i_temp]
        if t <= 0:
            return

        kappa_P = self._solver._kappa[i_sigma, i_temp]
        kappa_P_RTA = self._solver._kappa_RTA[i_sigma, i_temp]
        kappa_C = self._kappa_C

        print(
            ("#%6s       " + " %-10s" * 6)
            % ("         \t\t  T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )
        print("K_P_exact\t\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_P)))
        print(
            "(K_P_RTA)\t\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_P_RTA))
        )
        if kappa_C is not None:
            print(
                "K_C      \t\t"
                + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_C[i_sigma, i_temp]))
            )
            print(" ")
            kappa_tot = kappa_P + kappa_C[i_sigma, i_temp]
            print(
                "K_TOT=K_P_exact+K_C\t"
                + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_tot))
            )
        print("-" * 76, flush=True)
