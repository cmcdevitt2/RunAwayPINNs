"""Small host-side 0D bulk-plasma subsystem for the coupled FV driver.

This module owns the state-resolved collisional-radiative, bulk-energy, Ohm-law,
and lumped-induction equations from the LaTeX reference.  It is deliberately
independent of Warp/cuDSS: the kinetic driver can provide its current moment,
advance its GPU state, and use the returned stage plasma state to rebuild the
kinetic coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import scipy.constants as const
from scipy.optimize import root

from openadas_data import OpenADASBundle, OpenADASError


@dataclass(frozen=True)
class BulkSpecies:
    """One charge-state species and its atomic-energy reference data."""

    name: str
    adas_element: str
    atomic_number: int
    total_density_m3: float
    ionization_energies_eV: tuple[float, ...]
    initial_charge: int = 0
    excitation_energies_eV: tuple[float, ...] | None = None
    screening_lengths: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.atomic_number < 1:
            raise ValueError("atomic_number must be positive")
        if self.total_density_m3 <= 0.0 or not math.isfinite(self.total_density_m3):
            raise ValueError("total_density_m3 must be finite and positive")
        if len(self.ionization_energies_eV) != self.atomic_number:
            raise ValueError("one positive ionization energy is required per charge transition")
        if any(e <= 0.0 or not math.isfinite(e) for e in self.ionization_energies_eV):
            raise ValueError("ionization energies must be finite and positive")
        if not 0 <= self.initial_charge <= self.atomic_number:
            raise ValueError("initial_charge must lie in [0, atomic_number]")
        if (self.excitation_energies_eV is None) != (self.screening_lengths is None):
            raise ValueError("excitation energies and screening lengths must be supplied together")
        if self.excitation_energies_eV is not None:
            if len(self.excitation_energies_eV) != self.atomic_number + 1:
                raise ValueError("one excitation energy is required per charge state")
            if len(self.screening_lengths or ()) != self.atomic_number + 1:
                raise ValueError("one screening length is required per charge state")
            if any(e < 0.0 or not math.isfinite(e) for e in self.excitation_energies_eV):
                raise ValueError("excitation energies must be finite and non-negative")
            if any(a < 0.0 or not math.isfinite(a) for a in self.screening_lengths or ()):
                raise ValueError("screening lengths must be finite and non-negative")

    @property
    def cumulative_ionization_energies_eV(self) -> np.ndarray:
        return np.concatenate(([0.0], np.cumsum(self.ionization_energies_eV)))


@dataclass(frozen=True)
class BulkState:
    """0D bulk state; populations are ordered q=0,...,Z for each species."""

    temperature_eV: float
    populations_m3: tuple[np.ndarray, ...]
    current_A: float


@dataclass(frozen=True)
class BulkDiagnostics:
    electron_density_m3: float
    z_eff: float
    radiated_power_W_m3: float
    atomic_energy_J_m3: float
    electron_energy_J_m3: float
    total_energy_J_m3: float
    resistivity_ohm_m: float
    electric_field_V_m: float
    plasma_current_A_m2: float
    runaway_current_A_m2: float
    ohmic_power_W_m3: float


@dataclass(frozen=True)
class BulkDerivative:
    temperature_eV_s: float
    populations_m3_s: tuple[np.ndarray, ...]
    current_A_s: float
    diagnostics: BulkDiagnostics


class BulkPlasmaModel:
    """State-resolved 0D CR/energy/induction model.

    ``kinetic_current_A`` is supplied by the kinetic solver for each stage.  A
    bulk-only step therefore freezes that moment over one attempted step; the
    eventual coupled driver can iterate the kinetic and bulk stage values.
    """

    def __init__(
        self,
        species: Sequence[BulkSpecies],
        bundles: Mapping[str, OpenADASBundle],
        *,
        area_m2: float,
        major_radius_m: float,
        inductance_H: float,
        applied_voltage_V: float = 0.0,
    ) -> None:
        self.species = tuple(species)
        self.bundles = dict(bundles)
        if not self.species:
            raise ValueError("at least one bulk species is required")
        if area_m2 <= 0.0 or major_radius_m <= 0.0 or inductance_H <= 0.0:
            raise ValueError("area, major radius, and inductance must be positive")
        if not math.isfinite(applied_voltage_V):
            raise ValueError("applied_voltage_V must be finite")
        self.area_m2 = float(area_m2)
        self.major_radius_m = float(major_radius_m)
        self.inductance_H = float(inductance_H)
        self.applied_voltage_V = float(applied_voltage_V)
        for item in self.species:
            bundle = self.bundles.get(item.name)
            if bundle is None:
                raise ValueError(f"missing OpenADAS bundle for species {item.name!r}")
            if bundle.atomic_number != item.atomic_number:
                raise ValueError(f"OpenADAS Z mismatch for species {item.name!r}")

    def initial_state(self, temperature_eV: float, current_A: float = 0.0) -> BulkState:
        if temperature_eV <= 0.0 or not math.isfinite(temperature_eV):
            raise ValueError("temperature_eV must be finite and positive")
        populations = []
        for item in self.species:
            pop = np.zeros(item.atomic_number + 1, dtype=np.float64)
            pop[item.initial_charge] = item.total_density_m3
            populations.append(pop)
        return BulkState(float(temperature_eV), tuple(populations), float(current_A))

    def _validate_state(self, state: BulkState, *, nonnegative: bool = True) -> None:
        if state.temperature_eV <= 0.0 or not math.isfinite(state.temperature_eV):
            raise ValueError("bulk temperature must be finite and positive")
        if len(state.populations_m3) != len(self.species):
            raise ValueError("bulk population/species count mismatch")
        for item, pop in zip(self.species, state.populations_m3):
            values = np.asarray(pop, dtype=np.float64)
            if values.shape != (item.atomic_number + 1,) or not np.all(np.isfinite(values)):
                raise ValueError(f"invalid charge populations for {item.name!r}")
            if nonnegative and np.any(values < 0.0):
                raise ValueError(f"negative charge population for {item.name!r}")
            if not math.isclose(float(values.sum()), item.total_density_m3,
                                rel_tol=2.0e-12, abs_tol=1.0e6):
                raise ValueError(f"charge populations do not conserve nuclei for {item.name!r}")
        if not math.isfinite(state.current_A):
            raise ValueError("bulk current must be finite")

    def plasma_moments(self, state: BulkState) -> tuple[float, float]:
        self._validate_state(state)
        ne = sum(float(np.dot(np.arange(item.atomic_number + 1), pop))
                 for item, pop in zip(self.species, state.populations_m3))
        if ne <= 0.0 or not math.isfinite(ne):
            raise ValueError("charge-state populations produce no free electrons")
        charge_square = sum(float(np.dot(np.arange(item.atomic_number + 1)**2, pop))
                            for item, pop in zip(self.species, state.populations_m3))
        return ne, charge_square/ne

    def screening_inputs(self, state: BulkState) -> dict[str, tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]]:
        """Return state-resolved screening inputs for kinetic coefficient assembly.

        Species without explicit atomic screening tables are omitted.  The
        caller must reject that omission before enabling partial screening;
        this method never substitutes ionization energies or a mean charge.
        """
        self._validate_state(state)
        result = {}
        for item, population in zip(self.species, state.populations_m3):
            if item.excitation_energies_eV is None:
                continue
            assert item.screening_lengths is not None
            result[item.name] = (
                np.asarray(population, dtype=np.float64).copy(),
                item.excitation_energies_eV,
                item.screening_lengths,
            )
        return result

    def _charge_derivative(self, item: BulkSpecies, bundle: OpenADASBundle,
                           state: BulkState, population: np.ndarray,
                           electron_density_m3: float) -> np.ndarray:
        scd = bundle.ionization_rates(state.temperature_eV, electron_density_m3)
        acd = bundle.recombination_rates(state.temperature_eV, electron_density_m3)
        ne_cm3 = electron_density_m3*1.0e-6
        derivative = np.zeros_like(population)
        for q in range(item.atomic_number):
            ionization = ne_cm3*scd[q]*population[q]
            derivative[q] -= ionization
            derivative[q+1] += ionization
        for q in range(1, item.atomic_number + 1):
            recombination = ne_cm3*acd[q]*population[q]
            derivative[q] -= recombination
            derivative[q-1] += recombination
        return derivative

    def diagnostics(self, state: BulkState, kinetic_current_A: float = 0.0) -> BulkDiagnostics:
        self._validate_state(state)
        ne, z_eff = self.plasma_moments(state)
        ln_lambda = 14.9 - 0.5*math.log(ne/1.0e20) + math.log(state.temperature_eV/1.0e3)
        if ln_lambda <= 0.0:
            raise ValueError("thermal Coulomb logarithm is non-positive")
        radiation = 0.0
        atomic_energy = 0.0
        for item, population in zip(self.species, state.populations_m3):
            bundle = self.bundles[item.name]
            # ADF11 block Z1=q+1 represents the originating q charge state.
            radiation += bundle.radiated_power_density(
                state.temperature_eV, ne,
                {q+1: float(population[q]) for q in range(item.atomic_number)},
            )
            atomic_energy += const.e*float(np.dot(
                item.cumulative_ionization_energies_eV, population
            ))
        electron_energy = 1.5*ne*state.temperature_eV*const.e
        resistivity = 1.65e-9*z_eff*ln_lambda/(state.temperature_eV/1.0e3)**1.5
        plasma_current_density = state.current_A/self.area_m2
        runaway_current_density = float(kinetic_current_A)/self.area_m2
        electric_field = resistivity*(plasma_current_density-runaway_current_density)
        return BulkDiagnostics(
            electron_density_m3=ne, z_eff=z_eff,
            radiated_power_W_m3=radiation,
            atomic_energy_J_m3=atomic_energy,
            electron_energy_J_m3=electron_energy,
            total_energy_J_m3=atomic_energy+electron_energy,
            resistivity_ohm_m=resistivity,
            electric_field_V_m=electric_field,
            plasma_current_A_m2=plasma_current_density,
            runaway_current_A_m2=runaway_current_density,
            ohmic_power_W_m3=electric_field*plasma_current_density,
        )

    def rhs(self, state: BulkState, kinetic_current_A: float = 0.0) -> BulkDerivative:
        self._validate_state(state, nonnegative=False)
        # The state used by a physical RHS must still have positive free-electron
        # density; intermediate Newton iterates are handled by the TR--BDF2 wrapper.
        ne, _ = self.plasma_moments(state)
        derivatives = tuple(
            self._charge_derivative(item, self.bundles[item.name], state, population, ne)
            for item, population in zip(self.species, state.populations_m3)
        )
        diagnostics = self.diagnostics(state, kinetic_current_A)
        dne_dt = sum(float(np.dot(np.arange(item.atomic_number + 1), dp))
                     for item, dp in zip(self.species, derivatives))
        d_atomic_energy_dt = sum(const.e*float(np.dot(
            item.cumulative_ionization_energies_eV, dp
        )) for item, dp in zip(self.species, derivatives))
        total_power = diagnostics.ohmic_power_W_m3 - diagnostics.radiated_power_W_m3
        dtemperature = (
            total_power - d_atomic_energy_dt
            - 1.5*const.e*state.temperature_eV*dne_dt
        )/(1.5*const.e*ne)
        plasma_resistance = 2.0*math.pi*self.major_radius_m*diagnostics.resistivity_ohm_m/self.area_m2
        runaway_current = float(kinetic_current_A)
        dcurrent = (
            self.applied_voltage_V - plasma_resistance*(state.current_A-runaway_current)
        )/self.inductance_H
        return BulkDerivative(dtemperature, derivatives, dcurrent, diagnostics)

    def pack(self, state: BulkState) -> np.ndarray:
        self._validate_state(state)
        return np.concatenate((
            np.asarray([state.temperature_eV], dtype=np.float64),
            *(np.asarray(pop, dtype=np.float64) for pop in state.populations_m3),
            np.asarray([state.current_A], dtype=np.float64),
        ))

    def unpack(self, vector: np.ndarray) -> BulkState:
        vector = np.asarray(vector, dtype=np.float64)
        expected = 2 + sum(item.atomic_number + 1 for item in self.species)
        if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
            raise ValueError("invalid packed bulk state")
        offset = 1
        populations = []
        for item in self.species:
            size = item.atomic_number + 1
            populations.append(np.ascontiguousarray(vector[offset:offset+size]))
            offset += size
        return BulkState(float(vector[0]), tuple(populations), float(vector[offset]))

    def _rhs_vector(self, vector: np.ndarray, kinetic_current_A: float) -> np.ndarray:
        derivative = self.rhs(self.unpack(vector), kinetic_current_A)
        return np.concatenate((
            np.asarray([derivative.temperature_eV_s]),
            *(np.asarray(pop, dtype=np.float64) for pop in derivative.populations_m3_s),
            np.asarray([derivative.current_A_s]),
        ))

    def step_trbdf2(
        self, state: BulkState, dt_s: float, kinetic_current_A: float = 0.0,
        *, root_tolerance: float = 1.0e-10,
    ) -> BulkState:
        """Advance the 0D subsystem by one implicit TR--BDF2 step.

        The kinetic current is frozen over this bulk step. The coupled driver
        is responsible for stage iteration when the kinetic current changes
        materially during the same attempted step.
        """
        if dt_s <= 0.0 or not math.isfinite(dt_s):
            raise ValueError("dt_s must be finite and positive")
        _, final = self.solve_trbdf2_stages(
            state, dt_s,
            kinetic_current_n_A=kinetic_current_A,
            kinetic_current_gamma_A=kinetic_current_A,
            kinetic_current_one_A=kinetic_current_A,
            root_tolerance=root_tolerance,
        )
        return final

    def solve_trbdf2_stages(
        self, state: BulkState, dt_s: float, *,
        kinetic_current_n_A: float = 0.0,
        kinetic_current_gamma_A: float | None = None,
        kinetic_current_one_A: float | None = None,
        root_tolerance: float = 1.0e-10,
    ) -> tuple[BulkState, BulkState]:
        """Solve bulk TR--BDF2 stages for supplied kinetic current moments.

        The three current moments are stage inputs from the kinetic solver. A
        coupled driver iterates them with the returned bulk stages until its
        stage residuals meet the configured coupling tolerance.
        """
        if dt_s <= 0.0 or not math.isfinite(dt_s):
            raise ValueError("dt_s must be finite and positive")
        self._validate_state(state)
        if kinetic_current_gamma_A is None:
            kinetic_current_gamma_A = kinetic_current_n_A
        if kinetic_current_one_A is None:
            kinetic_current_one_A = kinetic_current_gamma_A
        currents = (kinetic_current_n_A, kinetic_current_gamma_A, kinetic_current_one_A)
        if any(not math.isfinite(float(current)) for current in currents):
            raise ValueError("kinetic current moments must be finite")
        y_n = self.pack(state)
        f_n = self._rhs_vector(y_n, float(kinetic_current_n_A))
        d = 1.0 - 1.0/math.sqrt(2.0)
        w = math.sqrt(2.0)/4.0

        def safe_root(fun, guess: np.ndarray, label: str) -> np.ndarray:
            def residual(vector: np.ndarray) -> np.ndarray:
                try:
                    return fun(vector)
                except (OpenADASError, ValueError, FloatingPointError):
                    return np.full_like(vector, 1.0e100)
            result = root(residual, guess, method="hybr", options={"xtol": root_tolerance})
            if not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError(f"bulk TR--BDF2 {label} solve failed: {result.message}")
            return result.x

        y_gamma = safe_root(
            lambda y: y-y_n-d*dt_s*(
                f_n+self._rhs_vector(y, float(kinetic_current_gamma_A))
            ),
            y_n+ d*dt_s*f_n, "stage-1",
        )
        f_gamma = self._rhs_vector(y_gamma, float(kinetic_current_gamma_A))
        y_one = safe_root(
            lambda y: y-y_n-d*dt_s*self._rhs_vector(y, float(kinetic_current_one_A))
            -dt_s*(w*f_n+w*f_gamma),
            y_gamma, "stage-2",
        )
        stage = self.unpack(y_gamma)
        final = self.unpack(y_one)
        self._validate_state(stage)
        self._validate_state(final)
        return stage, final
