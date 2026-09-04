"""Small, dependency-light reader for unresolved OpenADAS ADF11 files.

OpenADAS files are user-supplied data and are intentionally not distributed by
this repository.  This module reads the standard unresolved ADF11 tables used
by the bulk model (ACD, SCD, PLT, and PRB), interpolates only inside the native
log-temperature/log-density rectangle, and exposes provenance for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np


_HEADER_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
_CHARGE_RE = re.compile(r"/\s*Z1\s*=\s*(\d+)")
_CLASSES = {"acd", "scd", "plt", "prb"}
_UNITS = {
    "acd": "cm^3 s^-1",
    "scd": "cm^3 s^-1",
    "plt": "W cm^3",
    "prb": "W cm^3",
}


class OpenADASError(ValueError):
    """Malformed, unsupported, or out-of-range OpenADAS input."""


def _numeric_line(line: str) -> list[float] | None:
    tokens = line.split()
    if not tokens:
        return None
    try:
        return [float(token) for token in tokens]
    except ValueError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ADF11Table:
    """One unresolved ADF11 coefficient table in native logarithmic form."""

    path: Path
    data_class: str
    atomic_number: int
    log_temperature_eV: np.ndarray
    log_density_cm3: np.ndarray
    charge_states: np.ndarray
    log_coefficients: np.ndarray
    header: str
    sha256: str

    @property
    def element(self) -> str:
        match = re.search(r"/([A-Z][A-Z ]*)\s+/", self.header)
        return match.group(1).strip() if match else self.header.strip()

    @property
    def temperature_eV_range(self) -> tuple[float, float]:
        return tuple(float(x) for x in (10.0**self.log_temperature_eV[[0, -1]]))

    @property
    def density_cm3_range(self) -> tuple[float, float]:
        return tuple(float(x) for x in (10.0**self.log_density_cm3[[0, -1]]))

    def _charge_index(self, charge_state: int) -> int:
        hits = np.flatnonzero(self.charge_states == int(charge_state))
        if hits.size != 1:
            raise OpenADASError(
                f"{self.path}: {self.data_class} has no unique charge state {charge_state}"
            )
        return int(hits[0])

    @staticmethod
    def _bracket(axis: np.ndarray, value: float, label: str) -> tuple[int, float]:
        if not np.isfinite(value) or value < axis[0] or value > axis[-1]:
            raise OpenADASError(
                f"OpenADAS {label}={value:g} is outside native range "
                f"[{axis[0]:g}, {axis[-1]:g}]; extrapolation is disabled"
            )
        hi = int(np.searchsorted(axis, value, side="right"))
        if hi == 0:
            return 0, 0.0
        if hi == axis.size:
            return axis.size - 2, 1.0
        lo = hi - 1
        fraction = (value-axis[lo])/(axis[hi]-axis[lo])
        return lo, float(fraction)

    def evaluate(self, charge_state: int, te_eV: float, ne_m3: float) -> float:
        """Return a coefficient, refusing all temperature/density extrapolation."""
        q = self._charge_index(charge_state)
        log_te = math_log10_positive(te_eV, "te_eV")
        log_ne = math_log10_positive(ne_m3*1.0e-6, "ne_m3")
        it, ft = self._bracket(self.log_temperature_eV, log_te, "log10(Te/eV)")
        id_, fd = self._bracket(self.log_density_cm3, log_ne, "log10(ne/cm^-3)")
        values = self.log_coefficients[q]
        top = (1.0-fd)*values[it, id_] + fd*values[it, id_+1]
        bottom = (1.0-fd)*values[it+1, id_] + fd*values[it+1, id_+1]
        return float(10.0**((1.0-ft)*top + ft*bottom))

    def provenance(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "class": self.data_class,
            "atomic_number": self.atomic_number,
            "element": self.element,
            "temperature_eV_range": self.temperature_eV_range,
            "density_cm3_range": self.density_cm3_range,
            "units": _UNITS[self.data_class],
        }


def math_log10_positive(value: float, label: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise OpenADASError(f"{label} must be finite and positive")
    return float(np.log10(value))


def load_adf11(path: Path, data_class: str) -> ADF11Table:
    """Parse an unresolved standard ADF11 file."""
    path = Path(path).expanduser().resolve()
    data_class = data_class.lower()
    if data_class not in _CLASSES:
        raise OpenADASError(f"unsupported ADF11 class {data_class!r}; choose {_CLASSES}")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise OpenADASError(f"cannot read OpenADAS file {path}") from exc
    if not lines:
        raise OpenADASError(f"empty OpenADAS file: {path}")
    header_match = _HEADER_RE.match(lines[0])
    if header_match is None:
        raise OpenADASError(f"unrecognized ADF11 header in {path}")
    atomic_number, n_density, n_temperature, charge_min, charge_max = (
        int(x) for x in header_match.groups()
    )
    block_starts = [i for i, line in enumerate(lines) if _CHARGE_RE.search(line)]
    if not block_starts:
        raise OpenADASError(f"no ADF11 charge-state blocks found in {path}")

    preamble: list[float] = []
    for line in lines[1:block_starts[0]]:
        values = _numeric_line(line)
        if values is not None:
            preamble.extend(values)
    expected_preamble = n_temperature+n_density
    if len(preamble) != expected_preamble:
        raise OpenADASError(
            f"{path}: expected {expected_preamble} temperature/density values, got {len(preamble)}"
        )
    # Standard unresolved ADF11 stores the density grid first, followed by
    # the temperature grid; coefficient blocks are then temperature-major
    # with density as the fastest-changing index.
    log_density = np.asarray(preamble[:n_density], dtype=np.float64)
    log_temperature = np.asarray(preamble[n_density:], dtype=np.float64)
    if np.any(np.diff(log_temperature) <= 0.0) or np.any(np.diff(log_density) <= 0.0):
        raise OpenADASError(f"{path}: ADF11 axes must be strictly increasing")

    charges: list[int] = []
    arrays: list[np.ndarray] = []
    expected = n_temperature*n_density
    for block, start in enumerate(block_starts):
        match = _CHARGE_RE.search(lines[start])
        assert match is not None
        charge = int(match.group(1))
        if charge < charge_min or charge > charge_max or charge in charges:
            raise OpenADASError(f"{path}: invalid or duplicate ADF11 charge block {charge}")
        values: list[float] = []
        stop = block_starts[block+1] if block+1 < len(block_starts) else len(lines)
        for line in lines[start+1:stop]:
            numeric = _numeric_line(line)
            if numeric is not None:
                values.extend(numeric)
        if len(values) != expected:
            raise OpenADASError(
                f"{path}: charge {charge} has {len(values)} values, expected {expected}"
            )
        charges.append(charge)
        arrays.append(np.asarray(values, dtype=np.float64).reshape(n_temperature, n_density))

    table = ADF11Table(
        path=path,
        data_class=data_class,
        atomic_number=atomic_number,
        log_temperature_eV=np.ascontiguousarray(log_temperature),
        log_density_cm3=np.ascontiguousarray(log_density),
        charge_states=np.asarray(charges, dtype=np.int32),
        log_coefficients=np.ascontiguousarray(np.stack(arrays)),
        header=lines[0],
        sha256=_sha256(path),
    )
    if table.atomic_number != charge_max:
        raise OpenADASError(f"{path}: unresolved ADF11 charge range does not reach bare nucleus")
    return table


@dataclass(frozen=True)
class OpenADASBundle:
    """Validated collection of ACD/SCD/PLT/PRB tables for one element."""

    element: str
    tables: Mapping[str, ADF11Table]

    @classmethod
    def from_paths(cls, element: str, paths: Mapping[str, Path]) -> "OpenADASBundle":
        tables = {kind.lower(): load_adf11(path, kind) for kind, path in paths.items()}
        if not tables:
            raise OpenADASError("an OpenADAS bundle needs at least one ADF11 table")
        atomic_numbers = {table.atomic_number for table in tables.values()}
        if len(atomic_numbers) != 1:
            raise OpenADASError("OpenADAS tables in a bundle have inconsistent atomic numbers")
        return cls(element=element, tables=tables)

    @property
    def atomic_number(self) -> int:
        return next(iter(self.tables.values())).atomic_number

    def coefficient(self, data_class: str, charge_state: int, te_eV: float, ne_m3: float) -> float:
        try:
            table = self.tables[data_class.lower()]
        except KeyError as exc:
            raise OpenADASError(f"bundle has no {data_class.upper()} table") from exc
        return table.evaluate(charge_state, te_eV, ne_m3)

    def ionization_rates(self, te_eV: float, ne_m3: float) -> np.ndarray:
        table = self.tables.get("scd")
        if table is None:
            raise OpenADASError("SCD is required for ionization rates")
        rates = np.zeros(self.atomic_number+1, dtype=np.float64)
        for charge in range(1, self.atomic_number+1):
            rates[charge-1] = table.evaluate(charge, te_eV, ne_m3)
        return rates

    def recombination_rates(self, te_eV: float, ne_m3: float) -> np.ndarray:
        table = self.tables.get("acd")
        if table is None:
            raise OpenADASError("ACD is required for recombination rates")
        rates = np.zeros(self.atomic_number+1, dtype=np.float64)
        for charge in range(1, self.atomic_number+1):
            rates[charge] = table.evaluate(charge, te_eV, ne_m3)
        return rates

    def radiated_power_density(
        self, te_eV: float, ne_m3: float, populations_m3: Mapping[int, float]
    ) -> float:
        """Return n_e sum_q n_q (PLT_q+PRB_q) in W/m^3.

        ``populations_m3`` uses the ADF11 charge-block index explicitly.  The
        caller owns the physical mapping from CR state labels to ADF11 stages.
        """
        total_w_cm3 = 0.0
        for charge, density in populations_m3.items():
            if density < 0.0 or not np.isfinite(density):
                raise OpenADASError("charge-state populations must be finite and non-negative")
            for kind in ("plt", "prb"):
                if kind in self.tables:
                    total_w_cm3 += density*self.coefficient(kind, charge, te_eV, ne_m3)
        return float(ne_m3*total_w_cm3*1.0e-6)

    def provenance(self) -> dict[str, object]:
        return {
            "element": self.element,
            "atomic_number": self.atomic_number,
            "tables": {kind: table.provenance() for kind, table in self.tables.items()},
        }

    def provenance_json(self) -> str:
        return json.dumps(self.provenance(), sort_keys=True)
