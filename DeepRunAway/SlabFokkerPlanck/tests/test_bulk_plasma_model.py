from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.constants as const

from bulk_plasma_model import BulkPlasmaModel, BulkSpecies
from openadas_data import OpenADASBundle


def synthetic_bundle(directory: Path) -> OpenADASBundle:
    header = "   1    2    2    1    1     /TEST                 ADF11"
    files = {}
    for kind in ("acd", "scd", "plt", "prb"):
        # Density grid precedes temperature grid in unresolved ADF11 files.
        coefficient = {"acd": -8.0, "scd": -8.0, "plt": -30.0, "prb": -30.0}[kind]
        text = "\n".join((
            header,
            "  13.00000  14.00000",
            "   0.00000   2.00000",
            " --------------------/ Z1= 1",
            f" {coefficient:.6f} {coefficient:.6f}",
            f" {coefficient:.6f} {coefficient:.6f}",
            "",
        ))
        path = directory/f"{kind}89_h.dat"
        path.write_text(text, encoding="ascii")
        files[kind] = path
    return OpenADASBundle.from_paths("D", files)


class BulkPlasmaModelTests(unittest.TestCase):
    def test_charge_conservation_energy_ledger_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = synthetic_bundle(Path(directory))
            species = BulkSpecies("D", "H", 1, 1.0e20, (13.6,), initial_charge=1)
            model = BulkPlasmaModel(
                (species,), {"D": bundle}, area_m2=1.0, major_radius_m=1.0,
                inductance_H=1.0, applied_voltage_V=0.0,
            )
            state = model.initial_state(10.0, current_A=1.0)
            derivative = model.rhs(state)
            self.assertAlmostEqual(float(derivative.populations_m3_s[0].sum()), 0.0, places=6)
            diagnostics = derivative.diagnostics
            self.assertGreater(diagnostics.electron_density_m3, 0.0)
            self.assertGreater(diagnostics.resistivity_ohm_m, 0.0)
            dne = float(np.dot(np.arange(2), derivative.populations_m3_s[0]))
            d_atomic = const.e*float(np.dot((0.0, 13.6), derivative.populations_m3_s[0]))
            dtotal = 1.5*const.e*(
                diagnostics.electron_density_m3*derivative.temperature_eV_s
                + state.temperature_eV*dne
            ) + d_atomic
            self.assertAlmostEqual(
                dtotal, diagnostics.ohmic_power_W_m3-diagnostics.radiated_power_W_m3,
                # The individual ionization/thermal terms are O(1e8 W/m^3)
                # and cancel to an O(1e-5 W/m^3) net power in this synthetic case.
                delta=1.0e-10*max(
                    1.0, abs(dtotal), abs(d_atomic),
                    abs(1.5*const.e*diagnostics.electron_density_m3*derivative.temperature_eV_s),
                ),
            )
            # The no-external-power model has no P_ext input.
            self.assertFalse(hasattr(derivative, "external_power_W_m3"))

    def test_trbdf2_preserves_species_total(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = synthetic_bundle(Path(directory))
            species = BulkSpecies("D", "H", 1, 1.0e20, (13.6,), initial_charge=1)
            model = BulkPlasmaModel(
                (species,), {"D": bundle}, area_m2=1.0, major_radius_m=1.0,
                inductance_H=1.0,
            )
            final = model.step_trbdf2(model.initial_state(10.0), 1.0e-10)
            self.assertGreater(final.temperature_eV, 0.0)
            self.assertTrue(np.all(final.populations_m3[0] >= 0.0))
            self.assertAlmostEqual(float(final.populations_m3[0].sum()), 1.0e20, delta=2.0e8)


if __name__ == "__main__":
    unittest.main()
