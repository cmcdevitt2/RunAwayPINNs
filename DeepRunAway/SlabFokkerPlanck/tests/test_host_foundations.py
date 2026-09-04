from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

import forward_fv_solver as solver
from openadas_data import OpenADASError, load_adf11


class HostFoundationTests(unittest.TestCase):
    def test_chang_cooper_bernoulli_limits_and_identity(self):
        self.assertAlmostEqual(solver.cc_bernoulli(0.0), 1.0)
        self.assertAlmostEqual(solver.cc_bernoulli(1.0e-8), 1.0-5.0e-9, places=12)
        for x in (-100.0, -3.0, -0.2, 0.2, 3.0, 100.0):
            self.assertGreaterEqual(solver.cc_bernoulli(x), 0.0)
            self.assertAlmostEqual(
                solver.cc_bernoulli(-x)-solver.cc_bernoulli(x), x, places=12
            )

    def test_unresolved_adf11_interpolation_and_range_rejection(self):
        text = """\
   1    2    2    1    1     /TEST                 ADF11
 -------------------------------------------------------------------------------
  10.00000  11.00000
   1.00000   2.00000
 --------------------/ IPRT= 1  / IGRD= 1  /IONIS   / Z1= 1
   0.00000   1.00000
   2.00000   3.00000
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scd_test.dat"
            path.write_text(text, encoding="ascii")
            table = load_adf11(path, "scd")
            # Bilinear interpolation in log10(Te) and log10(ne/cm^3).
            self.assertAlmostEqual(table.evaluate(1, 10.0**1.5, 10.0**16.5), 10.0**1.5)
            self.assertEqual(table.log_coefficients.shape, (1, 2, 2))
            self.assertTrue(table.provenance()["sha256"])
            with self.assertRaises(OpenADASError):
                table.evaluate(1, 1.0, 1.0e20)

    def test_initial_projection_preserves_requested_density(self):
        cfg = solver.SolverConfig(
            Np=8, Nxi=6, pmax=3.0, large_angle_model="none",
            init="gaussian", seed_fraction=0.2,
        )
        phys = solver.derive_physics(cfg)
        grid = solver.build_grid(cfg)
        state = solver.project_initial(cfg, phys, grid)
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertGreaterEqual(float(state.min()), 0.0)
        self.assertAlmostEqual(float(state.sum()), 1.0, places=13)

    def test_state_resolved_screening_uses_charge_populations(self):
        common = dict(
            name="Ne", Z=2, Z0=0, density_m3=2.0e20,
            I_eV=20.0, a_bar=0.1,
            I_eV_by_charge=(20.0, 40.0, 1.0),
            a_bar_by_charge=(0.1, 0.2, 1.0),
        )
        resolved = solver.IonSpecies(
            **common, charge_populations_m3=(0.0, 1.0e20, 1.0e20)
        )
        without_bare = solver.IonSpecies(
            **{**common, "density_m3": 1.0e20},
            charge_populations_m3=(0.0, 1.0e20, 0.0),
        )
        cfg_resolved = solver.SolverConfig(ne_m3=1.0e20, ions=(resolved,))
        cfg_without_bare = solver.SolverConfig(ne_m3=1.0e20, ions=(without_bare,))
        solver.validate_config(cfg_resolved)
        solver.validate_config(cfg_without_bare)
        p = np.asarray([0.3, 1.0, 3.0])
        h_resolved, g_resolved = solver.screening_h_g(
            p, cfg_resolved, solver.derive_physics(cfg_resolved)
        )
        h_without_bare, g_without_bare = solver.screening_h_g(
            p, cfg_without_bare, solver.derive_physics(cfg_without_bare)
        )
        np.testing.assert_allclose(h_resolved, h_without_bare)
        np.testing.assert_allclose(g_resolved, g_without_bare)

    def test_state_resolved_screening_changes_with_bound_charge_state(self):
        ion_q0 = solver.IonSpecies(
            name="Ne", Z=2, Z0=0, density_m3=1.0e20,
            I_eV=20.0, a_bar=0.1,
            charge_populations_m3=(1.0e20, 0.0, 0.0),
            I_eV_by_charge=(20.0, 40.0, 1.0),
            a_bar_by_charge=(0.1, 0.2, 1.0),
        )
        ion_q1 = solver.IonSpecies(
            name="Ne", Z=2, Z0=1, density_m3=1.0e20,
            I_eV=40.0, a_bar=0.2,
        )
        cfg_q0 = solver.SolverConfig(ne_m3=1.0e20, ions=(ion_q0,))
        cfg_q1 = solver.SolverConfig(ne_m3=1.0e20, ions=(ion_q1,))
        p = np.asarray([0.3, 1.0, 3.0])
        h_q0, g_q0 = solver.screening_h_g(p, cfg_q0, solver.derive_physics(cfg_q0))
        h_q1, g_q1 = solver.screening_h_g(p, cfg_q1, solver.derive_physics(cfg_q1))
        self.assertFalse(np.allclose(h_q0, h_q1))
        self.assertFalse(np.allclose(g_q0, g_q1))


if __name__ == "__main__":
    unittest.main()
