from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

import forward_fv_solver as solver
from bulk_plasma_model import BulkSpecies
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

    def test_state_resolved_plasma_moments_use_charge_square(self):
        ion = solver.IonSpecies(
            name="Ne", Z=2, Z0=0, density_m3=2.0e20,
            charge_populations_m3=(0.0, 1.0e20, 1.0e20),
            I_eV_by_charge=(20.0, 40.0, 1.0),
            a_bar_by_charge=(0.1, 0.2, 1.0),
        )
        cfg = solver.SolverConfig(ne_m3=3.0e20, ions=(ion,))
        phys = solver.derive_physics(cfg)
        self.assertAlmostEqual(phys.free_density_from_ions_m3, 3.0e20)
        self.assertAlmostEqual(phys.z_eff, 5.0/3.0)

    def test_bulk_coupling_rejects_missing_authoritative_screening(self):
        species = BulkSpecies("Ne", "Ne", 2, 1.0e20, (21.6, 41.0), initial_charge=1)
        bulk = solver.BulkCouplingConfig(
            enabled=True, bundle_paths=(("Ne", Path("data/openadas/Ne89")),),
            species=(species,),
        )
        cfg = solver.SolverConfig(ions=(solver.IonSpecies("D", 1, 1, 1.0e20),), bulk=bulk)
        with self.assertRaisesRegex(ValueError, "authoritative state-resolved"):
            solver.build_bulk_model(cfg)

    def test_scaled_coupling_residual_tracks_bulk_and_current(self):
        species = BulkSpecies("D", "H", 1, 1.0e20, (13.6,), initial_charge=1)
        settings = solver.BulkCouplingConfig(
            stage_tolerance=1.0e-8, te_error_scale_eV=1.0,
            population_error_scale_m3=1.0e20, current_error_scale_A=1.0,
        )
        old = solver.BulkState(10.0, (np.asarray([0.0, 1.0e20]),), 2.0)
        new = solver.BulkState(11.0, (np.asarray([1.0e18, 0.99e20]),), 3.0)
        self.assertAlmostEqual(
            solver.coupling_residual(old, new, (2.0, 2.0), (3.0, 3.0), settings),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
