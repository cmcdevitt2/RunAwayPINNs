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


if __name__ == "__main__":
    unittest.main()
