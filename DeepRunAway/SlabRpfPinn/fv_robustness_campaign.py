#!/usr/bin/env python3
"""B200 FV robustness campaign for the configured parametric RPF domain.

This is deliberately a qualification driver, not a PINN-data generator.  It
samples complete plasma states, runs the steady GPU FV solve for every state on
each configured grid/domain panel, and compares each panel with a designated
reference panel on a common probe grid.  The direct FV solver remains the sole
owner of assembly, transpose, cuDSS solves, and algebraic qualification.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc


CONFIG_PATH = Path(__file__).with_suffix(".toml")
PARAMETER_NAMES = ("E_over_Ec", "te_eV", "nD_m3", "nImp_m3", "zDavg", "zImpavg")


def _map_unit(u: np.ndarray, lo: float, hi: float, scale: str) -> np.ndarray:
    if scale == "linear":
        return lo + u * (hi - lo)
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def _require_range(section: dict, label: str, lo_key: str, hi_key: str, scale_key: str, *, positive: bool = True) -> None:
    lo = float(section[lo_key])
    hi = float(section[hi_key])
    scale = str(section[scale_key]).lower()
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"{label} requires finite increasing bounds")
    if positive and lo <= 0.0:
        raise ValueError(f"{label} requires positive lower bound")
    if scale not in ("linear", "log"):
        raise ValueError(f"{scale_key} must be linear or log")


def load_config(path: Path) -> tuple[dict, Path, str]:
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    for section in ("solver", "domain", "sampling", "comparison", "output"):
        if section not in cfg:
            raise ValueError(f"missing [{section}] in {path}")
    if bool(cfg["solver"].get("energy_diffusion", False)):
        raise ValueError("the PINN-domain FV campaign requires solver.energy_diffusion=false")
    panels = cfg.get("panel", [])
    if not panels:
        raise ValueError("at least one [[panel]] is required")

    domain = cfg["domain"]
    for args in (
        ("e_parallel", "e_parallel_min", "e_parallel_max", "e_parallel_sampling"),
        ("te", "te_min_ev", "te_max_ev", "te_sampling"),
        ("nD", "nD_min_m3", "nD_max_m3", "nD_sampling"),
        ("nImp", "nImp_min_m3", "nImp_max_m3", "nImp_sampling"),
    ):
        _require_range(domain, *args)
    for prefix, zmax in (("zDavg", 1.0), ("zImpavg", float(domain["impurity_Z"]))):
        lo = float(domain[f"{prefix}_min"])
        hi = float(domain[f"{prefix}_max"])
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi <= zmax):
            raise ValueError(f"{prefix} must lie in [0, {zmax:g}]")

    if int(cfg["sampling"].get("n_sobol", 0)) < 0:
        raise ValueError("sampling.n_sobol must be non-negative")
    if not cfg.get("case") and not any((bool(cfg["sampling"].get(key, False)) for key in ("include_center", "include_axis_extrema", "include_corners"))) and int(cfg["sampling"].get("n_sobol", 0)) == 0:
        raise ValueError("sampling selects no parameter cases")

    names = set()
    for panel in panels:
        name = str(panel["name"])
        if name in names:
            raise ValueError(f"duplicate panel name {name!r}")
        names.add(name)
        if int(panel["Np"]) < 2 or int(panel["Nxi"]) < 2:
            raise ValueError(f"panel {name!r} requires Np,Nxi >= 2")
        pmin = float(panel["pmin"])
        pmax = float(panel["pmax"])
        if not (0.0 < pmin < pmax):
            raise ValueError(f"panel {name!r} requires 0 < pmin < pmax")
        mapping = str(panel.get("p_mapping", "uniform")).strip().lower()
        if mapping not in ("uniform", "log", "exponential"):
            raise ValueError(f"panel {name!r} has unsupported p_mapping {mapping!r}")
        kappa = float(panel.get("p_mapping_kappa", 4.0))
        if not (math.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"panel {name!r} requires positive p_mapping_kappa")
        xi_mapping = str(panel.get("xi_mapping", "uniform")).strip().lower()
        if xi_mapping not in ("uniform", "theta"):
            raise ValueError(f"panel {name!r} has unsupported xi_mapping {xi_mapping!r}")

    comparison = cfg["comparison"]
    if str(comparison["reference_panel"]) not in names:
        raise ValueError("comparison.reference_panel must name a panel")
    cmin = float(comparison["pmin"])
    cmax = float(comparison["pmax"])
    if int(comparison["Np"]) < 2 or int(comparison["Nxi"]) < 2 or not (0.0 < cmin < cmax):
        raise ValueError("comparison requires Np,Nxi >= 2 and 0 < pmin < pmax")
    p_sampling = str(comparison.get("p_sampling", "linear")).strip().lower()
    if p_sampling not in ("linear", "log"):
        raise ValueError("comparison.p_sampling must be 'linear' or 'log'")
    for panel in panels:
        if cmin < float(panel["pmin"]) or cmax > float(panel["pmax"]):
            raise ValueError("comparison momentum domain must be contained in every panel")
    return cfg, path, text


def load_fv_module(config_path: Path, solver_script: str):
    path = Path(solver_script)
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"FV solver not found: {path}")
    spec = importlib.util.spec_from_file_location("fv_robustness_solver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import FV solver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def build_cases(cfg: dict) -> tuple[np.ndarray, list[str]]:
    explicit = cfg.get("case", [])
    if explicit:
        domain = cfg["domain"]
        cases: list[list[float]] = []
        labels: list[str] = []
        for item in explicit:
            label = str(item["label"])
            values = [
                float(item["E_over_Ec"]), float(item["te_eV"]),
                float(item["nD_m3"]), float(item["nImp_m3"]),
                float(item["zDavg"]), float(item["zImpavg"]),
            ]
            bounds = [
                (domain["e_parallel_min"], domain["e_parallel_max"]),
                (domain["te_min_ev"], domain["te_max_ev"]),
                (domain["nD_min_m3"], domain["nD_max_m3"]),
                (domain["nImp_min_m3"], domain["nImp_max_m3"]),
                (domain["zDavg_min"], domain["zDavg_max"]),
                (domain["zImpavg_min"], domain["zImpavg_max"]),
            ]
            if any(value < float(lo) or value > float(hi) for value, (lo, hi) in zip(values, bounds)):
                raise ValueError(f"explicit case {label!r} lies outside configured domain")
            if label in labels:
                raise ValueError(f"duplicate explicit case label {label!r}")
            labels.append(label)
            cases.append(values)
        return np.asarray(cases, dtype=np.float64), labels

    sampling = cfg["sampling"]
    units: list[np.ndarray] = []
    labels: list[str] = []
    seen: set[tuple[float, ...]] = set()

    def add(label: str, unit: np.ndarray) -> None:
        key = tuple(np.asarray(unit, dtype=np.float64))
        if key not in seen:
            seen.add(key)
            units.append(np.asarray(unit, dtype=np.float64))
            labels.append(label)

    center = np.full(6, 0.5, dtype=np.float64)
    if bool(sampling.get("include_center", True)):
        add("center", center)
    if bool(sampling.get("include_axis_extrema", True)):
        for dim, name in enumerate(PARAMETER_NAMES):
            for value, tag in ((0.0, "min"), (1.0, "max")):
                point = center.copy()
                point[dim] = value
                add(f"{name}_{tag}", point)
    if bool(sampling.get("include_corners", False)):
        for index in range(1 << 6):
            add(f"corner_{index:02d}", np.array([(index >> dim) & 1 for dim in range(6)], dtype=np.float64))

    n_sobol = int(sampling.get("n_sobol", 0))
    if n_sobol:
        power = int(math.ceil(math.log2(n_sobol)))
        sampler = qmc.Sobol(d=6, scramble=True, seed=int(sampling["seed"]))
        for index, point in enumerate(sampler.random_base2(power)[:n_sobol]):
            add(f"sobol_{index:03d}", point)

    unit = np.asarray(units, dtype=np.float64)
    d = cfg["domain"]
    physical = np.empty_like(unit)
    physical[:, 0] = _map_unit(unit[:, 0], float(d["e_parallel_min"]), float(d["e_parallel_max"]), str(d["e_parallel_sampling"]).lower())
    physical[:, 1] = _map_unit(unit[:, 1], float(d["te_min_ev"]), float(d["te_max_ev"]), str(d["te_sampling"]).lower())
    physical[:, 2] = _map_unit(unit[:, 2], float(d["nD_min_m3"]), float(d["nD_max_m3"]), str(d["nD_sampling"]).lower())
    physical[:, 3] = _map_unit(unit[:, 3], float(d["nImp_min_m3"]), float(d["nImp_max_m3"]), str(d["nImp_sampling"]).lower())
    physical[:, 4] = float(d["zDavg_min"]) + unit[:, 4] * (float(d["zDavg_max"]) - float(d["zDavg_min"]))
    physical[:, 5] = float(d["zImpavg_min"]) + unit[:, 5] * (float(d["zImpavg_max"]) - float(d["zImpavg_min"]))
    return physical, labels


def _case_config(fv, cfg: dict, panel: dict, theta: np.ndarray):
    solver = cfg["solver"]
    domain = cfg["domain"]
    return fv.SolverConfig(
        te_eV=float(theta[1]), E_over_Ec=float(theta[0]), B_T=float(solver["B_T"]),
        species=(
            fv.AtomicSpecies("D", 1, float(theta[4]), float(theta[2])),
            fv.AtomicSpecies(str(domain["impurity_name"]), int(domain["impurity_Z"]), float(theta[5]), float(theta[3])),
        ),
        energy_diffusion=bool(solver.get("energy_diffusion", False)), solve_mode="steady",
        Np=int(panel["Np"]), Nxi=int(panel["Nxi"]), pmin=float(panel["pmin"]), pmax=float(panel["pmax"]),
        p_mapping=str(panel.get("p_mapping", "uniform")).strip().lower(),
        p_mapping_kappa=float(panel.get("p_mapping_kappa", 4.0)),
        xi_mapping=str(panel.get("xi_mapping", "uniform")).strip().lower(),
        device=str(solver["device"]), write_output=False, save_every=0,
    )


def _probe_points(comparison: dict) -> tuple[np.ndarray, tuple[int, int]]:
    np_ = int(comparison["Np"])
    nxi = int(comparison["Nxi"])
    pmin = float(comparison["pmin"])
    pmax = float(comparison["pmax"])
    p_sampling = str(comparison.get("p_sampling", "linear")).strip().lower()
    if p_sampling == "linear":
        p = pmin + (np.arange(np_, dtype=np.float64) + 0.5) * (pmax - pmin) / np_
    elif p_sampling == "log":
        edges = np.exp(np.linspace(np.log(pmin), np.log(pmax), np_ + 1, dtype=np.float64))
        p = np.sqrt(edges[:-1] * edges[1:])
    else:
        raise ValueError(f"unsupported comparison.p_sampling {p_sampling!r}")
    xi = -1.0 + (np.arange(nxi, dtype=np.float64) + 0.5) * 2.0 / nxi
    pp, xx = np.meshgrid(p, xi, indexing="ij")
    return np.column_stack((pp.ravel(), xx.ravel())), (np_, nxi)


def _probe_field(grid, field: np.ndarray, points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    interp = RegularGridInterpolator((grid.p_centers, grid.xi_centers), field, bounds_error=True)
    return np.asarray(interp(points), dtype=np.float64).reshape(shape)


def _scalar_diagnostics(result: dict, field: np.ndarray) -> dict:
    names = (
        "steady_linear_residual", "final_steady_residual", "success_rhs_rel_l2", "failure_rhs_rel_l2",
        "total_escape_rel_l2", "total_escape_backward_max", "transpose_max_abs", "transpose_rel_l2",
        "generator_diag_min", "generator_offdiag_max", "generator_sign_tol", "assembly_s", "analysis_s",
        "steady_factor_s", "steady_solve_s", "factor_s_total",
    )
    diag = {name: float(result[name]) for name in names}
    diag.update({
        "transpose_structure_ok": bool(result["transpose_structure_ok"]),
        "success_pitch_cells": int(result["steady_target"]["success_pitch_cells"]),
        "has_success_boundary": bool(result["steady_target"]["has_success_boundary"]),
        "rpf_min": float(np.min(field)), "rpf_max": float(np.max(field)),
        "factor_count": int(result["factor_count"]), "solve_count": int(result["solve_count"]),
    })
    return diag


def run_campaign(cfg: dict, config_path: Path, config_text: str) -> Path:
    fv, fv_path = load_fv_module(config_path, str(cfg["solver"]["solver_script"]))
    cases, labels = build_cases(cfg)
    outdir = Path(str(cfg["output"]["directory"]))
    if not outdir.is_absolute():
        outdir = config_path.parent / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"FV solver: {fv_path}")
    print(f"parameter cases: {len(cases)}")
    print("initializing Warp/cuDSS runtime ...", flush=True)
    gpu_context = fv.load_gpu_runtime(str(cfg["solver"]["device"]))
    runtime = gpu_context[3]
    points, probe_shape = _probe_points(cfg["comparison"])
    all_fields: dict[str, list[np.ndarray | None]] = {}
    panel_records: list[dict] = []
    start = time.perf_counter()

    for panel in cfg["panel"]:
        name = str(panel["name"])
        base = _case_config(fv, cfg, panel, cases[0])
        fv.validate_rpf_config(base)
        grid = fv.build_grid(base)
        print(
            f"panel={name} grid={grid.Np}x{grid.Nxi} p=[{base.pmin:g},{base.pmax:g}] "
            f"p_mapping={base.p_mapping} xi_mapping={base.xi_mapping}", flush=True
        )
        topo, topology_timing = fv.build_local_csr_topology_gpu(grid, base.device)
        fields: list[np.ndarray | None] = []
        records: list[dict] = []
        for index, theta in enumerate(cases):
            case = _case_config(fv, cfg, panel, theta)
            record = {"index": index, "label": labels[index], "parameters": dict(zip(PARAMETER_NAMES, map(float, theta)))}
            try:
                fv.validate_rpf_config(case)
                phys = fv.derive_physics(case)
                coll = fv.build_collision_data(case, phys, grid)
                fv.validate_collision_data(coll)
                result = fv.solve_adjoint_rpf(case, phys, grid, coll, topo, gpu_context)
                field = np.asarray(result["P_steady_direct"], dtype=np.float64)
                record.update({"status": "pass", "ne_m3": float(phys.free_density_from_ions_m3), "zeff": float(phys.z_eff), **_scalar_diagnostics(result, field)})
                fields.append(field)
                print(f"  {index + 1:3d}/{len(cases):3d} {labels[index]:18s} pass residual={record['steady_linear_residual']:.3e}", flush=True)
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                fields.append(None)
                print(f"  {index + 1:3d}/{len(cases):3d} {labels[index]:18s} FAILED: {record['error']}", flush=True)
            records.append(record)
        all_fields[name] = fields
        panel_records.append({"name": name, "grid": {"Np": grid.Np, "Nxi": grid.Nxi, "pmin": base.pmin, "pmax": base.pmax, "p_mapping": base.p_mapping, "p_mapping_kappa": base.p_mapping_kappa, "xi_mapping": base.xi_mapping}, "topology_timing": topology_timing, "cases": records})

    reference = str(cfg["comparison"]["reference_panel"])
    ref_panel = next(x for x in cfg["panel"] if str(x["name"]) == reference)
    ref_grid = fv.build_grid(_case_config(fv, cfg, ref_panel, cases[0]))
    records_by_panel = {str(item["name"]): item["cases"] for item in panel_records}
    reference_records = records_by_panel[reference]
    comparisons: list[dict] = []
    for panel in cfg["panel"]:
        name = str(panel["name"])
        if name == reference:
            continue
        grid = fv.build_grid(_case_config(fv, cfg, panel, cases[0]))
        for index, (ref_field, field) in enumerate(zip(all_fields[reference], all_fields[name])):
            record = {"reference_panel": reference, "panel": name, "index": index, "label": labels[index]}
            if ref_field is None or field is None:
                record.update({"status": "not_compared", "reason": "solver failure"})
            else:
                ref_probe = _probe_field(ref_grid, ref_field, points, probe_shape)
                probe = _probe_field(grid, field, points, probe_shape)
                delta = probe - ref_probe
                record.update({
                    "status": "compared",
                    "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(ref_probe), 1.0e-300)),
                    "max_abs": float(np.max(np.abs(delta))),
                    "reference_success_pitch_cells": int(reference_records[index]["success_pitch_cells"]),
                    "panel_success_pitch_cells": int(records_by_panel[name][index]["success_pitch_cells"]),
                })
                record["success_pitch_cell_delta"] = (
                    record["panel_success_pitch_cells"] - record["reference_success_pitch_cells"]
                )
            comparisons.append(record)

    failure_count = sum(
        case["status"] != "pass"
        for panel in panel_records
        for case in panel["cases"]
    )
    payload = {
        "schema": "fv-robustness-campaign-v1", "config_path": str(config_path), "config_toml": config_text,
        "fv_solver": str(fv_path), "runtime": runtime, "wall_s": time.perf_counter() - start,
        "status": "pass" if failure_count == 0 else "failed", "failure_count": failure_count,
        "parameter_names": list(PARAMETER_NAMES), "case_labels": labels, "panels": panel_records,
        "comparison": {"settings": cfg["comparison"], "results": comparisons},
    }
    summary = outdir / "fv_robustness_summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if bool(cfg["output"].get("save_fields", False)):
        fields_path = outdir / "fv_robustness_fields.npz"
        arrays = {f"P_{name}": np.asarray([x for x in fields], dtype=np.float64) for name, fields in all_fields.items() if all(x is not None for x in fields)}
        arrays["parameters"] = cases
        np.savez_compressed(fields_path, **arrays)
        print(f"wrote {fields_path}")
    print(f"wrote {summary}")
    if failure_count:
        raise RuntimeError(f"FV robustness campaign recorded {failure_count} failed case(s)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU FV robustness/refinement campaign")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="validate config and list cases without using a GPU")
    args = parser.parse_args()
    cfg, path, text = load_config(args.config)
    cases, labels = build_cases(cfg)
    print(f"loaded campaign: {path}")
    print(f"parameter cases: {len(cases)} ({', '.join(labels)})")
    print(f"panels: {', '.join(str(x['name']) for x in cfg['panel'])}")
    if args.dry_run:
        print("dry run: PASS")
        return
    run_campaign(cfg, path, text)


if __name__ == "__main__":
    main()
