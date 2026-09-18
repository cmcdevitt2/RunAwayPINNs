#!/usr/bin/env python3
"""Generate one deterministic FV dataset across an existing Slurm allocation.

Use one process per node and a shared work directory. Rank 0 owns the Sobol
stream, acceptance order, final dataset publication, and completion manifest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from core.fv_dataset import (
    FvDatasetConfig,
    coarsen_fv_cases,
    denormalize_parameter_cases,
    generate_cpu_cases,
    load_fv_cases,
    save_fv_dataset_npz,
    save_fv_dataset_directory,
)
from core.training_artifacts import (
    atomic_replace, atomic_write_json, atomic_write_npy, begin_run,
    finish_run, sha256_path,
)

MAX_FV_ROUNDS = 128


def _barrier(work_dir, label, rank, world, timeout_seconds=7200.0):
    """Synchronize ranks with shared marker files and propagate rank failures."""
    marker = work_dir / f"{label}.rank{rank}"
    deadline = time.monotonic() + timeout_seconds
    while True:
        errors = list(work_dir.glob("error.rank*"))
        if errors:
            message = errors[0].read_text(errors="replace")
            raise RuntimeError(f"distributed FV rank failed: {message}")
        if work_dir.exists():
            marker.touch()
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timeout waiting for work directory {work_dir}")
        time.sleep(0.5)
    while True:
        errors = list(work_dir.glob("error.rank*"))
        if errors:
            message = errors[0].read_text(errors="replace")
            raise RuntimeError(f"distributed FV rank failed: {message}")
        if len(list(work_dir.glob(f"{label}.rank*"))) >= world:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timeout waiting for distributed barrier {label}")
        time.sleep(0.5)


def _merge_round(work_dir, round_number, world):
    """Merge rank files in deterministic candidate order for this round."""
    records = []
    rank_records = []
    for rank in range(world):
        cases, results = load_fv_cases(
            work_dir / f"results.round{round_number}.rank{rank}.npz")
        rank_records.append(list(zip(cases, results)))
    for offset in range(max(len(items) for items in rank_records)):
        for rank in range(world):
            if offset < len(rank_records[rank]):
                records.append(rank_records[rank][offset])
    return records


def _main(config_path):
    """Run oversampled FV rounds until rank 0 accepts the requested case count."""
    with Path(config_path).open() as stream:
        run_config = json.load(stream)
    parameter_domain = run_config["parameter_domain"]
    dataset_config = run_config["dataset_config"]
    FvDatasetConfig(**dataset_config)
    candidate_oversample = dataset_config.get("candidate_oversample", 1.25)
    output_path = Path(run_config["dataset_path"])
    work_dir = Path(run_config["work_dir"])
    run_dir = Path(run_config.get("run_dir", work_dir.parent))
    launched_by_srun = "SLURM_PROCID" in os.environ
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1")) if launched_by_srun else 1
    if world < 1:
        raise ValueError("SLURM_NTASKS must be positive")
    if not 0 <= rank < world:
        raise ValueError(f"invalid Slurm rank {rank} for world size {world}")
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    allocated_count = int(allocated_cpus) if allocated_cpus else None
    # Half the allocated CPUs leaves headroom for the rank and sparse solver;
    # explicit worker counts remain bounded by Slurm's CPU allocation.
    worker_count = dataset_config["n_jobs"]
    if worker_count == -1 and allocated_count is not None:
        worker_count = max(1, allocated_count // 2)
    if worker_count < -1:
        raise ValueError("dataset n_jobs must be positive or -1")
    if allocated_count is not None and worker_count > allocated_count:
        raise ValueError(
            f"dataset n_jobs={worker_count} exceeds "
            f"SLURM_CPUS_PER_TASK={allocated_cpus}")
    if rank == 0:
        if work_dir.exists():
            (work_dir / "error.rank0").write_text(
                f"stale FV work directory exists: {work_dir}\n")
            raise FileExistsError(f"distributed FV work directory exists: {work_dir}")
        manifest_path = begin_run(run_dir, "fv_dataset", run_config)
        work_dir.mkdir(parents=True)
        print(f"FV ranks: {world}; Joblib workers/rank: {worker_count}; "
              f"target cases: {dataset_config['n_cases']}", flush=True)
    _barrier(work_dir, "workdir-ready", rank, world)

    # Only rank 0 advances Sobol state. Other ranks receive exact candidate
    # files, which keeps a multi-node run reproducible.
    engine = qmc.Sobol(d=7, scramble=True, seed=dataset_config["seed"]) if rank == 0 else None
    accepted_cases = []
    accepted_results = []
    round_number = 0
    while True:
        remaining = (dataset_config["n_cases"] - len(accepted_results)
                     if rank == 0 else dataset_config["n_cases"])
        # Oversampling compensates for trivial-zero and invalid FV cases. The
        # rounded count gives every rank the same number of candidates.
        candidate_count = max(world, int(np.ceil(
            candidate_oversample * remaining)))
        candidate_count = ((candidate_count + world - 1) // world) * world
        candidate_path = work_dir / f"candidates.round{round_number}.npy"
        if rank == 0:
            # Acceptance order follows the deterministic rank-interleaved
            # candidate order, independent of worker completion timing.
            print(f"FV round {round_number}: solving {candidate_count} candidates; "
                  f"accepted {dataset_config['n_cases'] - remaining}/"
                  f"{dataset_config['n_cases']}", flush=True)
            candidates = denormalize_parameter_cases(
                engine.random(candidate_count), parameter_domain)
            atomic_write_npy(candidate_path, candidates)
        _barrier(work_dir, f"candidates-ready.round{round_number}", rank, world)

        candidates = np.load(candidate_path, allow_pickle=False)
        local_cases = candidates[rank::world]
        local_results = generate_cpu_cases(
            local_cases, p_max=dataset_config["p_max"],
            Np=dataset_config["fv_Np"], Nxi=dataset_config["fv_Nxi"],
            p_min_global=dataset_config["fv_p_min"],
            N_p_coarse=dataset_config["fv_p_coarse_N"], n_jobs=worker_count,
            p_grid_mode=dataset_config.get("p_grid_mode", "log"),
            p_split_fraction=dataset_config.get("p_split_fraction", 0.75),
            p_cluster_power=dataset_config.get("p_cluster_power", 2.5))
        local_results = coarsen_fv_cases(
            local_results, p_stride=dataset_config["fv_p_stride"],
            xi_stride=dataset_config["fv_xi_stride"],
            boundary_p_cells=dataset_config.get("fv_boundary_p_cells", 0),
            boundary_xi_cells=dataset_config.get("fv_boundary_xi_cells", 0))
        result_path = work_dir / f"results.round{round_number}.rank{rank}.npz"
        temporary_result = result_path.with_name(
            f".{result_path.name}.{os.getpid()}.tmp.npz")
        save_fv_dataset_npz(
            temporary_result, local_cases, local_results,
            p_max=dataset_config["p_max"])
        atomic_replace(temporary_result, result_path)
        _barrier(work_dir, f"results-ready.round{round_number}", rank, world)

        if rank == 0:
            records = _merge_round(work_dir, round_number, world)
            for case, result in records:
                if not result["trivial_zero"] and result.get("valid", True):
                    accepted_cases.append(case)
                    accepted_results.append(result)
                    if len(accepted_results) == dataset_config["n_cases"]:
                        break
            atomic_write_json(work_dir / f"status.round{round_number}.json", {
                "accepted_cases": len(accepted_results),
                "complete": len(accepted_results) == dataset_config["n_cases"],
            })
        _barrier(work_dir, f"round-complete.round{round_number}", rank, world)
        status = json.loads(
            (work_dir / f"status.round{round_number}.json").read_text())
        if status["complete"]:
            break
        if round_number + 1 >= MAX_FV_ROUNDS:
            if rank == 0:
                finish_run(
                    manifest_path, status="failed",
                    error=f"FV acceptance did not reach target after {MAX_FV_ROUNDS} rounds")
            raise RuntimeError(
                f"FV acceptance did not reach target after {MAX_FV_ROUNDS} rounds")
        round_number += 1

    if rank == 0:
        # Publish only after all accepted cases exist. The manifest checksum
        # then covers the complete dataset artifact.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_format = run_config.get("dataset_format", "npz")
        if dataset_format == "directory":
            save_fv_dataset_directory(
                output_path, np.asarray(accepted_cases), accepted_results,
                p_floor=dataset_config["fv_p_min"],
                p_max=dataset_config["p_max"])
        elif dataset_format == "npz":
            temporary_output = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.tmp.npz")
            save_fv_dataset_npz(
                temporary_output, np.asarray(accepted_cases), accepted_results,
                p_floor=dataset_config["fv_p_min"],
                p_max=dataset_config["p_max"])
            atomic_replace(temporary_output, output_path)
        else:
            raise ValueError("dataset_format must be 'npz' or 'directory'")
        runtimes = [result["runtime_seconds"] for result in accepted_results]
        finish_run(
            manifest_path,
            status="complete",
            artifacts={
                "dataset": str(output_path),
                "dataset_sha256": sha256_path(output_path),
            },
            cases=len(accepted_results),
            average_runtime_seconds=float(np.mean(runtimes)),
        )
        print(f"distributed FV ranks: {world}")
        print(f"generated FV cases: {len(accepted_results)}")
        print(f"average FV runtime: {np.mean(runtimes):.3f} s/case")


def main(config_path):
    """Run FV generation and publish failures to all distributed ranks."""
    config_path = Path(config_path)
    with config_path.open() as stream:
        run_config = json.load(stream)
    work_dir = Path(run_config["work_dir"])
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    try:
        return _main(config_path)
    except Exception as exc:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / f"error.rank{rank}").write_text(repr(exc) + "\n")
        raise
