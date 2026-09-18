"""Supervised training and high-level workflow orchestration.

Training kernels use FP64 JAX arrays. FV labels stay on the host and outside
automatic differentiation. Distributed paths use one JAX process per node;
rank 0 owns final artifacts and synchronization protects shared files.
"""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from core.training_config import PinnDomain, make_pinn_config
from core.model import load_model, make_probability, save_model
from core.pde import (
    analytic_threshold_collocation, normalize_momentum, sobol_9d_nontrivial,
    sobol_plow_boundary_nontrivial, sobol_pmax_boundary_nontrivial,
)
from core.sampling import split_cases
from core.training_data import (
    _normalize_parameter_cases,
    exact_deeponet_inputs, grouped_to_pointwise,
    train_supervised_deeponet, train_supervised_mlp_grouped,
)
from core.training_physics import (
    _train_physics_multi, train_physics_informed, train_physics_active,
    train_ssbroyden,
)


def detect_local_gpu_count():
    """Return the number of GPUs visible to this process."""
    for name in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS"):
        visible = os.environ.get(name)
        if visible and visible not in ("NoDevFiles", "-1"):
            return len([item for item in visible.split(",") if item.strip()])
    slurm_count = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if slurm_count:
        try:
            return int(slurm_count.rsplit(":", 1)[-1])
        except ValueError:
            pass
    visible = os.environ.get("SLURM_JOB_GPUS")
    if visible and visible not in ("NoDevFiles", "-1"):
        return len([item for item in visible.split(",") if item.strip()])
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        return sum(line.startswith("GPU ") for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0


def initialize_slurm_jax(local_gpu_count):
    """Initialize multi-host JAX using one task per node and shared networking."""
    if "SLURM_PROCID" not in os.environ:
        return
    process_count = int(os.environ.get("SLURM_NTASKS", "1"))
    process_id = int(os.environ.get("SLURM_PROCID", "0"))
    if process_count <= 1:
        return
    # The first allocated host coordinates all processes. Slurm must expose
    # hostnames, DNS, and a reachable port selected from the job ID.
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        text=True).splitlines()
    port = 29500 + int(os.environ.get("SLURM_JOB_ID", "0")) % 1000
    coordinator_host = socket.gethostbyname(hosts[0])
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_host}:{port}",
        num_processes=process_count,
        process_id=process_id,
        local_device_ids=list(range(local_gpu_count)),
        coordinator_bind_address=f"0.0.0.0:{port}",
    )


def train_data_from_config(config_path=Path("configs/train.json"),
                           run_config=None):
    """Validate data, train model, and publish rank-0 artifacts."""
    from core.fv_dataset import load_fv_cases
    from core.training_artifacts import (
        atomic_write_json, begin_run, distributed_file_barrier,
        ensure_training_output_dir, finish_run, make_checkpoint_callback,
        validate_dataset_manifest, write_training_outputs,
    )
    config_path = Path(config_path)
    if run_config is None:
        run_config = json.loads(config_path.read_text())
    local_gpus = detect_local_gpu_count()
    if local_gpus <= 0:
        raise RuntimeError("no visible GPUs found")
    launched = "SLURM_PROCID" in os.environ
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1")) if launched else 1
    run_dir = Path(run_config.get("run_dir", run_config["output_dir"]))
    launch_id = (f"{os.environ.get('SLURM_JOB_ID', 'local')}."
                 f"{os.environ.get('SLURM_STEP_ID', '0')}")
    if not launched:
        launch_id = f"{launch_id}.{os.getpid()}"
    launch_dir = run_dir.parent / f".{run_dir.name}.launch.{launch_id}"
    manifest = None
    try:
        if rank == 0:
            launch_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(launch_dir / "launch.json", {
                "config": str(config_path.resolve()), "world": world,
            })
        distributed_file_barrier(launch_dir, "launch", rank, world)
        initialize_slurm_jax(local_gpus)
        if len(jax.local_devices()) != local_gpus:
            raise RuntimeError("local GPU count changed during JAX initialization")
        config = make_pinn_config(run_config)
        if config.training_mode != "data":
            raise ValueError("data training requires mode='data'")
        validate_dataset_manifest(
            run_config["dataset_path"], run_config["dataset_manifest"],
            expected_config=run_config,
            expected_dataset_config={
                "p_max": run_config["domain"]["p_max"],
            "fv_p_min": run_config["domain"]["p_min"],
            })
        output_dir = Path(run_config["output_dir"])
        ensure_training_output_dir(output_dir)
        distributed_file_barrier(launch_dir, "preflight", rank, world)
        if jax.process_index() == 0:
            manifest = begin_run(run_dir, "pinn_training", run_config)
        distributed_file_barrier(launch_dir, "run_started", rank, world)
        checkpoint_dir = Path(run_config.get("checkpoint_dir", output_dir / "checkpoints"))
        checkpoint = (make_checkpoint_callback(
            checkpoint_dir, save_model, enabled=(jax.process_index() == 0))
            if config.checkpoint_every > 0 else None)
        cases, results = load_fv_cases(run_config["dataset_path"])
        train_cases, train_results, test_cases, test_results = split_cases(
            cases, results, config.train_case_fraction, config.seed + 1)
        process = jax.process_index()
        processes = jax.process_count()
        local_cases = train_cases[process::processes]
        local_results = train_results[process::processes]
        def points(result):
            return len(result["p"]) * len(result["xi"])
        max_points = max(points(result) for result in train_results + test_results)
        grouped_train = exact_deeponet_inputs(
            local_cases, local_results, run_config["parameter_domain"],
            config.domain, max_points=max_points)
        if process == 0:
            grouped_test = exact_deeponet_inputs(
                test_cases, test_results, run_config["parameter_domain"],
                config.domain, max_points=max_points)
        else:
            grouped_test = {
                "branch": np.empty((0, 7)),
                "trunk": np.empty((0, max_points, 2)),
                "target": np.empty((0, max_points)),
                "mask": np.empty((0, max_points)),
            }
        distributed_file_barrier(launch_dir, "data_ready", rank, world)
        if config.model_type == "deeponet":
            params, history, metrics = train_supervised_deeponet(
                grouped_train, grouped_test, config=config, distributed=True,
                checkpoint_callback=checkpoint,
                global_case_count=min(len(train_cases[i::processes])
                                      for i in range(processes)) * processes)
        elif config.model_type == "mlp":
            params, history, metrics = train_supervised_mlp_grouped(
                grouped_train, grouped_test, config=config, distributed=True,
                checkpoint_callback=checkpoint)
        else:
            raise ValueError(f"unsupported model_type: {config.model_type}")
        if jax.process_index() == 0:
            history = np.asarray(history)
            metadata = {
                "architecture": {
            "input_dim": 9, "model_type": config.model_type,
                    "width": config.width, "depth": config.depth,
                    "latent_width": config.latent_width,
                    "branch_width": config.branch_width,
                    "branch_depth": config.branch_depth,
                    "trunk_width": config.trunk_width,
                    "trunk_depth": config.trunk_depth,
                    "output_transform": config.output_transform,
                },
                "domain": run_config["domain"],
                "normalization": {"p_floor": config.domain.p_min,
                                  "p_max": config.domain.p_max},
                "training_config": {
                    "model": run_config.get("model", run_config.get("pinn_config", {})),
                    "loss": run_config.get("loss", {}),
                    "optimizer": run_config.get("optimizer", {}),
                    "data": run_config.get("data", {}),
                },
                "parameter_domain": run_config["parameter_domain"],
                "dataset": run_config["dataset_path"],
                "residual_regularization": {
                    "coeff_norm": config.residual_coeff_norm,
                    "floor": config.residual_floor,
                },
            }
            summary = {
                "dataset_path": run_config["dataset_path"],
                "output_dir": str(output_dir),
                "train_case_count": int(len(train_cases)),
                "test_case_count": int(len(test_cases)),
                "model_type": config.model_type, "metrics": metrics,
                "history": history.tolist(),
                "history_columns": (["train_loss", "test_loss"]
                                    if history.ndim == 2 and history.shape[1] == 2 else None),
                "history_steps": (np.arange(len(history)) * config.log_every).tolist(),
                "log_every": config.log_every,
                "checkpoint_every": config.checkpoint_every,
                "global_device_count": int(jax.device_count()),
                "process_count": int(jax.process_count()),
            }
            artifacts = write_training_outputs(
                output_dir, params, save_model, run_config=run_config,
                metadata=metadata, summary=summary,
                loss_history={"steps": summary["history_steps"],
                              "columns": summary["history_columns"],
                              "values": summary["history"]})
            finish_run(manifest, status="complete", artifacts=artifacts, metrics=metrics)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("training_artifacts_complete")
    except Exception as exc:
        try:
            atomic_write_json(launch_dir / f"failure.rank{rank}.json", {
                "rank": rank, "error": repr(exc),
            })
        except OSError:
            pass
        if manifest is not None:
            finish_run(manifest, status="failed", error=repr(exc))
        raise


def _build_physics_inputs(dataset, cases, parameter_domain, domain):
    """Convert exact FV cells into normalized nine-coordinate PINN samples."""
    from core.fv_dataset import flatten_fv_dataset
    # Flattening preserves exact FV cell order; only coordinate normalization
    # changes before values enter the PINN.
    flat = flatten_fv_dataset(dataset)
    case_index = np.asarray(flat["case_index"], dtype=np.int64)
    p = np.asarray(flat["p"], dtype=np.float64)
    z = np.empty((len(p), 9), dtype=np.float64)
    z[:, 0] = normalize_momentum(
        p, domain.p_min, domain.p_max, domain.momentum_sampling)
    z[:, 1] = 0.5 * (np.asarray(flat["xi"]) + 1.0)
    case_norm = _normalize_parameter_cases(cases, parameter_domain)
    z[:, 2:] = case_norm[case_index]
    y = np.asarray(flat["P"], dtype=np.float64)
    return z, y, case_index


def train_physics_from_config(config_path=Path("configs/train.json"),
                              config=None):
    """Build collocation inputs, run physics phases, and publish artifacts."""
    from core.fv_dataset import (
        coarsen_fv_cases, denormalize_parameter_cases, load_fv_dataset,
        generate_cpu_cases,
    )
    from core.training_artifacts import (
        atomic_write_json, begin_run, distributed_file_barrier,
        ensure_training_output_dir, finish_run, make_checkpoint_callback,
        validate_dataset_manifest, write_training_outputs,
    )
    config_path = Path(config_path)
    if config is None:
        config = json.loads(config_path.read_text())
    if config.get("mode") not in (None, "physics"):
        raise ValueError("set mode='physics' in the training config")
    local_gpus = detect_local_gpu_count()
    if local_gpus <= 0:
        raise RuntimeError("no visible GPUs found")
    launched = "SLURM_PROCID" in os.environ
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1")) if launched else 1
    run_dir = Path(config.get("run_dir", config["output_dir"]))
    launch_id = (f"{os.environ.get('SLURM_JOB_ID', 'local')}."
                 f"{os.environ.get('SLURM_STEP_ID', '0')}")
    if not launched:
        launch_id = f"{launch_id}.{os.getpid()}"
    launch_dir = run_dir.parent / f".{run_dir.name}.launch.{launch_id}"
    manifest = None
    try:
        if rank == 0:
            launch_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(launch_dir / "launch.json", {
                "config": str(config_path.resolve()), "world": world,
            })
        distributed_file_barrier(launch_dir, "launch", rank, world)
        initialize_slurm_jax(local_gpus)
        if len(jax.local_devices()) != local_gpus:
            raise RuntimeError("local GPU count changed during JAX initialization")
        if jax.default_backend() != "gpu":
            raise RuntimeError("GPU backend required for physics training")
        training_config = make_pinn_config(config)
        if training_config.training_mode != "physics_informed":
            raise ValueError("physics training requires mode='physics'")
        if training_config.model_type != "mlp":
            raise ValueError("physics training requires model.model_type='mlp'")
        initial_model = config.get("initial_model")
        initial_params = None
        if initial_model:
            initial_params = load_model(
                initial_model, model_type=training_config.model_type,
                width=training_config.width, depth=training_config.depth,
                latent_width=training_config.latent_width,
                branch_width=training_config.branch_width,
                branch_depth=training_config.branch_depth,
                trunk_width=training_config.trunk_width,
                trunk_depth=training_config.trunk_depth)
        dataset_path = Path(config["dataset_path"])
        validate_dataset_manifest(
            dataset_path, config["dataset_manifest"], expected_config=config,
            expected_dataset_config={"p_max": config["domain"]["p_max"],
                                     "fv_p_min": config["domain"]["p_min"]})
        output_dir = Path(config["output_dir"])
        ensure_training_output_dir(output_dir)
        distributed_file_barrier(launch_dir, "preflight", rank, world)
        if jax.process_index() == 0:
            manifest = begin_run(run_dir, "physics_pinn_training", config)
        distributed_file_barrier(launch_dir, "run_started", rank, world)
        domain = PinnDomain(**config["domain"])
        checkpoint_dir = Path(config.get("checkpoint_dir", output_dir / "checkpoints"))
        checkpoint = (make_checkpoint_callback(
            checkpoint_dir, save_model, enabled=(jax.process_index() == 0))
            if training_config.checkpoint_every > 0 else None)
        dataset = load_fv_dataset(dataset_path)
        cases = np.asarray(dataset["cases"], dtype=np.float64)
        z_all, y_all, case_index = _build_physics_inputs(
            dataset, cases, config["parameter_domain"], domain)
        rng = np.random.default_rng(training_config.seed)
        permutation = rng.permutation(len(cases))
        n_train = min(
            max(1, int(round(training_config.train_case_fraction * len(cases)))),
            len(cases) - 1)
        train_cases = np.sort(permutation[:n_train])
        test_cases = np.sort(permutation[n_train:])
        train_pool = np.flatnonzero(np.isin(case_index, train_cases))
        test_pool = np.flatnonzero(np.isin(case_index, test_cases))
        train_count = min(int(config.get("data", {}).get("train_points", len(train_pool))),
                          len(train_pool))
        test_count = min(int(config.get("data", {}).get("test_points", len(test_pool))),
                         len(test_pool))
        train_idx = rng.choice(train_pool, size=train_count, replace=False)
        test_idx = rng.choice(test_pool, size=test_count, replace=False)
        collocation = config["collocation"]
        if collocation.get("pde_on_data", False):
            if int(collocation.get("pde_points", 0)) != 0:
                raise ValueError("pde_on_data requires collocation.pde_points=0")
            z_pde = z_all[train_idx]
        else:
            z_pde = sobol_9d_nontrivial(
                int(collocation["pde_points"]), domain, training_config.seed + 1,
                angular_sampling=training_config.angular_sampling,
                momentum_sampling=training_config.momentum_sampling)
        z_threshold = analytic_threshold_collocation(
            int(collocation["threshold_points"]), domain,
            seed=training_config.seed + 2,
            band_width=float(collocation.get("threshold_band_width", 0.02)))
        z_low = sobol_plow_boundary_nontrivial(
            int(collocation["low_p_points"]), domain, training_config.seed + 3,
            angular_sampling=training_config.angular_sampling)
        z_bc = sobol_pmax_boundary_nontrivial(
            int(collocation["boundary_points"]), domain, training_config.seed + 4,
            angular_sampling=training_config.angular_sampling)
        start = time.perf_counter()
        active = config.get("active", {})
        active_records = []
        active_cycles = int(active.get("cycles", 1))
        if active.get("enabled", False):
            # Active acquisition performs CPU FV solves between JAX training
            # cycles and is intentionally limited to one JAX process.
            if jax.process_count() > 1:
                raise RuntimeError("active FV acquisition requires one JAX process")
            def acquire_data(z_selected):
                case_norm = np.unique(np.asarray(z_selected)[:, 2:], axis=0)
                selected_cases = denormalize_parameter_cases(
                    case_norm, config["parameter_domain"])
                results = generate_cpu_cases(
                    selected_cases, p_max=domain.p_max,
                    Np=int(active.get("fv_Np", 256)), Nxi=int(active.get("fv_Nxi", 64)),
                    p_min_global=domain.p_min,
                    N_p_coarse=int(active.get("fv_p_coarse_N", 32)),
                    n_jobs=int(active.get("n_jobs", -1)))
                keep = np.asarray([
                    not result["trivial_zero"] and result.get("valid", True)
                    for result in results])
                if not np.any(keep):
                    return np.empty((0, 9)), np.empty(0)
                kept_cases = selected_cases[keep]
                kept_results = [r for r, k in zip(results, keep) if k]
                kept_results = coarsen_fv_cases(
                    kept_results, p_stride=int(active.get("fv_p_stride", 1)),
                    xi_stride=int(active.get("fv_xi_stride", 1)))
                grouped = exact_deeponet_inputs(
                    kept_cases, kept_results, config["parameter_domain"], domain)
                return grouped_to_pointwise(grouped)
            params, soap_history, active_records = train_physics_active(
                z_all[train_idx], y_all[train_idx], z_pde, z_bc, domain=domain,
                z_pde_threshold=z_threshold, z_low=z_low,
                config=training_config, active_config=active,
                acquire_data=acquire_data, checkpoint_callback=checkpoint,
                distributed=False, initial_params=initial_params)
            active_cycles = max(1, active_cycles)
        else:
            params, soap_history, _ = train_physics_informed(
                z_all[train_idx], y_all[train_idx], z_pde, z_bc, domain=domain,
                z_pde_threshold=z_threshold, z_low=z_low,
                config=training_config, checkpoint_callback=checkpoint,
                distributed=(jax.process_count() > 1), initial_params=initial_params)
            active_cycles = 1
        ssb_history = []
        if config.get("ssbroyden", {}).get("enabled", False):
            if jax.process_count() > 1:
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices("ssbroyden_start")
                if jax.process_index() == 0:
                    params, ssb_history = train_ssbroyden(
                        params, z_all[train_idx], y_all[train_idx], z_pde, z_bc,
                        domain=domain, z_pde_threshold=z_threshold, z_low=z_low,
                        config=training_config, checkpoint_callback=checkpoint,
                        step_offset=training_config.steps * active_cycles)
                params = multihost_utils.broadcast_one_to_all(
                    params, is_source=(jax.process_index() == 0))
                multihost_utils.sync_global_devices("ssbroyden_complete")
            elif jax.process_index() == 0:
                params, ssb_history = train_ssbroyden(
                    params, z_all[train_idx], y_all[train_idx], z_pde, z_bc,
                    domain=domain, z_pde_threshold=z_threshold, z_low=z_low,
                    config=training_config, checkpoint_callback=checkpoint,
                    step_offset=training_config.steps * active_cycles)
        if jax.process_index() == 0:
            _, predict_fn = make_probability(training_config.output_transform)
            prediction_train = np.asarray(jax.device_get(
                predict_fn(params, jnp.asarray(z_all[train_idx]))))
            prediction_test = np.asarray(jax.device_get(
                predict_fn(params, jnp.asarray(z_all[test_idx]))))
            metadata = {"architecture": {
                "input_dim": 9, "model_type": training_config.model_type,
                "width": training_config.width, "depth": training_config.depth,
                "latent_width": training_config.latent_width,
                "branch_width": training_config.branch_width,
                "branch_depth": training_config.branch_depth,
                "trunk_width": training_config.trunk_width,
                "trunk_depth": training_config.trunk_depth,
                "output_transform": training_config.output_transform,
            }, "domain": config["domain"], "normalization": {
                "p_floor": domain.p_min, "p_max": domain.p_max,
            }, "training_config": {"model": config.get("model", {}),
                                    "loss": config.get("loss", {}),
                                    "optimizer": config.get("optimizer", {}),
                                    "data": config.get("data", {})},
            "parameter_domain": config["parameter_domain"],
            "dataset": str(dataset_path),
            "initial_model": initial_model,
            "residual_regularization": {
                "coeff_norm": training_config.residual_coeff_norm,
                "floor": training_config.residual_floor,
            }}
            summary = {"dataset": str(dataset_path),
                       "train_cases": int(len(train_cases)),
                       "test_cases": int(len(test_cases)),
                       "train_mse": float(np.mean((prediction_train - y_all[train_idx]) ** 2)),
                       "test_mse": float(np.mean((prediction_test - y_all[test_idx]) ** 2)),
                       "soap_loss_history": np.asarray(soap_history).tolist(),
                       "ssbroyden_history": ssb_history,
                       "active_history": active_records,
                       "elapsed_seconds": time.perf_counter() - start,
                       "checkpoint_every": training_config.checkpoint_every,
                       "global_device_count": int(jax.device_count()),
                       "process_count": int(jax.process_count())}
            loss_history = {"soap": {"steps": np.arange(len(soap_history)).tolist(),
                                      "columns": ["total", "data", "PDE", "threshold PDE", "low-p", "p_max BC"],
                                      "values": np.asarray(soap_history).tolist()},
                            "ssbroyden": ssb_history, "active": active_records}
            artifacts = write_training_outputs(
                output_dir, params, save_model, run_config=config,
                metadata=metadata, summary=summary, loss_history=loss_history)
            finish_run(manifest, status="complete", artifacts=artifacts,
                       metrics={"train_mse": summary["train_mse"],
                                "test_mse": summary["test_mse"]})
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("training_artifacts_complete")
    except Exception as exc:
        try:
            atomic_write_json(launch_dir / f"failure.rank{rank}.json", {
                "rank": rank, "error": repr(exc),
            })
        except OSError:
            pass
        if manifest is not None:
            finish_run(manifest, status="failed", error=repr(exc))
        raise
