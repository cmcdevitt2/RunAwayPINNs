"""Shared model checkpoint and training-artifact persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def utc_now():
    """Return an unambiguous UTC timestamp for manifests and barriers."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    """Write JSON through a same-directory temporary file and atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_replace(source, destination):
    """Atomically replace destination; both paths must share a filesystem."""
    os.replace(Path(source), Path(destination))


def atomic_write_npy(path, array):
    """Write a pickle-disabled NumPy array atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
    import numpy as np
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash file bytes in bounded chunks for manifest checksums."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path):
    """Hash one file or a directory's sorted relative paths and file bytes."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_manifest(dataset_path, manifest_path, *, verify_file=True,
                              expected_config=None,
                              expected_dataset_config=None):
    """Validate dataset completion, path, compatibility, and checksum invariants."""
    dataset_path = Path(dataset_path).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"dataset manifest is not complete: {manifest_path}")
    artifact = manifest.get("artifacts", {})
    recorded_path = Path(artifact.get("dataset", "")).resolve()
    if recorded_path != dataset_path:
        raise ValueError(
            f"dataset path mismatch: config={dataset_path} manifest={recorded_path}")
    generated_config = manifest.get("config", {})
    if expected_config is not None:
        for key in ("parameter_domain", "dataset_config"):
            if key in expected_config and generated_config.get(key) != expected_config[key]:
                raise ValueError(f"dataset {key} mismatch: {manifest_path}")
    if expected_dataset_config:
        generated_dataset_config = generated_config.get("dataset_config", {})
        for key, expected in expected_dataset_config.items():
            if generated_dataset_config.get(key) != expected:
                raise ValueError(
                    f"dataset dataset_config.{key} mismatch: {manifest_path}")
    if not dataset_path.is_file() and not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    recorded_hash = artifact.get("dataset_sha256")
    if not recorded_hash:
        raise ValueError(f"dataset manifest lacks SHA256: {manifest_path}")
    if verify_file and sha256_path(dataset_path) != recorded_hash:
        raise ValueError(f"dataset checksum mismatch: {dataset_path}")
    return manifest


def validate_model_manifest(model_path, manifest_path):
    """Validate model completion, recorded path, and model checksum."""
    model_path = Path(model_path).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"model manifest is not complete: {manifest_path}")
    recorded_path = Path(manifest.get("artifacts", {}).get("model", "")).resolve()
    if recorded_path != model_path:
        raise ValueError(
            f"model path mismatch: config={model_path} manifest={recorded_path}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    recorded_hash = manifest.get("artifacts", {}).get("model_sha256")
    if not recorded_hash:
        raise ValueError(f"model manifest lacks SHA256: {manifest_path}")
    if sha256_file(model_path) != recorded_hash:
        raise ValueError(f"model checksum mismatch: {model_path}")
    return manifest


def slurm_context():
    """Capture reproducibility-relevant Slurm variables without requiring Slurm."""
    names = (
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NTASKS",
        "SLURM_PROCID", "SLURM_LOCALID", "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_GPUS", "SLURM_STEP_ID",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def distributed_file_barrier(directory, label, rank, world, timeout_seconds=7200.0):
    """Synchronize ranks through shared markers and fail fast on failure markers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / f"{label}.rank{rank}.json", {
        "rank": int(rank), "world": int(world), "time": utc_now(),
    })
    deadline = time.monotonic() + timeout_seconds
    while True:
        failures = list(directory.glob("failure.rank*.json"))
        if failures:
            raise RuntimeError(f"distributed preflight failed: {failures[0].read_text()}")
        if len(list(directory.glob(f"{label}.rank*.json"))) >= world:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timeout waiting for distributed barrier {label}")
        time.sleep(0.5)


def begin_run(run_dir, kind, config, *, inputs=None):
    """Create a fresh running manifest; reject non-empty run directories."""
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"run directory is not empty: {run_dir}; choose a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    atomic_write_json(manifest_path, {
        "kind": kind, "status": "running", "started_at": utc_now(),
        "config": config, "inputs": inputs or {}, "slurm": slurm_context(),
    })
    return manifest_path


def finish_run(manifest_path, *, status, artifacts=None, **extra):
    """Atomically set terminal manifest state while preserving idempotence."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if payload.get("status") in ("complete", "failed"):
        return
    payload.update({
        "status": status, "finished_at": utc_now(),
        "artifacts": artifacts or {}, **extra,
    })
    atomic_write_json(manifest_path, payload)


def plot_training_history(config_path=Path("configs/train.json"), run_config=None):
    """Compatibility wrapper for the focused training-plot module."""
    from core.training_plots import plot_training_history as plot
    return plot(config_path, run_config=run_config)


def ensure_training_output_dir(output_dir):
    """Require a fresh output directory before a training run starts."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"training output directory is not empty: {output_dir}; "
            "choose a new output directory")


def make_checkpoint_callback(checkpoint_dir, save_model, *, enabled=True):
    """Return atomic parameter-snapshot callback owned by the calling rank."""
    checkpoint_dir = Path(checkpoint_dir)

    if not enabled:
        return lambda step, params: None

    def save_checkpoint(step, params):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f"pinn_step_{step:08d}.npz"
        temporary = checkpoint_dir / f".pinn_step_{step:08d}.tmp.npz"
        save_model(temporary, params)
        atomic_replace(temporary, model_path)
        atomic_write_json(checkpoint_dir / f"pinn_step_{step:08d}.json", {
            "step": int(step),
            "model": str(model_path),
            "model_sha256": sha256_file(model_path),
        })

    return save_checkpoint


def write_training_outputs(output_dir, params, save_model, *, run_config,
                           metadata, summary, loss_history):
    """Persist fixed final artifacts after enforcing a fresh output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_names = (
        "pinn_params.npz", "run_config.json", "model_metadata.json",
        "loss_history.json", "training_summary.json",
    )
    existing = [name for name in final_names
                if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"training output artifacts already exist in {output_dir}: "
            f"{', '.join(existing)}")
    temporary_model = output_dir / ".pinn_params.tmp.npz"
    model_path = output_dir / "pinn_params.npz"
    save_model(temporary_model, params)
    atomic_replace(temporary_model, model_path)
    atomic_write_json(output_dir / "run_config.json", run_config)
    atomic_write_json(output_dir / "model_metadata.json", metadata)
    atomic_write_json(output_dir / "loss_history.json", loss_history)
    atomic_write_json(output_dir / "training_summary.json", summary)
    return {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "run_config": str(output_dir / "run_config.json"),
        "metadata": str(output_dir / "model_metadata.json"),
        "summary": str(output_dir / "training_summary.json"),
        "loss_history": str(output_dir / "loss_history.json"),
    }
