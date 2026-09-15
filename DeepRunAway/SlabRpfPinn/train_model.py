#!/usr/bin/env python3
"""Standalone config-driven PINN/DeepONet training entry point."""

import json
import os
from pathlib import Path
import subprocess
import sys

from core.training import train_data_from_config as train_data
from core.training import train_physics_from_config as train_physics
from core.training_artifacts import plot_training_history as plot_training
from core.training_artifacts import finish_run

CONFIG_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run_configs/train.json")


def launch_distributed(mode):
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if (nodes <= 1 or "SLURM_PROCID" in os.environ
            or mode not in ("data", "physics")):
        return False
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "1"))
    gpus = os.environ.get("SLURM_GPUS_ON_NODE", "")
    if ":" in gpus:
        gpus = gpus.rsplit(":", 1)[-1]
    if not gpus.isdigit():
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        gpus = str(sum(line.startswith("GPU ")
                       for line in result.stdout.splitlines()))
    if int(gpus) <= 0:
        raise RuntimeError("no GPUs visible for distributed training")
    command = [
        "srun", "--nodes", str(nodes), "--ntasks", str(nodes),
        "--ntasks-per-node", "1", "--cpus-per-task", str(cpus),
        "--gpus-per-task", gpus, "--cpu-bind", "cores",
        sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
    ]
    subprocess.run(command, check=True)
    return True


if __name__ == "__main__":
    config = json.loads(CONFIG_PATH.read_text())
    mode = config.get("mode", "data")
    if launch_distributed(mode):
        raise SystemExit(0)
    try:
        if mode == "data":
            train_data(CONFIG_PATH, run_config=config)
        elif mode == "physics":
            train_physics(CONFIG_PATH, config=config)
        else:
            raise ValueError("train config mode must be 'data' or 'physics'")
        if int(os.environ.get("SLURM_PROCID", "0")) == 0:
            plot_training(CONFIG_PATH, run_config=config)
    except Exception as exc:
        config = json.loads(CONFIG_PATH.read_text())
        manifest = Path(config.get("run_dir", config["output_dir"])) / "run_manifest.json"
        if manifest.exists():
            finish_run(manifest, status="failed", error=repr(exc))
        raise
