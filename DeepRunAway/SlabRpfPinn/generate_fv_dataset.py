#!/usr/bin/env python3
"""Standalone entry point for distributed CPU-FV dataset generation.

Launch this script with one Slurm task per node. It reads a JSON configuration
and writes a durable dataset artifact. No notebook state is required.
"""

from core.fv_distributed import main as generate
from core.fv_dataset import analyze_fv_dataset as analyze
import os
from pathlib import Path
import json
import subprocess
import sys

from core.training_artifacts import finish_run


CONFIG_PATH = Path("configs/fv_dataset.json")


def launch_distributed():
    """Recursively launch one FV process per Slurm node when needed."""
    nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    if nodes <= 1 or "SLURM_PROCID" in os.environ:
        return False
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", "1"))
    command = [
        "srun", "--nodes", str(nodes), "--ntasks", str(nodes),
        "--ntasks-per-node", "1", "--cpus-per-task", str(cpus),
        "--cpu-bind", "cores", sys.executable, str(Path(__file__).resolve()),
    ]
    subprocess.run(command, check=True)
    return True


if __name__ == "__main__":
    # Rank 0 performs analytics after generation; every rank records failures
    # through the shared run manifest path when possible.
    if launch_distributed():
        raise SystemExit(0)
    config_path = CONFIG_PATH
    try:
        generate(config_path)
        if int(os.environ.get("SLURM_PROCID", "0")) == 0:
            analyze(config_path)
    except Exception as exc:
        config = json.loads(Path(config_path).read_text())
        manifest = Path(config.get("run_dir", Path(config["work_dir"]).parent))
        manifest = manifest / "run_manifest.json"
        if manifest.exists():
            finish_run(manifest, status="failed", error=repr(exc))
        raise
