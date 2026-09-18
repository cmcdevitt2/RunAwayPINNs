"""Training-history plots."""

from __future__ import annotations

import json
import os
from pathlib import Path


def plot_training_history(config_path=Path("configs/train.json"),
                          run_config=None):
    """Plot all stored data, SOAP, and optional SSBroyden history points."""
    import numpy as np

    config = (json.loads(Path(config_path).read_text())
              if run_config is None else run_config)
    run_dir = Path(config["run_dir"])
    output_dir = Path(config.get("output_dir", run_dir))
    summary = json.loads((output_dir / "training_summary.json").read_text())
    loss_path = output_dir / "loss_history.json"
    loss_payload = (json.loads(loss_path.read_text())
                    if loss_path.exists() else {})
    traces = []
    history = np.empty((0, 0), dtype=np.float64)
    log_every = int(summary.get("log_every", 1))
    if "soap" in loss_payload:
        soap = np.asarray(loss_payload["soap"].get("values", []), dtype=np.float64)
        columns = loss_payload["soap"].get("columns", [])
        steps = np.asarray(loss_payload["soap"].get("steps", []))
        if soap.ndim == 2 and soap.size:
            if len(steps) != len(soap):
                steps = np.arange(1, len(soap) + 1)
            if len(columns) != soap.shape[1]:
                columns = [f"SOAP loss_{i}" for i in range(soap.shape[1])]
            traces.extend((steps, soap[:, i], f"SOAP {label}")
                          for i, label in enumerate(columns))
        records = []
        for item in loss_payload.get("ssbroyden", []):
            try:
                before, after = float(item["before"]), float(item["after"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(before) and np.isfinite(after):
                records.append((before, after))
        if records:
            offset = int(steps[-1]) if len(steps) else 0
            ssb_steps = offset + np.arange(1, len(records) + 1)
            traces.append((ssb_steps, np.asarray([x[1] for x in records]),
                           "SSBroyden after"))
            traces.append((ssb_steps, np.asarray([x[0] for x in records]),
                           "SSBroyden before"))
    else:
        history = np.asarray(summary.get("history", []), dtype=np.float64)
        columns = summary.get("history_columns")
        if history.ndim == 1 and history.size:
            history = history[:, None]
        if history.ndim != 2 or not history.size:
            history = np.asarray(summary.get("soap_loss_history", []), dtype=np.float64)
            if history.ndim == 2 and history.size:
                columns = ["total", "data", "PDE", "threshold PDE", "low-p", "p_max BC"]
                log_every = 1
        if history.ndim == 2 and history.size:
            columns = columns if columns and len(columns) == history.shape[1] else [
                f"loss_{i}" for i in range(history.shape[1])]
            steps = np.asarray(summary.get("history_steps", []))
            if len(steps) != len(history):
                steps = np.arange(1, len(history) + 1) * log_every
            traces.extend((steps, history[:, i], label)
                          for i, label in enumerate(columns))
    if not traces:
        raise ValueError("training summary contains no plottable loss history")
    output = Path(config.get("output", output_dir / "loss_history.png"))
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(output.parent / ".cache"))
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plot_max = 1.0e-8
    paired_history = None
    if "soap" not in loss_payload:
        history_columns = summary.get("history_columns", [])
        if {"train_loss", "test_loss"}.issubset(history_columns):
            history_steps = np.asarray(summary.get("history_steps", []))
            if len(history_steps) != len(history):
                history_steps = np.arange(1, len(history) + 1) * log_every
            paired_history = (
                history_steps,
                history[:, history_columns.index("train_loss")],
                history[:, history_columns.index("test_loss")],
            )
    if paired_history is not None:
        steps, train_loss, test_loss = paired_history
        train_line, = axis.semilogy(
            steps, np.maximum(train_loss, 1.0e-300), label="train_loss")
        axis.semilogy(
            steps, np.maximum(test_loss, 1.0e-300),
            color=train_line.get_color(), linestyle="None", marker="x",
            label="test_loss")
        plot_max = max(plot_max, float(np.max(train_loss)), float(np.max(test_loss)))
    else:
        for steps, values, label in traces:
            axis.semilogy(steps, np.maximum(values, 1.0e-300), label=label)
            plot_max = max(plot_max, float(np.max(values)))
    axis.set(xlabel="training step", ylabel="loss", title="Training history")
    axis.set_ylim(1.0e-8, max(1.0e-7, 2.0 * plot_max))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"saved: {output}")
