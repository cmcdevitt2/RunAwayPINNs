"""Plots for model-validation diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from core.pde import drift_up


def save_validation_plots(output_base, target, prediction, error, pde,
                          case_index, results, per_case, z, domain):
    """Save correlation and worst-case field plots with analytic threshold overlay."""
    import matplotlib.pyplot as plt

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_base.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    plot_count = min(len(target), 200000)
    plot_index = np.linspace(0, len(target) - 1, plot_count, dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes[0, 0].scatter(target[plot_index], prediction[plot_index],
                       c=case_index[plot_index], s=2, alpha=0.25,
                       cmap="turbo")
    axes[0, 0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 0].set(xlabel="FV P", ylabel="PINN P", title="Prediction correlation")
    axes[0, 1].scatter(target[plot_index], np.abs(error[plot_index]),
                       c=case_index[plot_index], s=2, alpha=0.25,
                       cmap="turbo")
    axes[0, 1].set(xlabel="FV P", ylabel="|PINN − FV|", title="Absolute error")
    axes[1, 0].scatter(target[plot_index], np.abs(pde[plot_index]),
                       c=case_index[plot_index], s=2, alpha=0.25,
                       cmap="turbo")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(xlabel="FV P", ylabel="|PDE residual|",
                   title="PDE residual correlation")
    axes[1, 1].scatter(np.maximum(np.abs(error[plot_index]), 1.0e-16),
                       np.maximum(np.abs(pde[plot_index]), 1.0e-16),
                       c=case_index[plot_index], s=2, alpha=0.25,
                       cmap="turbo")
    axes[1, 1].set(xscale="log", yscale="log",
                   xlabel="|PINN − FV|", ylabel="|PDE residual|",
                   title="Error/residual correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    correlation_path = output_base.with_name(output_base.name + "_correlation.png")
    fig.savefig(correlation_path, dpi=150)
    plt.close(fig)

    worst = sorted(per_case, key=lambda item: item["mse"], reverse=True)
    selected = [item["case"] for item in worst[:min(3, len(worst))]]
    fig, axes = plt.subplots(len(selected), 4,
                             figsize=(18, 4.5 * len(selected)),
                             squeeze=False, constrained_layout=True)
    for row, case_number in enumerate(selected):
        result = results[case_number]
        mask = case_index == case_number
        p = np.asarray(result["p"])
        xi = np.asarray(result["xi"])
        shape = (len(p), len(xi))
        fields = [
            np.asarray(result["P"]),
            prediction[mask].reshape(shape),
            np.abs(error[mask]).reshape(shape),
            np.abs(pde[mask]).reshape(shape),
        ]
        z_case = z[mask].reshape(shape + (9,))
        up = np.asarray(jax.device_get(
            drift_up(jnp.asarray(z_case.reshape(-1, 9)), domain)
        )).reshape(shape)
        titles = ["FV RPF", "PINN RPF", "|PINN − FV|", "|PDE residual|"]
        for column, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row, column]
            energy = 511.0e3 * (np.sqrt(1.0 + p * p) - 1.0)
            values = np.clip(field.T, 0.0, 1.0) if column < 2 else np.log10(
                np.maximum(field.T, 1.0e-16))
            mesh = axis.contourf(
                energy, xi, values, levels=50,
                cmap="turbo" if column < 2 else "magma",
                vmin=0.0 if column < 2 else None,
                vmax=1.0 if column < 2 else None)
            axis.set(xscale="log", xlabel="Energy [eV]", ylabel=r"$\xi$",
                     title=f"case {case_number}: {title}")
            axis.grid(alpha=0.25)
            axis.contour(
                energy, xi, up.T, levels=[0.0], colors="white",
                linewidths=1.0)
            if column == 0:
                axis.plot([], [], color="white", label=r"$U_p=0$")
                axis.legend(loc="best", fontsize="small")
            fig.colorbar(mesh, ax=axis, pad=0.02)
    cases_path = output_base.with_name(output_base.name + "_cases.png")
    fig.savefig(cases_path, dpi=150)
    plt.close(fig)
    return correlation_path, cases_path
