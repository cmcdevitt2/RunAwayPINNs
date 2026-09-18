"""Plots for finite-volume dataset analytics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def plot_fv_coverage(dataset_path, output_path):
    """Save parameter-coverage and adaptive-grid summary plots."""
    from core.fv_dataset import load_fv_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    dataset = load_fv_dataset(dataset_path)
    cases = np.asarray(dataset["cases"], dtype=np.float64)
    fields = dataset["P_grid"]
    metadata = json.loads(str(np.asarray(dataset["case_metadata_json"]).item()))
    p_dense_min = np.asarray(
        [item["p_dense_min"] for item in metadata], dtype=np.float64)
    runtime = np.asarray(
        [item["runtime_seconds"] for item in metadata], dtype=np.float64)
    # Scan the large probability array in chunks.  Keep the 10 GB dataset
    # memory-mapped instead of copying it into plotting-process memory.
    zero_fraction = np.empty(len(cases), dtype=np.float64)
    for start in range(0, len(cases), 512):
        stop = min(start + 512, len(cases))
        zero_fraction[start:stop] = np.mean(
            fields[start:stop] == 0.0, axis=(1, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes[0, 0].hist(p_dense_min, bins=40)
    axes[0, 0].set(xscale="log", xlabel="adaptive dense-zone front", ylabel="cases")
    axes[0, 1].hist(runtime, bins=40)
    axes[0, 1].set(xlabel="runtime [s/case]", ylabel="cases")
    axes[0, 2].scatter(cases[:, 0], cases[:, 1], s=4, alpha=0.35)
    axes[0, 2].set(xscale="log", yscale="log", xlabel="E/Ec", ylabel="Te [eV]")
    axes[1, 0].scatter(cases[:, 2], cases[:, 3], s=4, alpha=0.35)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="nD", ylabel="nNe")
    axes[1, 1].scatter(cases[:, 4], cases[:, 5], s=4, alpha=0.35)
    axes[1, 1].set(xlabel="zD", ylabel="zNe")
    axes[1, 2].scatter(p_dense_min, zero_fraction, s=4, alpha=0.35)
    axes[1, 2].set(xscale="log", xlabel="adaptive dense-zone front",
                   ylabel="zero-P fraction")
    for axis in axes.ravel():
        axis.grid(alpha=0.25)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_representative_rpf(dataset_path, output_path, case_indices=None):
    """Save representative clipped RPF fields from a saved FV dataset."""
    from core.fv_dataset import load_fv_dataset

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    dataset = load_fv_dataset(dataset_path)
    cases = np.asarray(dataset["cases"], dtype=np.float64)
    p_grid = dataset["p_grid"]
    xi_grid = dataset["xi_grid"]
    fields = dataset["P_grid"]

    if case_indices is None:
        # Pick distinct fields spanning below-threshold through saturated.
        maxima = np.empty(len(cases), dtype=np.float64)
        zero_fraction = np.empty(len(cases), dtype=np.float64)
        for start in range(0, len(cases), 256):
            stop = min(start + 256, len(cases))
            block = fields[start:stop]
            maxima[start:stop] = np.max(block, axis=(1, 2))
            zero_fraction[start:stop] = np.mean(block == 0.0, axis=(1, 2))
        below = np.flatnonzero(maxima <= 1.0e-12)
        selected = [int(below[0])]
        targets = np.array([0.8, 0.5, 0.3, 0.2, 0.15, 0.1, 0.085, 0.08])
        eligible = np.flatnonzero(maxima > 1.0e-12)
        for target in targets:
            order = eligible[np.argsort(np.abs(zero_fraction[eligible] - target))]
            selected.append(next(index for index in order
                                  if int(index) not in selected))
        case_indices = selected
    case_indices = [int(index) for index in case_indices]

    levels = np.linspace(0.0, 1.0, 21)
    norm = BoundaryNorm(levels, ncolors=plt.get_cmap("turbo").N,
                        clip=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True,
                             sharey=True, constrained_layout=True)
    axes = axes.ravel()
    for axis, index in zip(axes, case_indices):
        p = np.asarray(p_grid[index], dtype=np.float64)
        xi = np.asarray(xi_grid[index], dtype=np.float64)
        energy = 511.0e3 * (np.sqrt(1.0 + p * p) - 1.0)
        probability = np.clip(np.asarray(fields[index], dtype=np.float64),
                              0.0, 1.0)
        axis.contourf(energy, xi, probability.T, levels=levels,
                      cmap="turbo", norm=norm)
        axis.set_xscale("log")
        axis.set_ylim(-1.0, 1.0)
        axis.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        axis.grid(False)
        E_over_Ec, te, nD, nNe, zD, zNe, B = cases[index]
        axis.set_title(
            f"case {index}: max P={probability.max():.3f}, zero-P={np.mean(probability == 0.0):.2f}\n"
            f"E/Ec={E_over_Ec:.3g}, Te={te:.3g} eV, B={B:.3g} T; "
            f"zD={zD:.3g}, zNe={zNe:.3g}",
            fontsize=9)
    for axis in axes[6:]:
        axis.set_xlabel("Energy [eV]")
    for axis in axes[::3]:
        axis.set_ylabel(r"$\xi$")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="turbo"), ax=axes,
        ticks=np.linspace(0.0, 1.0, 9), shrink=0.82)
    colorbar.set_label("Clipped RPF $P$")
    fig.suptitle("Representative FV runaway-probability regimes")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
