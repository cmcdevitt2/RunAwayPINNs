# bin_particles_cpu_cic.py
"""
Multi-device binning of saved particles → CIC f(gamma, xi) per step → TXT outputs.

Assumes generator wrote files like:
    <base_dir>/<run_id>/device_0/step_000025.h5
    <base_dir>/<run_id>/device_1/step_000025.h5
    ...

Each file contains datasets: "gamma", "xi", "ids" (ids<0 => absorbed).

We:
  - intersect available step indices across all device_* folders
  - load particles for that step from each device
  - keep only alive particles (ids >= 0)
  - CIC bin on a uniform (gamma, xi) node grid, using your exact scheme
  - save g_nodes.txt, xi_nodes.txt once and fDist_step_XXXXXX.txt per step
"""

from pathlib import Path
import re
import h5py
import numpy as np
import tqdm
import os
import glob
import argparse

# =========================
# CONFIG
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description='Bin particles from JAX simulation output')
    parser.add_argument('--run_id', type=str, default='.', 
                       help='Run identifier for input directory')
    parser.add_argument('--base_dir', type=str, default='particles', 
                       help='Base directory containing run data (default: ./particles)')
    parser.add_argument('--start_time', type=int, default=0, 
                       help='Start time for processing (default: 0.0)')
    parser.add_argument('--end_time', type=int, default=None, 
                       help='End time for processing (default: None = all available)')
    parser.add_argument('--dt', type=int, default=1000, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--Ebar', type=float, default=2.5, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--Z_eff', type=float, default=2.0, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.1, 
                       help='Time step size used in simulation (default: 1e-3)')
    parser.add_argument('--N', type=int, default=1_000_000, 
                       help='number of particles')
    return parser.parse_args()

args = parse_args()
    
# Domain parameters - 6D version (with variable Zeff and alpha)
mecSQ = 511e3

EnergyMaxeV = 16e6
EnergyMineV = 1e6
gMax = 1 + EnergyMaxeV / mecSQ
gMin = 1 + EnergyMineV / mecSQ

pMax = np.sqrt(gMax**2-1)
pMin = np.sqrt(gMin**2-1)

# Default config values
ng  =  128   # keep >1 to avoid p=0 Jacobian singularity
xiMin,xiMax,nxi = -1,1,128
# select which steps to process
STEP_STRIDE = 9999      # process every Nth common step
MAX_STEPS   = None   # None = all
# =========================


# ---------- grid & jacobian ----------
def make_grids():
    dg  = (gMax - gMin) / (ng - 1)
    dxi = (xiMax - xiMin) / (nxi - 1)
    g_nodes  = gMin + np.arange(ng,  dtype=np.float32) * dg
    xi_nodes = xiMin + np.arange(nxi, dtype=np.float32) * dxi
    return g_nodes, xi_nodes, dg, dxi

def jacobian_matrix(g_nodes, dg, dxi):
    """J[ii,jj] with edge factors 1/2 (edges) and 1/4 (corners)."""
    p_nodes = np.sqrt(np.maximum(g_nodes*g_nodes - 1.0, 0.0)).astype(np.float32)
    base = (2.0 * np.pi * p_nodes * g_nodes * dg * dxi).astype(np.float32)  # (ng,)
    edge_g  = np.ones((ng,),  np.float32); edge_g[[0, ng-1]] = 0.5
    edge_xi = np.ones((nxi,), np.float32); edge_xi[[0, nxi-1]] = 0.5
    J = (base[:, None] * edge_g[:, None] * edge_xi[None, :]).astype(np.float32)  # (ng,nxi)
    return J

# ---------- CIC binning (exactly like your working script) ----------
def cic_bin_frame(gamma, xi, g_nodes, xi_nodes, dg, dxi, J, weights=None):
    fDist = np.zeros((ng, nxi), dtype=np.float32)

    # inside open bounds & finite
    m = (
        np.isfinite(gamma) & np.isfinite(xi) &
        (gamma > gMin) & (gamma < gMax) &
        (xi > xiMin) & (xi < xiMax)
    )
    if not np.any(m):
        return fDist

    g = gamma[m].astype(np.float32)
    x = xi[m].astype(np.float32)
    w = (np.ones_like(g, np.float32) if weights is None else np.asarray(weights, np.float32)[m])

    # cell coords
    gcoord = (g - gMin) / dg
    xcoord = (x - xiMin) / dxi
    i = np.floor(gcoord).astype(np.int64)
    j = np.floor(xcoord).astype(np.int64)
    i = np.clip(i, 0, ng-2); j = np.clip(j, 0, nxi-2)
    ip1 = i + 1; jp1 = j + 1

    gi   = g_nodes[i];    gip1 = g_nodes[ip1]
    xj   = xi_nodes[j];   xjp1 = xi_nodes[jp1]

    wx_i   = 1.0 - np.abs(g - gi)   / dg
    wx_ip1 = 1.0 - np.abs(g - gip1) / dg
    wy_j   = 1.0 - np.abs(x - xj)   / dxi
    wy_jp1 = 1.0 - np.abs(x - xjp1) / dxi

    w_ij     = (wx_i   * wy_j   ) * w
    w_ijp1   = (wx_i   * wy_jp1 ) * w
    w_ip1j   = (wx_ip1 * wy_j   ) * w
    w_ip1jp1 = (wx_ip1 * wy_jp1 ) * w

    J_ij     = J[i,   j  ]
    J_ijp1   = J[i,   jp1]
    J_ip1j   = J[ip1, j  ]
    J_ip1jp1 = J[ip1, jp1]

    # scatter-add (CIC / Jacobian)
    np.add.at(fDist, (i,   j  ), w_ij     / J_ij)
    np.add.at(fDist, (i,   jp1), w_ijp1   / J_ijp1)
    np.add.at(fDist, (ip1, j  ), w_ip1j   / J_ip1j)
    np.add.at(fDist, (ip1, jp1), w_ip1jp1 / J_ip1jp1)

    return fDist  # (ng, nxi)

# ---------- utilities ----------
_step_pat = re.compile(r"step_(\d{6})\.h5$")

def discover_devices(run_dir: Path):
    dev_dirs = sorted([p for p in run_dir.glob("device_*") if p.is_dir()])
    if not dev_dirs:
        raise RuntimeError(f"No device_* folders found under {run_dir}")
    return dev_dirs

def list_steps_for_device(dev_dir: Path):
    steps = []
    for f in dev_dir.glob("step_*.h5"):
        m = _step_pat.search(f.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)

def intersect_steps(dev_dirs):
    sets = []
    for d in dev_dirs:
        steps = list_steps_for_device(d)
        if not steps:
            raise RuntimeError(f"No step_*.h5 in {d}")
        sets.append(set(steps))
    common = sorted(set.intersection(*sets))
    return common

# ---------- main ----------
def main():

    run_dir = Path(args.base_dir) / args.run_id
    out_dir = args.base_dir + "/" + args.run_id + '/particles/'
    print(f"Processing run: {run_dir}")
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    # plot_single_step_distribution(run_dir = run_dir,out_dir=outp,step=99990)
    # grids & Jacobian
    g_nodes, xi_nodes, dg, dxi = make_grids()
    J = jacobian_matrix(g_nodes, dg, dxi)

    # discover devices and common steps
    dev_dirs = discover_devices(run_dir)
    common_steps = intersect_steps(dev_dirs)
    if STEP_STRIDE > 1:
        common_steps = common_steps[::STEP_STRIDE]
    if MAX_STEPS is not None:
        common_steps = common_steps[:int(MAX_STEPS)]
    if not common_steps:
        print("No common steps found to process.")
        return

    # write grids once
    np.savetxt(outp / "g_nodes.txt",  g_nodes,  fmt="%.8e")
    np.savetxt(outp / "xi_nodes.txt", xi_nodes, fmt="%.8e")

    # loop over steps
    # for step in tqdm.tqdm(common_steps, desc="Binning"):
    #     print(step)
        # gather particles from all devices for this step
    # os.makedirs('data',exist_ok=True)
    for step in np.arange(args.start_time,args.end_time+args.dt,args.dt):
        # out_path = f'data/full_distrib_{step:06d}_E0_{args.E0}_xi0_{args.xi0}_Ebar_{args.Ebar}_Zeff_{args.Z_eff}_alpha_{args.alpha}.txt'
        out_path = f'particles/fDist_step_{step:06d}.txt'
    # step = 500_000
        gammas = []
        xis    = []
        for ddir in dev_dirs:
            fpath = ddir / f"step_{step:06d}.h5"
            with h5py.File(fpath, "r") as f:
                g  = f["gamma"][:]
                x  = f["xi"][:]
                # ids= f["ids"][:]
            # keep only alive particles
            # m_alive = (ids >= 0)
            # if m_alive.any():
                gammas.append(g)
                xis.append(x)
        if not gammas:
            # no alive particles anywhere at this step
            fDist = np.zeros((ng, nxi), dtype=np.float32)
        else:
            gamma_all = np.concatenate(gammas).astype(np.float32, copy=False)
            xi_all    = np.concatenate(xis).astype(np.float32,    copy=False)
            fDist = cic_bin_frame(gamma_all, xi_all, g_nodes, xi_nodes, dg, dxi, J)

        energy_nodes = 0.511 * (g_nodes - 1)
        z_values = fDist/args.N
        Z = z_values.reshape(len(energy_nodes), len(xi_nodes)).T  # (n_xi, n_energy)
        X, Y = np.meshgrid(energy_nodes, xi_nodes)
        
        # save per-step distribution as TXT (shape: ng rows × nxi cols)
        np.savetxt(out_path, np.column_stack((X.ravel(), Y.ravel(), Z.ravel())), header='energy_MeV xi f')

        print(f"[ok] wrote binned distributions to {outp/ f"fDist_step_{step:06d}.txt"}")

if __name__ == "__main__":
    main()