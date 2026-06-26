# mcdevitt_rk4_jax_fixed_dt.py  (single cell)
import os,time,threading,queue,argparse
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import tqdm
from functools import partial
from dataclasses import dataclass
import h5py
import time

jax.config.update("jax_enable_x64", True)

plt.rcParams.update({'font.size': 25})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ---------------- Params ----------------
@jax.tree_util.register_pytree_node_class
@dataclass
class McDevittParams:
    Ebar: jnp.ndarray  # Constant electric field
    alpha: jnp.ndarray
    Z_eff: jnp.ndarray
    vte_c: jnp.ndarray
    p_thermal: jnp.ndarray
    g_max: jnp.ndarray
    def tree_flatten(self):
        return (self.Ebar, self.alpha, self.Z_eff, self.vte_c, self.p_thermal, self.g_max), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        Ebar, alpha, Z_eff, vte_c, p_thermal, g_max = children
        return cls(Ebar, alpha, Z_eff, vte_c, p_thermal, g_max)

# ---------------- Helpers ----------------
def p_from_gamma(gamma):
    return jnp.sqrt(jnp.maximum(gamma*gamma - 1.0, 0.0))


def derived_from_gamma_xi(gamma, xi, t, params: McDevittParams):
    """Return p, x, gdot, xidot_det (with your simplified CF)."""
    alpha, vte_c = params.alpha, params.vte_c
    Ebar = params.Ebar  # Constant electric field
    gamma = jnp.maximum(gamma, 1.0 + 1e-12)
    p = p_from_gamma(gamma)
    x = p / (gamma * vte_c)

    # C_F = (gamma * gamma) / (p * p + 1e-30)  # CF = γ^2 / p^2
    # gdot = (p / gamma) * (-Ebar * xi - C_F - alpha * p * gamma * (1.0 - xi*xi))
    gdot = (p / gamma) * (-Ebar * xi - alpha * p * gamma * (1.0 - xi*xi))
    xidot_det = -(Ebar / (p + 1e-30)) * (1.0 - xi*xi) + alpha * xi * (1.0 - xi*xi) / gamma
    return p, x, gdot, xidot_det

def nuD_from_gamma(gamma, params: McDevittParams):
    """Simplified ν_D = γ / p^3 * (Zeff + 1)."""
    Zeff = params.Z_eff
    gamma = jnp.maximum(gamma, 1.0 + 1e-12)
    p = p_from_gamma(gamma)
    return (gamma / (p**3 + 1e-30)) * (Zeff + 1.0)

def absorb_mask_high(gamma, params: McDevittParams):
    """Boolean mask of survivors (p >= p_th)."""
    # p = p_from_gamma(gamma)
    return (gamma >= params.g_max)
def absorb_mask_low(gamma, params: McDevittParams):
    """Boolean mask of survivors (p >= p_th)."""
    p = p_from_gamma(gamma)
    return (p <= params.p_thermal )


def reflect_xi_loop(xi_new):
    # while loop until inside [-1,1]
    def cond(x):
        return jnp.any((x > 1.0) | (x < -1.0))
    def body(x):
        x = jnp.where(x > 1.0, 1.0 - (x - 1.0), x)
        x = jnp.where(x < -1.0, -1.0 - (x + 1.0), x)
        return x
    return jax.lax.while_loop(cond, body, xi_new)

def reflect_xi(xi_new):
    # case 1: xi_new > 1
    xi_reflect_hi = 1.0 - (xi_new - 1.0)
    # case 2: xi_new < -1
    xi_reflect_lo = -1.0 - (xi_new + 1.0)

    xi_new = jnp.where(xi_new > 1.0, xi_reflect_hi, xi_new)
    xi_new = jnp.where(xi_new < -1.0, xi_reflect_lo, xi_new)
    return xi_new

# ---------------- One RK4 + scatter step with FIXED dt ----------------
# @partial(jit, static_argnames=("freeze_mode",))
@jit
def step_rk4_scatter_fixeddt_freeze(key, gamma, xi, t, dt, params: McDevittParams):
    # Ensure scalar dt with correct dtype
    dt = jnp.asarray(dt, dtype=gamma.dtype)
    t = jnp.asarray(t, dtype=gamma.dtype)
    xi_in_domain = reflect_xi(xi)
    # k1
    _, _, g1, x1 = derived_from_gamma_xi(gamma, xi, t, params)
    # k2
    g2_state = gamma + 0.5*dt*g1
    x2_state = xi    + 0.5*dt*x1
    _, _, g2, x2 = derived_from_gamma_xi(g2_state, x2_state, t + 0.5*dt, params)
    # k3
    g3_state = gamma + 0.5*dt*g2
    x3_state = xi    + 0.5*dt*x2
    _, _, g3, x3 = derived_from_gamma_xi(g3_state, x3_state, t + 0.5*dt, params)
    # k4
    g4_state = gamma + dt*g3
    x4_state = xi    + dt*x3
    _, _, g4, x4 = derived_from_gamma_xi(g4_state, x4_state, t + dt, params)

    gamma_det = gamma + (dt/6.0) * (g1 + 2*g2 + 2*g3 + g4)
    xi_det    = xi    + (dt/6.0) * (x1 + 2*x2 + 2*x3 + x4)
    # xi_det    = jnp.clip(xi_det, -1.0, 1.0)

    # ν_D at updated gamma
    nuD = nuD_from_gamma(gamma_det, params)

    # Stochastic pitch-angle kick (same Rademacher noise style you used)
    key, k_u = random.split(key)
    randu = random.uniform(k_u, shape=xi.shape)
    signs = jnp.where(randu < 0.5, -1.0, 1.0)

    one_minus_xi2 = jnp.clip(1.0 - xi_det*xi_det, 0.0, 1.0)
    sigma = jnp.sqrt(jnp.maximum(one_minus_xi2 * nuD * dt, 0.0))
    xi_new = xi_det * (1.0 - nuD * dt) + signs * sigma
    xi_new = reflect_xi(xi_new)


    p = p_from_gamma(gamma)
    v = p / gamma
    C_F = (gamma * gamma) / (p * p)
    gamma_new = gamma_det - v * C_F * dt

    # absorbing boundary (park absorbed to keep shapes static)

    dead_low = absorb_mask_low(gamma_new,params)
    gamma_new = jnp.where(dead_low, params.p_thermal,gamma_new)
    xi_new    = jnp.where(dead_low, xi_in_domain,xi_new)

    dead_high = absorb_mask_high(gamma_new, params)
    gamma_new = jnp.where(dead_high, params.g_max + 1e-1, gamma_new)
    # For frozen particles, ensure xi is constrained to [-1, 1]
    xi_frozen = jnp.clip(xi_in_domain, -1.0, 1.0)
    xi_new = jnp.where(dead_high, xi_frozen, xi_new)

    return key, gamma_new, xi_new, t + dt

@jit
def step_rk4_scatter_fixeddt(key, gamma, xi, t, dt, params: McDevittParams):
    # Ensure scalar dt with correct dtype
    dt = jnp.asarray(dt, dtype=gamma.dtype)
    t = jnp.asarray(t, dtype=gamma.dtype)
    xi_in_domain = reflect_xi(xi)
    # k1
    _, _, g1, x1 = derived_from_gamma_xi(gamma, xi, t, params)
    # k2
    g2_state = gamma + 0.5*dt*g1
    x2_state = xi    + 0.5*dt*x1
    _, _, g2, x2 = derived_from_gamma_xi(g2_state, x2_state, t + 0.5*dt, params)
    # k3
    g3_state = gamma + 0.5*dt*g2
    x3_state = xi    + 0.5*dt*x2
    _, _, g3, x3 = derived_from_gamma_xi(g3_state, x3_state, t + 0.5*dt, params)
    # k4
    g4_state = gamma + dt*g3
    x4_state = xi    + dt*x3
    _, _, g4, x4 = derived_from_gamma_xi(g4_state, x4_state, t + dt, params)

    gamma_det = gamma + (dt/6.0) * (g1 + 2*g2 + 2*g3 + g4)
    xi_det    = xi    + (dt/6.0) * (x1 + 2*x2 + 2*x3 + x4)
    # xi_det    = jnp.clip(xi_det, -1.0, 1.0)

    # ν_D at updated gamma
    nuD = nuD_from_gamma(gamma_det, params)

    # Stochastic pitch-angle kick (same Rademacher noise style you used)
    key, k_u = random.split(key)
    randu = random.uniform(k_u, shape=xi.shape)
    signs = jnp.where(randu < 0.5, -1.0, 1.0)

    one_minus_xi2 = jnp.clip(1.0 - xi_det*xi_det, 0.0, 1.0)
    sigma = jnp.sqrt(jnp.maximum(one_minus_xi2 * nuD * dt, 0.0))
    xi_new = xi_det * (1.0 - nuD * dt) + signs * sigma
    xi_new = reflect_xi(xi_new)


    p = p_from_gamma(gamma)
    v = p / gamma
    C_F = (gamma * gamma) / (p * p)
    gamma_new = gamma_det - v * C_F * dt

    # absorbing boundary (park absorbed to keep shapes static)

    dead_low = absorb_mask_low(gamma_new,params)
    gamma_new = jnp.where(dead_low, params.p_thermal,gamma_new)
    xi_new    = jnp.where(dead_low, xi_in_domain,xi_new)

    return key, gamma_new, xi_new, t + dt

# ---------------- Initialization ----------------
def initialize_particles_jax(
    key,
    N,
    *,
    init_mode="uniform",
    pmin=0.2, pmax=2.0,
    ximin=-1.0, ximax=1.0,
    p0=1.0, xi0=0.0, sigma_p=0.2, sigma_xi=0.2,
    start_id=0,
    device_unique=False,
    axis_name=None,
):
    if device_unique and axis_name is not None:
        dev_idx = lax.axis_index(axis_name)
        ids = start_id + dev_idx * N + jnp.arange(N, dtype=jnp.int32)
    else:
        ids = start_id + jnp.arange(N, dtype=jnp.int32)

    if init_mode == "uniform":
        key, kp, kx = random.split(key, 3)
        p  = random.uniform(kp, (N,)) * (pmax - pmin) + pmin
        xi = random.uniform(kx, (N,)) * (ximax - ximin) + ximin
    elif init_mode == "gaussian":
        key, kp, kx = random.split(key, 3)
        p  = random.normal(kp, (N,)) * sigma_p + p0
        # xi = jnp.abs(random.normal(kx, (N,)) * sigma_xi) + xi0
        xi = random.normal(kx, (N,)) * sigma_xi + xi0
        # xi_norm = (xi+1)/2
        # xi_norm_abs = jnp.abs(xi_norm)
        # xi_norm_rem = jnp.remainder(xi_norm_abs,2.0)
        # xi = jnp.where(xi_norm_rem>1,2-xi_norm_rem,xi_norm_rem)
        p  = jnp.clip(p,  pmin, pmax)
        # xi = jnp.clip(xi, ximin, ximax)
    elif init_mode == "delta":
        key, kp, kx = random.split(key, 3)
        p  = jnp.full((N,), jnp.asarray(p0))
        xi = jnp.full((N,), jnp.asarray(xi0))
    elif init_mode == "custom":
        key, kp, kx = random.split(key, 3)
        p  = random.normal(kp, (N,)) * sigma_p + p0
        xi = random.uniform(kx, (N,)) * (ximax - ximin) + ximin
        p  = jnp.clip(p,  pmin, pmax)
        xi = jnp.clip(xi, ximin, ximax)
    else:
        raise ValueError("init_mode must be 'uniform' or 'gaussian'")
    gamma = jnp.sqrt(1.0 + p*p)
    return key, gamma, xi, ids


def _writer_thread_fn(q: "queue.Queue", device_dir: str):
    while True:
        try:
            item = q.get(timeout=1.0)  # Add timeout to prevent hanging
            if item is None:
                q.task_done()
                break
                
            step_idx, gamma, xi, ids = item
            fname = os.path.join(device_dir, f"step_{step_idx:06d}.h5")

            with h5py.File(fname, "w") as f:
                f.create_dataset("gamma", data=gamma, compression="lzf")
                f.create_dataset("xi",    data=xi,    compression="lzf")
                f.create_dataset("ids",   data=ids,   compression="lzf")            

            q.task_done()
        except queue.Empty:
            # Continue waiting for items or termination signal
            continue
        except Exception as e:
            print(f"Writer thread error: {e}")
            break
def plot_px_scatter(E_MeV, xi, t):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(E_MeV, xi, '.', ms=0.5, alpha=0.1, color='black')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel(r'$\xi$')
    # ax.set_xlim(0,20)
    ax.set_ylim(-1, 1)
    ax.set_title(rf'$t/\tau_c = {t:.2f}$')
    fig.savefig(f'figures/jaxtest1_{int(t)}')

def plot_fn(p_host, xi_host, t):
    gamma = np.sqrt(1.0 + p_host*p_host)
    E = (gamma - 1.0) * 511e3 / 1e6  # MeV
    plot_px_scatter(E, xi_host, t)

# ---------------- Fixed-dt runner ----------------
def run_with_fixed_dt(
    *,
    key, gamma, xi, ids, params,
    dt: float,         # fixed timestep
    n_steps: int,
    t0: float = 0.0,   # initial time
    plot_every: int = 0,
    plot_fn=None,
    freeze_mode: bool = False,
):
    # JIT a single fixed-dt step to get maximum speed in the loop
    t = jnp.array(t0, dtype=gamma.dtype)
    if freeze_mode == "True":
        step_jit = jax.jit(
            lambda k, g, x, t_curr: step_rk4_scatter_fixeddt_freeze(
                k, g, x, t_curr, dt, params
            )
        )
    else:
        step_jit = jax.jit(
            lambda k, g, x, t_curr: step_rk4_scatter_fixeddt(
                k, g, x, t_curr, dt, params
            )
        )
    # Warm-up compilation
    k_tmp, g_tmp, x_tmp, t_tmp = step_jit(key, gamma, xi, t)
    g_tmp.block_until_ready(); x_tmp.block_until_ready(); t_tmp.block_until_ready()

    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    di = 0
    ddir = os.path.join(run_dir, f"device_{di}")
    os.makedirs(ddir, exist_ok=True)
    q = queue.Queue(maxsize=3)  # small buffer; tune if needed
    writer_thread = threading.Thread(target=_writer_thread_fn, args=(q, ddir), daemon=True)
    writer_thread.start()
    # Save initial state (step 0)
    q.put((0, gamma, xi, ids))
    for step in tqdm.trange(n_steps + 1):
        key, gamma, xi, t = step_jit(key, gamma, xi, t)
    
        if plot_every and plot_fn and (step % plot_every == 0):
            p_host  = np.asarray(p_from_gamma(gamma))
            xi_host = np.asarray(xi)
            t_host = float(t)
            plot_fn(p_host, xi_host, t_host)
            q.put((step, gamma, xi, ids))
    
    # Send termination signal and wait for thread to finish
    q.put(None)    
    q.join()
    writer_thread.join(timeout=5.0)  # Increased timeout for better cleanup
    
    if writer_thread.is_alive():
        print("Warning: Writer thread did not terminate cleanly")
    
    return key, gamma, xi


# ---------------- Command line arguments ----------------
def parse_args():
    parser = argparse.ArgumentParser(description='JAX particle simulation with configurable parameters')
    parser.add_argument('--Ebar', type=float, default=2.5, 
                       help='Constant electric field strength (default: 2.5)')
    parser.add_argument('--alpha', type=float, default=0.1, 
                       help='Alpha parameter (default: 0.1)')
    parser.add_argument('--Z_eff', type=float, default=2.0, 
                       help='Effective charge (default: 2.0)')
    parser.add_argument('--E0', type=float, default=8,)
    parser.add_argument('--ximin', type=float, default=0.0,)
    parser.add_argument('--ximax', type=float, default=0.0,)
    parser.add_argument('--n_steps', type=int, default=10000, 
                       help='Number of simulation steps (default: 10000)')
    parser.add_argument('--dt', type=float, default=1e-3, 
                       help='Time step size (default: 1e-3)')
    parser.add_argument('--N', type=int, default=1_000_000, 
                       help='Number of particles (default: 1000000)')
    parser.add_argument('--run_id', type=str, default='test', 
                       help='Run identifier for output directory (default: test)')
    parser.add_argument(
        '--freeze_mode',
        type=str,
        default="True",
        # action='store_true',
        help=(
            'Clamp particles at high-energy boundary (gamma >= g_max): set gamma to g_max+0.1 '
            'and freeze xi in [-1, 1] instead of letting them exceed the ceiling. '
            'Flag omitted = off (default).'
        ),
    )
    return parser.parse_args()

# ---------------- Main execution ----------------
if __name__ == "__main__":
    args = parse_args()
    
    Ebar = args.Ebar
    alpha = args.alpha
    Z_eff = args.Z_eff
    mec = 511.e3
    energy_max = 16.e6
    gamma_max=(energy_max/mec+1)

    p_thermal = np.sqrt((1+0.01/0.511)**2-1)
    params = McDevittParams(
        Ebar=jnp.array(Ebar),
        alpha=jnp.array(alpha),
        Z_eff=jnp.array(Z_eff),
        vte_c=jnp.array(0.02),
        p_thermal=jnp.array(p_thermal),##10 keV thermal energy
        g_max=jnp.array(gamma_max)
    )
    key = random.PRNGKey(123)
    base_dir: str = "./particles"
    run_id: str = args.run_id
    Energy_mid = args.E0 ##MeV
    Gamma_mid = Energy_mid/0.511+1
    p_mid = np.sqrt(Gamma_mid**2-1)

    key, gamma, xi, ids = initialize_particles_jax(
        key, N=args.N, init_mode="custom", 
        p0=p_mid, sigma_p = 3.5,pmax = 40,ximin=args.ximin, ximax=args.ximax
    )

    p = np.asarray(p_from_gamma(gamma))
    xi = np.asarray(xi)

    # Use command line arguments for dt and n_steps
    dt = jnp.array(args.dt)
    n_steps = args.n_steps

    start_time = time.time()
    key, gamma, xi = run_with_fixed_dt(
        key=key, gamma=gamma, xi=xi, ids=ids, params=params,
        dt=dt, n_steps=n_steps,
        plot_every=int(1.0/args.dt), plot_fn=plot_fn,
        freeze_mode=args.freeze_mode,
    )
    end_time = time.time()
    print('JONTA time',end_time-start_time)

    import sys
    sys.exit(0)