# Known issues

Bugs and defects tracked for future fix. Not operating instructions — see
`AGENTS.md` for those.

## `save_fv_dataset_directory` name collision on case-insensitive filesystems

`save_fv_dataset_directory` (`core/fv_dataset.py:231-282`) writes sibling
files `p_grid.npy` and `P_grid.npy` in the same directory via
`np.lib.format.open_memmap`. On a case-insensitive filesystem (default macOS,
Windows), these two paths resolve to the same file: `p_grid` is opened first
with shape `(n_cases, n_p)`, then `P_grid` is opened at the same path with
shape `(n_cases, n_p, n_xi)` and the full probability field, silently
clobbering the momentum-grid data. The loaded `p_grid` array ends up holding
`P_grid`'s contents (observed as `p_grid.max() == P_grid.max()` exactly, with
values in the field range, not the momentum range).

`dataset_format="npz"` (`save_fv_dataset_npz`) is unaffected — it stores both
arrays as distinct keys inside one archive, no filename collision possible.
Linux HPC clusters (case-sensitive filesystems) are also unaffected.

Repro: generate any FV dataset with `"dataset_format": "directory"` on macOS,
then `np.load(path / "p_grid.npy")` and compare against
`np.load(path / "P_grid.npy")` — they match exactly.

Not yet fixed. Workaround: use `"dataset_format": "npz"` for local
(non-cluster) FV generation, or rename the on-disk files to avoid the
case collision (e.g. `momentum_grid.npy`/`prob_grid.npy`) if a real fix is
wanted.
