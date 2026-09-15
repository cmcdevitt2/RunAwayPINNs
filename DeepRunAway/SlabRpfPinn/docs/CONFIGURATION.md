# Configuration reference

The workflow has exactly three active JSON configuration files. Root scripts
read them directly; they do not accept runtime CLI overrides.

## `run_configs/fv_dataset.json`

### `parameter_domain`

Each parameter maps to `[minimum, maximum, scale]`. `scale` is either `log` or
`linear` and is used when mapping scrambled Sobol samples to physical values.
The six parameters are `E/Ec`, `Te_eV`, `nD_m3`, `nNe_m3`, `zD`, and `zNe`.

### `dataset_config`

- `n_cases`: number of accepted nontrivial FV cases to save.
- `p_max`: maximum normalized momentum for every FV solve.
- `B_T`: magnetic field in tesla.
- `fv_Np`, `fv_Nxi`: fine FV momentum and angular cell counts before coarsening.
- `fv_p_stride`, `fv_xi_stride`: exact cell-center strides saved to disk.
- `n_jobs`: Joblib workers per CPU rank; `-1` derives a safe count from Slurm.
- `seed`: scrambled Sobol seed.
- `candidate_oversample`: candidate multiplier used while rejecting trivial-zero
  cases.

### Dataset paths

- `dataset_path`: final directory-backed dataset.
- `dataset_format`: `directory` for memory-mapped arrays or `npz` for the
  compatibility format.
- `work_dir`: shared per-rank intermediate files.
- `run_dir`: dataset manifest and automatic analytics.

## `run_configs/train.json`

### Top-level mode and domains

- `mode`: `data` for supervised MLP/DeepONet training or `physics` for the
  pointwise MLP PDE objective.
- `parameter_domain`: must match the FV dataset parameter domain.
- `domain`: model normalization and physical-domain bounds. `p_min`, `p_max`,
  and `B_T` must be compatible with the FV dataset.

### `model`

- `model_type`: `mlp` or `deeponet`.
- `width`, `depth`: pointwise MLP width and hidden-layer count.
- `latent_width`: DeepONet latent size.
- `branch_width`, `branch_depth`: DeepONet parameter-branch architecture.
- `trunk_width`, `trunk_depth`: DeepONet phase-space-trunk architecture.
- `output_transform`: named scheme mapping raw network output to `P∈(0,1)`.
  `sigmoid` (default) is unconstrained; `structural_lowp` architecturally
  enforces `P(p_min,ξ)=0` via `P=tanh(p̂²·raw²)` instead of relying only on the
  `enable_low_p_bc` loss term.

### `loss`

- `enable_data`, `enable_pde`, `enable_threshold_pde`, `enable_low_p_bc`, and
  `enable_pmax_bc`: enable the corresponding objective terms.
- `data_weight`, `pde_weight`, `threshold_weight`, `low_p_weight`, and
  `bc_weight`: nonnegative term weights.
- `residual_coeff_norm`: named scheme rescaling the PDE coefficient vector
  before the residual is formed (separate from the `dp_dpnorm`
  change-of-variables Jacobian, which is always applied and not configurable).
  `cf_ebar` (default) divides by `|cf|·√ēbar`; `coeff_l2` divides by the
  per-point coefficient-vector L2 norm; `none` disables rescaling.
- `residual_floor`: additive floor `P + residual_floor` in the PDE residual
  denominator, guarding against blow-up near `P→0`. Must be `>= 0`.

### `optimizer`
- `steps`: SOAP optimizer steps.
- `learning_rate`: initial learning rate.
- `learning_rate_schedule`: `constant` or `cosine`.
- `learning_rate_final_fraction`: cosine schedule endpoint fraction.
- `batch_size`: point batch size for MLP/physics training; zero means full data.
- `case_batch_size`: DeepONet case batch size; zero derives it from
  `batch_size`.
- `test_batch_size`: held-out points/cases used for periodic metrics.
- `log_every`: history readback interval.
- `checkpoint_every`: parameter checkpoint interval; zero disables checkpoints.
- `n_devices`: local GPUs; zero means all visible local GPUs.
- `soap_*`: SOAP optimizer hyperparameters.
- `ssbroyden_*`: SSBroyden tolerances, block count, and block iterations.

### `sampling`

- `sampling.angular_sampling`: `xi` or uniform-`theta` collocation sampling.
- `domain.momentum_sampling`: `linear` or `log` normalized momentum coordinate.

Physics mode requires `model.model_type: "mlp"`. Data mode supports both models.

### `data`

- `train_case_fraction`: fraction of parameter cases used for training; the
  split is by case, never by individual FV cell.
- `train_points`, `test_points`: point counts used by the physics-data split.
- `seed`: deterministic case split and sampling seed.
- `low_p_Np`: optional number of synthetic zero-target points inserted between
  the global `p_min` and each adaptive FV case minimum for DeepONet regularization.
  This is a training option, not an FV-generation option.

### `collocation`

- `pde_points`: general Sobol PDE points.
- `threshold_points`: points sampled around the analytic physical `U_p=0` curve.
- `threshold_band_width`: normalized width around that curve.
- `low_p_points`: low-momentum boundary points.
- `boundary_points`: upper-momentum boundary points.

### Other paths and phases

- `ssbroyden.enabled`: run SSBroyden after SOAP; in multi-node mode rank 0
  performs refinement and broadcasts the parameters.
- `active.enabled`: run residual-guided FV acquisition cycles after each SOAP
  cycle. This currently requires one JAX process, but may use all local GPUs.
- `active.cycles`: number of train/acquire cycles.
- `active.dense_points`: Sobol PDE points evaluated for residual ranking.
- `active.threshold_points`: optional analytic `U_p=0`-curve candidates added
  to the residual-ranking pool.
- `active.threshold_band_width`: normalized width around that analytic curve.
- `active.acquire_points`: highest-residual points used to choose parameter
  cases for new FV labels.
- `active.fv_Np`, `active.fv_Nxi`: CPU FV resolution for acquired cases.
- `active.fv_p_stride`, `active.fv_xi_stride`: exact FV coarsening strides.
- `active.n_jobs`: CPU workers for acquired FV cases.
- `active.seed`: Sobol seed for dense residual candidates.
- `initial_model`: optional saved parameter file used to restart/refine a
  physics-informed model. Set `steps` to zero to perform SSBroyden-only
  refinement from that model.
- `dataset_path`: saved FV dataset to load.
- `dataset_manifest`: completed FV manifest used for checksum and compatibility
  validation.
- `output_dir`: final model, metadata, history, and checkpoints.
- `checkpoint_dir`: periodic parameter snapshots.
- `run_dir`: training manifest and run ownership.

Use fresh `output_dir`, `checkpoint_dir`, and `run_dir` values for every run.

## `run_configs/validate_model.json`

- `model_dir`: saved model directory containing parameters and metadata.
- `model_manifest`: optional completed training manifest used to verify the
  saved model checksum.
- `output_dir`: validation JSON and plots.
- `plot`: plot basename; produces correlation and selected-case figures.
- `dataset_path`: optional saved FV dataset. If set, `dataset_manifest` is
  required.
- `dataset_manifest`: checksum/compatibility manifest for `dataset_path`.
- `cases`: number of new Sobol validation cases when no saved dataset is used.
- `fv_Np`, `fv_Nxi`: CPU FV resolution for fresh validation cases.
- `pde_chunk`: maximum JAX PDE-residual batch size.
- `n_jobs`: CPU workers for fresh FV generation.
- `seed`: validation Sobol seed.
