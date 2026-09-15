"""Validated training configuration shared by model, PDE, and artifact code.

JSON values are grouped by concern here so kernels can use one immutable
object. Validation belongs beside the fields it protects; legacy flat JSON is
accepted only through the compatibility adapter at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PinnDomain:
    """Physical bounds mapped from normalized model coordinates in ``[0, 1]``."""
    p_min: float
    p_max: float
    B_T: float
    momentum_sampling: str = "log"
    ebar_min: float = 1.0
    ebar_max: float = 1000.0
    te_min_eV: float = 1.0
    te_max_eV: float = 100.0
    nD_min_m3: float = 1.0e20
    nD_max_m3: float = 1.0e22
    nNe_min_m3: float = 1.0e16
    nNe_max_m3: float = 1.0e22
    zD_min: float = 0.01
    zD_max: float = 1.0
    zNe_min: float = 0.01
    zNe_max: float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    """MLP or DeepONet architecture and probability-output transform."""
    model_type: str = "mlp"
    width: int = 32
    depth: int = 3
    latent_width: int = 64
    branch_width: int = 32
    branch_depth: int = 3
    trunk_width: int = 32
    trunk_depth: int = 3
    output_transform: str = "sigmoid"

    def __post_init__(self):
        if self.model_type not in ("mlp", "deeponet"):
            raise ValueError("model_type must be 'mlp' or 'deeponet'")
        if (self.width <= 0 or self.depth < 0 or self.latent_width <= 0
                or self.branch_width <= 0 or self.branch_depth < 0
                or self.trunk_width <= 0 or self.trunk_depth < 0):
            raise ValueError("invalid model architecture setting")
        from core.model import OUTPUT_TRANSFORMS
        if self.output_transform not in OUTPUT_TRANSFORMS:
            raise ValueError(
                f"output_transform must be one of {sorted(OUTPUT_TRANSFORMS)}")


@dataclass(frozen=True)
class LossConfig:
    """Enabled objective terms, weights, and stabilized PDE residual settings."""
    enable_data: bool = True
    enable_pde: bool = True
    enable_threshold_pde: bool = True
    enable_low_p_bc: bool = True
    enable_pmax_bc: bool = True
    data_weight: float = 1.0
    pde_weight: float = 1.0
    threshold_weight: float = 1.0
    low_p_weight: float = 1.0
    bc_weight: float = 1.0
    residual_coeff_norm: str = "cf_ebar"
    residual_floor: float = 0.1

    def __post_init__(self):
        if not any((self.enable_data, self.enable_pde,
                    self.enable_low_p_bc, self.enable_pmax_bc)):
            raise ValueError("at least one training loss term must be enabled")
        if self.enable_threshold_pde and not self.enable_pde:
            raise ValueError("threshold PDE loss requires enable_pde=True")
        if any(weight < 0.0 for weight in (
                self.data_weight, self.pde_weight, self.threshold_weight,
                self.low_p_weight, self.bc_weight)):
            raise ValueError("loss weights must be non-negative")
        if self.residual_coeff_norm not in ("cf_ebar", "coeff_l2", "none"):
            raise ValueError(
                "residual_coeff_norm must be 'cf_ebar', 'coeff_l2', or 'none'")
        if self.residual_floor < 0.0:
            raise ValueError("residual_floor must be non-negative")


@dataclass(frozen=True)
class OptimizerConfig:
    """SOAP training settings and optional SSBroyden refinement parameters."""
    steps: int = 1000
    learning_rate: float = 3.0e-3
    learning_rate_schedule: str = "constant"
    learning_rate_final_fraction: float = 0.1
    learning_rate_decay_steps: int = 0
    batch_size: int = 0
    case_batch_size: int = 0
    test_batch_size: int = 262144
    log_every: int = 1
    checkpoint_every: int = 0
    n_devices: int = 1
    soap_b1: float = 0.95
    soap_b2: float = 0.95
    soap_shampoo_beta: float = -1.0
    soap_eps: float = 1.0e-8
    soap_weight_decay: float = 0.0
    soap_correct_bias: bool = True
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 10000
    soap_precondition_1d: bool = False
    ssbroyden_rtol: float = 1.0e-12
    ssbroyden_atol: float = 1.0e-12
    ssbroyden_blocks: int = 5
    ssbroyden_block_iters: int = 100

    def __post_init__(self):
        if self.steps < 0 or self.learning_rate <= 0.0:
            raise ValueError("invalid optimizer setting")
        if self.learning_rate_schedule not in ("constant", "cosine"):
            raise ValueError("learning_rate_schedule must be 'constant' or 'cosine'")
        if not 0.0 <= self.learning_rate_final_fraction <= 1.0:
            raise ValueError("learning_rate_final_fraction must be in [0, 1]")
        if (self.learning_rate_decay_steps < 0 or self.batch_size < 0
                or self.case_batch_size < 0 or self.test_batch_size < 0
                or self.log_every <= 0 or self.checkpoint_every < 0
                or self.n_devices < 0):
            raise ValueError("invalid batching or logging setting")
        if (self.soap_precondition_frequency <= 0
                or self.soap_max_precond_dim <= 0):
            raise ValueError("invalid SOAP preconditioner setting")
        if self.ssbroyden_blocks < 0 or self.ssbroyden_block_iters <= 0:
            raise ValueError("invalid SSBroyden setting")
        if self.ssbroyden_rtol < 0.0 or self.ssbroyden_atol < 0.0:
            raise ValueError("SSBroyden tolerances must be non-negative")


@dataclass(frozen=True)
class DataConfig:
    """Case split and point-sampling settings for supervised or physics data."""
    train_case_fraction: float = 0.8
    train_points: int = 200000
    test_points: int = 100000
    seed: int = 2026

    def __post_init__(self):
        if not 0.0 < self.train_case_fraction < 1.0:
            raise ValueError("train_case_fraction must be between 0 and 1")
        if self.train_points < 0 or self.test_points < 0:
            raise ValueError("invalid data sampling setting")


@dataclass(frozen=True)
class CollocationConfig:
    """Counts for interior, threshold, low-momentum, and high-momentum samples."""
    pde_points: int = 200000
    threshold_points: int = 100000
    threshold_band_width: float = 0.02
    low_p_points: int = 200000
    boundary_points: int = 200000

    def __post_init__(self):
        if any(value < 0 for value in (
                self.pde_points, self.threshold_points, self.low_p_points,
                self.boundary_points)):
            raise ValueError("collocation counts must be non-negative")
        if not 0.0 < self.threshold_band_width < 1.0:
            raise ValueError("threshold_band_width must be in (0, 1)")


@dataclass(frozen=True)
class ActiveConfig:
    """Residual-guided CPU-FV acquisition settings for physics training cycles."""
    enabled: bool = False
    cycles: int = 2
    dense_points: int = 131072
    threshold_points: int = 0
    threshold_band_width: float = 0.02
    acquire_points: int = 32
    fv_Np: int = 256
    fv_Nxi: int = 64
    fv_p_stride: int = 1
    fv_xi_stride: int = 1
    fv_p_coarse_N: int = 32
    n_jobs: int = -1
    seed: int = 12026

    def __post_init__(self):
        if (self.cycles <= 0 or self.dense_points <= 0
                or self.threshold_points < 0 or self.acquire_points <= 0
                or self.fv_Np <= 0 or self.fv_Nxi <= 0
                or self.fv_p_stride <= 0 or self.fv_xi_stride <= 0
                or self.fv_p_coarse_N < 1
                or self.n_jobs == 0):
            raise ValueError("invalid active-training setting")
        if not 0.0 < self.threshold_band_width < 1.0:
            raise ValueError("active threshold_band_width must be in (0, 1)")


@dataclass(frozen=True)
class SsbroydenConfig:
    """Switch for post-SOAP SSBroyden refinement."""
    enabled: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Single hierarchical schema consumed by all model-training kernels."""
    mode: str
    domain: PinnDomain
    model: ModelConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    data: DataConfig
    collocation: CollocationConfig
    angular_sampling: str = "xi"
    active: ActiveConfig = field(default_factory=ActiveConfig)
    ssbroyden: SsbroydenConfig = field(default_factory=SsbroydenConfig)

    def __post_init__(self):
        if self.mode not in ("data", "physics"):
            raise ValueError("mode must be 'data' or 'physics'")
        if self.domain.momentum_sampling not in ("log", "linear"):
            raise ValueError("momentum_sampling must be 'log' or 'linear'")
        if self.angular_sampling not in ("xi", "theta"):
            raise ValueError("angular_sampling must be 'xi' or 'theta'")
        if self.model.model_type == "deeponet" and self.mode == "physics":
            raise ValueError("physics mode currently requires model_type='mlp'")

    @property
    def training_mode(self):
        return "physics_informed" if self.mode == "physics" else "data"

    @property
    def momentum_sampling(self):
        return self.domain.momentum_sampling

    def __getattr__(self, name):
        # Kernels keep flat attribute access while JSON remains grouped by
        # concern. Missing names must still raise normal AttributeError.
        for section in (self.model, self.loss, self.optimizer, self.data):
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(name)


PinnConfig = TrainingConfig


def _legacy_sections(run_config):
    """Partition legacy flat fields into model, loss, and optimizer sections."""
    flat = dict(run_config.get("pinn_config", {}))
    model_keys = {
        "model_type", "width", "depth", "latent_width", "branch_width",
        "branch_depth", "trunk_width", "trunk_depth",
    }
    loss_keys = {
        "enable_data", "enable_pde", "enable_threshold_pde",
        "enable_low_p_bc", "enable_pmax_bc", "data_weight", "pde_weight",
        "threshold_weight", "low_p_weight", "bc_weight",
    }
    optimizer_keys = set(flat) - model_keys - loss_keys - {
        "angular_sampling", "momentum_sampling",
    }
    return ({key: flat[key] for key in model_keys if key in flat},
            {key: flat[key] for key in loss_keys if key in flat},
            {key: flat[key] for key in optimizer_keys if key in flat})


def make_pinn_config(run_config):
    """Build schema from JSON, preferring hierarchical sections over legacy ones."""
    legacy_model, legacy_loss, legacy_optimizer = _legacy_sections(run_config)
    model = dict(run_config.get("model", legacy_model))
    loss = dict(run_config.get("loss", legacy_loss))
    optimizer = dict(run_config.get("optimizer", legacy_optimizer))
    data = dict(run_config.get("data", {}))
    collocation = dict(run_config.get("collocation", {}))
    active = dict(run_config.get("active", {}))
    mode = run_config.get("mode", "data")
    sampling = dict(run_config.get("sampling", {}))
    angular_sampling = sampling.get(
        "angular_sampling",
        run_config.get("angular_sampling",
                       run_config.get("pinn_config", {}).get(
                           "angular_sampling", "xi")))
    return TrainingConfig(
        mode=mode,
        domain=PinnDomain(**run_config["domain"]),
        angular_sampling=angular_sampling,
        model=ModelConfig(**model),
        loss=LossConfig(**loss),
        optimizer=OptimizerConfig(**optimizer),
        data=DataConfig(**data),
        collocation=CollocationConfig(**collocation),
        active=ActiveConfig(**active),
        ssbroyden=SsbroydenConfig(**run_config.get("ssbroyden", {})),
    )
