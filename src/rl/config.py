from dataclasses import dataclass, field

from omegaconf import DictConfig

from ..config import RootCfg, load_typed_config
from .reward import SudokuRewardCfg


@dataclass
class RolloutCfg:
    # conditions (masked Sudokus) per iteration PER GPU (like the repo's
    # data_loader batch_size); each gets group_size rollouts. Too few
    # conditions -> highly correlated advantages -> noisy gradients
    num_conditions: int = 8
    group_size: int = 8                     # G
    # number of cells given to the model: [lo, hi] inclusive; [0, 26] = hard
    num_fill: list[int] = field(default_factory=lambda: [0, 26])
    # denoising reduction: reduced step budget for training rollouts
    max_steps: int = 243
    top_k: int = 1
    overlap: float = 0.0
    alpha: float = 1.0                      # must be > 0 (stochastic sampling)
    temperature: float = 1.0
    storage_device: str = "cpu"
    storage_dtype: str = "float32"
    progress_bar: bool = False              # tqdm over rollout denoising steps
    compile: bool = False                   # torch.compile the rollout denoiser forward


@dataclass
class OrderPolicyCfg:
    # Stage 2: stochastic categorical order policy (cf. Decisions.md D3, D23)
    enabled: bool = False
    # temperature in units of the sigma spread across candidate patches
    # (logits are standardized, D23). Small -> near-greedy; ~1 -> explore within
    # ~1 std of the greedy choice. Verify iter-0 train acc is close to eval.
    temperature: float = 0.5
    loss_weight: float = 1.0


@dataclass
class UpdateCfg:
    lr: float = 3.e-5
    weight_decay: float = 0.0
    inner_epochs: int = 2
    # fraction of each trajectory's active steps trained per inner epoch
    step_fraction: float = 0.25
    # force order-decision steps into the subsample (relevant for Stage 2)
    include_order_steps: bool = True
    # (trajectory, step) pairs per forward/backward microbatch. Pure throughput
    # knob: gradient accumulation (below) already averages microbatches, so this
    # does not change the optimization, only GPU utilization and peak memory.
    # This is the primary OOM knob. The RL update holds more per-sample memory
    # than pretraining (KL reference forward + Gaussian tensors + three model
    # copies), so the safe ceiling is below pretraining's batch 28: 32 OOM'd on
    # A100-80GB, 16 is the safe default (still ~2x fewer kernels than the
    # original 8). Lower to 8 if OOM, raise toward 24 if there is headroom.
    update_batch_size: int = 16
    # gradients are accumulated so each inner epoch takes only this many
    # optimizer steps; stepping per microbatch makes AdamW take hundreds of
    # noise-driven steps per rollout batch and the policy drifts off the
    # pretrained manifold (cf. Decisions.md D16). Raised 4 -> 8 once D18 showed
    # the policy was barely moving (KL ~0.001) with the conservative defaults.
    optimizer_steps_per_epoch: int = 8
    clip_range: float = 1.e-2               # calibrated for per-scalar mean log-probs
    kl_beta: float = 0.04                   # 0 disables the reference forward pass (initial value if adaptive)
    # adaptive KL controller (cf. Decisions.md D24): if set, kl_beta is nudged to
    # keep the measured KL near kl_target, preventing drift past the good region
    kl_target: float | None = None
    kl_adapt_rate: float = 1.5
    kl_beta_min: float = 1.e-3
    kl_beta_max: float = 1.0
    noise_aware_weighting: bool = False
    adv_eps: float = 1.e-4
    skip_degenerate_groups: bool = True
    grad_clip: float = 1.0
    # auxiliary supervised losses on dataset batches (cf. Decisions.md D3/D12/D13)
    sigma_aux_weight: float = 0.01          # matches pretraining loss.sigma.weight
    flow_anchor_weight: float = 0.1
    aux_batch_size: int = 8
    aux_batches_per_epoch: int = 1


@dataclass
class RLEvalCfg:
    every: int = 25                         # iterations between evals (0 disables)
    num_samples: int = 128                  # larger -> less noisy eval (SE ~ 0.5/sqrt(n))
    batch_size: int = 16
    num_fill: list[int] = field(default_factory=lambda: [0, 26])
    max_steps: int = 1000                   # full budget at eval time
    top_k: int = 1
    overlap: float = 0.0
    alpha: float = 1.0
    use_ema: bool = True
    progress_bar: bool = True               # tqdm over eval batches and denoising steps
    dump_samples: bool = False              # write per-sample results for paired comparison (D25)


@dataclass
class RLCfg:
    # path to pretrained checkpoint: .ckpt (Lightning) or .pth (state_dict)
    pretrained_checkpoint: str
    num_iterations: int = 1000
    checkpoint_every: int = 25
    progress_bar: bool = True       # tqdm bar over training iterations (rank 0)
    # "bf16": denoiser forwards in bfloat16 autocast during rollouts/updates
    # (Gaussian transition math stays float32); "32": full precision
    precision: str = "bf16"
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    update: UpdateCfg = field(default_factory=UpdateCfg)
    order_policy: OrderPolicyCfg = field(default_factory=OrderPolicyCfg)
    reward: SudokuRewardCfg = field(default_factory=SudokuRewardCfg)
    eval: RLEvalCfg = field(default_factory=RLEvalCfg)


@dataclass
class RLRootCfg(RootCfg):
    rl: RLCfg = None


def load_typed_rl_config(cfg: DictConfig) -> RLRootCfg:
    return load_typed_config(cfg, RLRootCfg)
