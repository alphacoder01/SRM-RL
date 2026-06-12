from dataclasses import dataclass, field

from jaxtyping import Bool, Float, Int64
import torch
from torch import Tensor


@dataclass
class RolloutBatch:
    """A batch of recorded SRM sampling trajectories for GRPO updates.

    B trajectories of S steps each over P patches. Trajectory tensors live on
    a (typically CPU) storage device; conditioning tensors stay small.
    """

    # --- trajectory ---
    z_seq: Float[Tensor, "frames batch dim height width"]   # post-repaint latents, z_seq[0] = initial state
    t_patch: Float[Tensor, "steps batch patches"]            # noise level read at each step
    t_next_patch: Float[Tensor, "steps batch patches"]       # noise level after each step
    old_logp: Float[Tensor, "steps batch"]                   # behavior log-prob, mean over active scalars (0 if inactive)
    active: Bool[Tensor, "steps batch"]                      # whether the step has any active pixel

    # --- conditioning ---
    mask: Float[Tensor, "batch 1 height width"]              # 1 = unknown / to generate
    masked: Float[Tensor, "batch dim height width"]          # (1 - mask) * image

    # --- order decisions (one event per (step, batch) where a new patch was scheduled) ---
    order_event_idx: Int64[Tensor, "steps batch"]            # index into event tensors, -1 if no event
    order_patch: Int64[Tensor, "events"]                     # chosen patch id (top_k == 1)
    order_unknown: Bool[Tensor, "events patches"]            # unknown-patch snapshot at decision time
    order_old_logp: Float[Tensor, "events"]                  # behavior categorical log-prob (0 if greedy)

    # --- results (filled after reward computation) ---
    sample: Float[Tensor, "batch dim height width"] | None = None
    reward: Float[Tensor, "batch"] | None = None
    advantage: Float[Tensor, "batch"] | None = None
    group_ids: Int64[Tensor, "batch"] | None = None
    metrics: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return self.t_patch.shape[0]

    @property
    def num_trajectories(self) -> int:
        return self.t_patch.shape[1]

    def compute_group_advantages(
        self,
        group_ids: Int64[Tensor, "batch"],
        eps: float = 1.e-4,
        skip_degenerate: bool = True
    ) -> None:
        """Group-relative advantages: (r - mean_g) / (std_g + eps).

        Groups with (near-)zero reward std carry no GRPO signal; their advantage
        is set to 0 (skip_degenerate), which removes them from the objective.
        """
        assert self.reward is not None
        reward, device = self.reward, self.reward.device
        group_ids = group_ids.to(device)
        num_groups = int(group_ids.max().item()) + 1
        cnt = torch.zeros(num_groups, device=device).scatter_add_(
            0, group_ids, torch.ones_like(reward)
        )
        mean = torch.zeros(num_groups, device=device).scatter_add_(
            0, group_ids, reward
        ) / cnt
        var = torch.zeros(num_groups, device=device).scatter_add_(
            0, group_ids, (reward - mean[group_ids]) ** 2
        ) / cnt
        std = var.sqrt()
        advantage = (reward - mean[group_ids]) / (std[group_ids] + eps)
        if skip_degenerate:
            advantage = torch.where(std[group_ids] < eps, torch.zeros_like(advantage), advantage)
        self.group_ids = group_ids
        self.advantage = advantage
        self.metrics["degenerate_group_frac"] = (std < eps).float().mean().item()
