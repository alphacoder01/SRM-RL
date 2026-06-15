from contextlib import nullcontext
from dataclasses import dataclass
from math import prod
from typing import Literal

from jaxtyping import Bool, Float, Int64
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import avg_pool2d
from tqdm import tqdm

from ..model import Wrapper
from ..sampler.sequential_adaptive_sampler import (
    SequentialAdaptiveSampler,
    SequentialAdaptiveSamplerCfg,
)
from .trajectory import RolloutBatch


@dataclass
class TrajectoryRecordingSamplerCfg(SequentialAdaptiveSamplerCfg):
    name: Literal["trajectory_recording"] = "trajectory_recording"
    # > 0 turns the order selection into a stochastic categorical policy over
    # unknown patches with logits -sigma_patch / order_temperature (Stage 2);
    # None / 0 keeps the deterministic greedy argmin (Stage 1).
    order_temperature: float | None = None
    storage_device: str = "cpu"
    storage_dtype: str = "float32"
    # autocast dtype for the denoiser forward (e.g. "bfloat16"); the Gaussian
    # transition math always runs in float32 so log-probs stay precise
    autocast_dtype: str | None = None
    # use the torch.compile'd denoiser forward (rollout batch shape is fixed,
    # so compilation amortizes over max_steps network evaluations)
    compile: bool = False


class TrajectoryRecordingSampler(SequentialAdaptiveSampler):
    """SequentialAdaptiveSampler that records full trajectories for GRPO.

    Mirrors SequentialAdaptiveSampler.sample step by step, additionally recording
    per step: patch-level t / t_next actually used, post-repaint latents, behavior
    log-probs over active pixels, and order-decision events.
    """

    cfg: TrajectoryRecordingSamplerCfg

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Stochastic transitions are required for exact Gaussian log-probs
        assert self.cfg.alpha > 0, "GRPO rollouts require stochastic sampling (alpha > 0)"
        if self.stochastic_order:
            assert self.cfg.top_k == 1, "Stochastic order policy requires top_k == 1"

    @property
    def stochastic_order(self) -> bool:
        return self.cfg.order_temperature is not None and self.cfg.order_temperature > 0

    @staticmethod
    def standardized_order_logits(
        patch_sigma: Float[Tensor, "batch num_patches"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
        temperature: float,
        eps: float = 1.e-6,
    ) -> Float[Tensor, "batch num_patches"]:
        """Scale-invariant per-patch order logits.

        sigma_theta has an arbitrary absolute scale, so a raw `-sigma/temperature`
        makes `temperature` un-interpretable and checkpoint-dependent (cf. the
        Stage-2 collapse, Decisions.md D23). We standardize sigma across the
        *unknown* (candidate) patches per batch element, so `temperature` is in
        units of the sigma spread among candidates: small -> near-greedy, ~1 ->
        explore within roughly one std of the greedy choice. Known patches -> -inf.
        """
        # guard against non-finite sigma (e.g. exp overflow of an extreme
        # log-variance); applied identically in rollout and update so the
        # ratio==1 property is preserved
        patch_sigma = torch.nan_to_num(patch_sigma, nan=0.0, posinf=1.e4, neginf=0.0)
        mask = is_unknown_map.to(patch_sigma.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (patch_sigma * mask).sum(dim=-1, keepdim=True) / count
        var = (((patch_sigma - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / count
        std = var.sqrt().clamp(min=eps)
        # detach the standardization stats: they are a per-decision shift/scale
        # for temperature interpretation, not parameters to differentiate. This
        # also removes the sqrt(var) backward, which is +inf when var==0 (e.g.
        # a decision with a single candidate patch) and would NaN the policy.
        score = (patch_sigma - mean.detach()) / std.detach()
        # bound the standardized score: when the candidate sigmas are nearly
        # equal (std -> 0) the 1/std factor makes the per-logit gradient explode
        # (it NaN'd the order-KL anchor, which sums that gradient over all
        # patches). Clamping gives zero gradient in that degenerate region while
        # leaving the normal case (|score| <~ few) untouched. Applied identically
        # in rollout and update, so the ratio==1 property is preserved (D26).
        score = score.clamp(-8.0, 8.0)
        logits = -score / temperature
        # mask with a large finite negative (not -inf): exp(-1e9)=0 so sampling
        # and log_prob are identical to -inf masking, but KL(cur||ref) over the
        # categoricals stays finite in the BACKWARD pass (-inf gives 0*(-inf+inf)
        # = NaN gradients that blew up the order-KL anchor, D26).
        return logits.masked_fill(~is_unknown_map, -1.e9)

    def select_next_patches(
        self,
        sigma_theta: Float[Tensor, "batch d_data height width"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
    ) -> tuple[
        Int64[Tensor, "batch top_k"],
        Float[Tensor, "batch"]
    ]:
        """Order action. Returns chosen patch ids and the categorical log-prob
        of the choice (zeros for the deterministic greedy policy)."""
        if not self.stochastic_order:
            ids = self.get_next_patch_ids(sigma_theta, is_unknown_map)
            return ids, torch.zeros(ids.shape[0], device=ids.device)
        total_patches = prod(self.patch_grid_shape)
        patch_sigma = avg_pool2d(
            sigma_theta, kernel_size=self.patch_size, count_include_pad=False
        ).reshape(-1, total_patches)
        logits = self.standardized_order_logits(
            patch_sigma, is_unknown_map, self.cfg.order_temperature
        )
        dist = Categorical(logits=logits)
        ids = dist.sample()
        return ids.unsqueeze(1), dist.log_prob(ids)

    @staticmethod
    def order_logp(
        patch_sigma: Float[Tensor, "batch num_patches"],
        is_unknown_map: Bool[Tensor, "batch num_patches"],
        patch_ids: Int64[Tensor, "batch"],
        temperature: float
    ) -> Float[Tensor, "batch"]:
        """Categorical log-prob of given order decisions (used for recomputation
        under the current policy during GRPO updates). Must use the SAME
        standardization as select_next_patches for ratio==1 at epoch start."""
        logits = TrajectoryRecordingSampler.standardized_order_logits(
            patch_sigma, is_unknown_map, temperature
        )
        return Categorical(logits=logits).log_prob(patch_ids)

    @staticmethod
    def active_scalar_logp(
        logp_map: Float[Tensor, "batch dim height width"],
        active: Bool[Tensor, "batch 1 height width"]
    ) -> tuple[
        Float[Tensor, "batch"],
        Int64[Tensor, "batch"]
    ]:
        """Mean log-prob over active scalar dimensions (pixels x channels);
        0 for trajectories without active pixels at this step."""
        d_data = logp_map.shape[-3]
        masked_logp = (logp_map * active).flatten(1).sum(dim=1)
        count = d_data * active.flatten(1).sum(dim=1)
        return masked_logp / count.clamp(min=1), count

    @torch.no_grad()
    def sample_trajectory(
        self,
        model: Wrapper,
        z_t: Float[Tensor, "batch dim height width"],
        mask: Float[Tensor, "batch 1 height width"],
        masked: Float[Tensor, "batch dim height width"],
        label: Int64[Tensor, "batch"] | None = None,
    ) -> RolloutBatch:
        image_shape = z_t.shape[-2:]
        total_patches = prod(self.patch_grid_shape)
        batch_size = z_t.size(0)
        device = z_t.device
        storage = dict(
            device=torch.device(self.cfg.storage_device),
            dtype=getattr(torch, self.cfg.storage_dtype)
        )
        eps_threshold = self.cfg.epsilon

        assert model.cfg.patch_size is not None, "RL rollouts require a patch-based SRM"
        if not model.cfg.conditioning.label:
            label = None
        autocast_ctx = (
            torch.autocast(device.type, dtype=getattr(torch, self.cfg.autocast_dtype))
            if self.cfg.autocast_dtype else nullcontext()
        )

        # Conditioning by repainting known patches (cf. Sampler.get_defaults)
        z_t = masked + mask * z_t

        is_unknown_map = self.full_mask_to_sequence_mask(mask) > 0.5
        scheduling_matrix = torch.ones(
            [self.cfg.max_steps + 1, batch_size, total_patches], device=device
        )
        scheduling_matrix *= is_unknown_map.unsqueeze(0)

        num_unknown_patches = is_unknown_map.sum(dim=1).long()
        num_inference_blocks = torch.ceil(num_unknown_patches / self.cfg.top_k).int()
        ideal_block_lengths = self.get_inference_lengths(num_inference_blocks)
        block_lengths = ideal_block_lengths.ceil().int()
        block_starts = (
            torch.arange(num_inference_blocks.max() + 1, device=device).unsqueeze(0)
            * ideal_block_lengths.unsqueeze(1) * (1 - self.cfg.overlap)
        ).floor_()
        block_starts[:, -1] = -1    # This extra block should never be used!
        block_counters = torch.zeros(batch_size, device=device, dtype=torch.int64)
        step_targets = torch.zeros(batch_size, device=device, dtype=torch.int64)

        prototypes = self.get_schedule_prototypes(block_lengths)

        # Recording buffers
        z_seq = [z_t.to(**storage)]
        t_patch_seq, t_next_patch_seq = [], []
        old_logp_seq, active_seq = [], []
        order_event_idx = torch.full(
            (self.cfg.max_steps, batch_size), -1, dtype=torch.int64
        )
        order_patch, order_unknown, order_old_logp = [], [], []

        num_steps_done = 0
        step_iterator = range(self.cfg.max_steps)
        if self.cfg.progress_bar:
            step_iterator = tqdm(step_iterator, desc="rollout steps", leave=False)
        for step_id in step_iterator:
            # snapshot before the in-place block update below: exactly what the
            # forward pass sees (the update is a no-op on this row, but explicit
            # beats relying on that invariant, cf. Decisions.md D6)
            t_patch = scheduling_matrix[step_id].to(**storage, copy=True)
            t = self.get_timestep_from_schedule(scheduling_matrix, step_id, image_shape)

            with autocast_ctx:
                # sample=False routes through the compiled denoiser (and always
                # uses the raw weights, which is what RL rollouts want anyway)
                mean_theta, v_theta, sigma_theta = model.forward(
                    z_t=z_t.unsqueeze(1),
                    t=t.unsqueeze(1),
                    label=label,
                    c_cat=None,
                    sample=not self.cfg.compile,
                    use_ema=self.cfg.use_ema
                )
            mean_theta = mean_theta.float()
            v_theta = v_theta.float() if v_theta is not None else None
            sigma_theta = sigma_theta.float().squeeze(1)
            should_predict = step_targets == step_id

            if is_unknown_map.sum() > eps_threshold and should_predict.any():
                block_counters += should_predict.int()
                step_targets = block_starts[torch.arange(batch_size, device=device), block_counters]

                should_predict_batch_ids = should_predict.nonzero(as_tuple=True)[0]

                next_ids, decision_logp = self.select_next_patches(
                    sigma_theta[should_predict_batch_ids],
                    is_unknown_map[should_predict_batch_ids]
                )

                # Record order events (one per predicting batch element)
                for row, batch_id in enumerate(should_predict_batch_ids.tolist()):
                    order_event_idx[step_id, batch_id] = len(order_patch)
                    order_patch.append(next_ids[row, 0].item())
                    order_unknown.append(is_unknown_map[batch_id].to("cpu", copy=True))
                    order_old_logp.append(decision_logp[row].item())

                repeat_batch_ids = torch.repeat_interleave(
                    should_predict_batch_ids, repeats=next_ids.shape[1]
                )
                repeat_prototypes = torch.repeat_interleave(
                    prototypes[:, should_predict_batch_ids], repeats=next_ids.shape[1], dim=1
                )
                flat_next_ids = next_ids.flatten()
                is_unknown_map[repeat_batch_ids, flat_next_ids] = False

                length_to_consider = min(repeat_prototypes.shape[0], self.cfg.max_steps - step_id)
                scheduling_matrix[
                    step_id : step_id + length_to_consider,
                    repeat_batch_ids,
                    flat_next_ids,
                ] = torch.minimum(
                    repeat_prototypes[:length_to_consider],
                    scheduling_matrix[
                        step_id : step_id + length_to_consider,
                        repeat_batch_ids,
                        flat_next_ids,
                    ],
                )
                if step_id + length_to_consider < self.cfg.max_steps:
                    scheduling_matrix[
                        step_id + length_to_consider:, repeat_batch_ids, flat_next_ids
                    ] = 0
                scheduling_matrix[-1] = 0

            t_next_patch = scheduling_matrix[step_id + 1].to(**storage, copy=True)
            t_next = self.get_timestep_from_schedule(
                scheduling_matrix, step_id + 1, image_shape
            )

            conditional_p = model.flow.conditional_p(
                mean_theta, z_t.unsqueeze(1), t.unsqueeze(1), t_next.unsqueeze(1),
                self.cfg.alpha, self.cfg.temperature, v_theta=v_theta
            )
            # no noise when t_next == 0
            z_next = torch.where(
                t_next.unsqueeze(1) > 0, conditional_p.sample(), conditional_p.mean
            ).squeeze(1)
            # Repaint known patches (does not touch active pixels)
            z_next = masked + mask * z_next

            # Behavior log-prob over active pixels: strictly denoised this step
            # and not the deterministic final sub-step to t_next == 0 (cf. D5)
            active = (t > t_next).logical_and(t_next > 0)
            logp_map = -conditional_p.nll(z_next.unsqueeze(1)).squeeze(1)
            logp, _ = self.active_scalar_logp(logp_map, active)

            t_patch_seq.append(t_patch)
            t_next_patch_seq.append(t_next_patch)
            old_logp_seq.append(logp.to(**storage))
            active_seq.append(active.flatten(1).any(dim=1).to(storage["device"]))
            z_seq.append(z_next.to(**storage))

            z_t = z_next
            num_steps_done = step_id + 1
            if t_next.max() <= eps_threshold:
                break

        if self.cfg.progress_bar:
            step_iterator.close()

        return RolloutBatch(
            z_seq=torch.stack(z_seq),
            t_patch=torch.stack(t_patch_seq),
            t_next_patch=torch.stack(t_next_patch_seq),
            old_logp=torch.stack(old_logp_seq),
            active=torch.stack(active_seq),
            mask=mask.to(**storage),
            masked=masked.to(**storage),
            order_event_idx=order_event_idx[:num_steps_done],
            order_patch=torch.tensor(order_patch, dtype=torch.int64),
            order_unknown=(
                torch.stack(order_unknown) if order_unknown
                else torch.zeros(0, total_patches, dtype=torch.bool)
            ),
            order_old_logp=torch.tensor(order_old_logp, dtype=storage["dtype"]),
            sample=z_t.to(**storage),
        )
