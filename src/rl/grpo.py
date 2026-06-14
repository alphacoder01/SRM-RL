from contextlib import nullcontext
from copy import deepcopy
import json
from math import ceil, prod
from pathlib import Path

from jaxtyping import Bool, Float, Int64
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import avg_pool2d, interpolate, mse_loss
from tqdm import tqdm

from ..dataset import Dataset
from ..model import Wrapper
from ..model.denoiser import Denoiser
from ..model.diagonal_gaussian import DiagonalGaussian
from ..sampler.sequential_adaptive_sampler import (
    SequentialAdaptiveSampler,
    SequentialAdaptiveSamplerCfg,
)
from .config import RLRootCfg
from .distributed import (
    all_reduce_grads,
    all_reduce_mean_scalars,
    get_rank,
    get_world_size,
)
from .reward import SudokuReward
from .rollout import TrajectoryRecordingSampler, TrajectoryRecordingSamplerCfg
from .trajectory import RolloutBatch


class GRPOTrainer:
    """Flow-GRPO-style RL fine-tuning of a pretrained SRM (cf. Decisions.md).

    Plain PyTorch loop: collect grouped rollouts -> verify -> group-relative
    advantages -> PPO-clipped updates on exact Gaussian step log-probs, with a
    KL anchor to a frozen reference and supervised auxiliary losses keeping the
    uncertainty head calibrated.
    """

    def __init__(
        self,
        cfg: RLRootCfg,
        model: Wrapper,
        train_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: Path,
        use_wandb: bool = False,
    ) -> None:
        assert model.cfg.patch_size is not None, \
            "RL fine-tuning is only supported for patch-based SRMs (cf. Decisions.md D14)"
        assert model.cfg.model.learn_sigma, \
            "Uncertainty-ordered sampling requires a model trained with learn_sigma"
        assert cfg.rl.rollout.alpha > 0
        self.cfg = cfg
        self.rl = cfg.rl
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.device = next(model.parameters()).device
        self.image_shape = tuple(cfg.dataset.image_shape)
        self.iteration = 0
        self._pbar = None       # set in fit() on rank 0 when progress_bar is on
        self._last_eval_acc = float("nan")
        self.rank, self.world_size = get_rank(), get_world_size()
        if self.rank != 0:
            self.rl.eval.progress_bar = False
            self.rl.rollout.progress_bar = False
        self.autocast_dtype = (
            torch.bfloat16
            if self.rl.precision == "bf16" and self.device.type == "cuda"
            else None
        )

        # Policy starts from the EMA weights; reference = frozen copy of them
        if model.ema_denoiser is not None:
            model.denoiser.load_state_dict(model.ema_denoiser.module.state_dict())
        self.ref_denoiser: Denoiser = deepcopy(model.denoiser)
        self.ref_denoiser.requires_grad_(False)
        self.ref_denoiser.eval()

        # RL forwards run in eval mode (no dropout) so that behavior log-probs,
        # recomputed log-probs, and the reference share deterministic means
        self.model.eval()

        self.rollout_sampler = TrajectoryRecordingSampler(
            TrajectoryRecordingSamplerCfg(
                name="trajectory_recording",
                max_steps=self.rl.rollout.max_steps,
                alpha=self.rl.rollout.alpha,
                temperature=self.rl.rollout.temperature,
                use_ema=False,  # behavior policy == trained policy
                top_k=self.rl.rollout.top_k,
                overlap=self.rl.rollout.overlap,
                order_temperature=(
                    self.rl.order_policy.temperature
                    if self.rl.order_policy.enabled else None
                ),
                storage_device=self.rl.rollout.storage_device,
                storage_dtype=self.rl.rollout.storage_dtype,
                progress_bar=self.rl.rollout.progress_bar,
                autocast_dtype="bfloat16" if self.autocast_dtype is not None else None,
                compile=self.rl.rollout.compile,
            ),
            patch_size=model.cfg.patch_size,
            patch_grid_shape=model.patch_grid_size,
        )
        self.eval_sampler = SequentialAdaptiveSampler(
            SequentialAdaptiveSamplerCfg(
                name="sequential_adaptive",
                max_steps=self.rl.eval.max_steps,
                alpha=self.rl.eval.alpha,
                use_ema=self.rl.eval.use_ema,
                top_k=self.rl.eval.top_k,
                overlap=self.rl.eval.overlap,
                progress_bar=self.rl.eval.progress_bar,
            ),
            patch_size=model.cfg.patch_size,
            patch_grid_shape=model.patch_grid_size,
        )
        self.reward_fn = SudokuReward(self.rl.reward, cfg.mnist_classifier)
        self.optimizer = torch.optim.AdamW(
            self.model.denoiser.parameters(),
            lr=self.rl.update.lr,
            weight_decay=self.rl.update.weight_decay,
        )
        self.metrics_path = output_dir / "metrics.jsonl"
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # adaptive KL: self.kl_beta is mutable; if update.kl_target is set it is
        # nudged toward keeping the measured KL near the target (cf. D24)
        self.kl_beta = self.rl.update.kl_beta
        self._best_eval_acc = -float("inf")

    # ---------------------------------------------------------------- helpers

    def _autocast(self):
        """Autocast for denoiser forwards; Gaussian math stays float32 by
        casting network outputs back (cf. Decisions.md D17)."""
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def _patch_to_pixel(
        self,
        t_rows: Float[Tensor, "batch num_patches"]
    ) -> Float[Tensor, "batch 1 height width"]:
        t = t_rows.reshape(-1, *self.model.patch_grid_size)
        return interpolate(t.unsqueeze(1), size=self.image_shape, mode="nearest-exact")

    def _predict(
        self,
        denoiser: Denoiser,
        z_t: Float[Tensor, "batch 1 dim height width"],
        t: Float[Tensor, "batch 1 1 height width"],
    ) -> tuple[Tensor, Tensor | None]:
        """Wrapper.forward for an explicit denoiser module (used for the frozen
        reference); returns mean (model parameterization) and v_theta."""
        model_cfg = self.model.cfg.model
        with self._autocast():
            pred = denoiser.forward(z_t, t, None)
        pred = pred.float()
        d = self.model.d_data
        mean_theta = pred[..., :d, :, :]
        mean_theta = getattr(self.model.flow, f"get_{model_cfg.parameterization}")(
            t, zt=z_t, **{model_cfg.denoiser_parameterization: mean_theta}
        )
        v_theta = (
            (pred[..., d:2 * d, :, :] + 1) / 2
            if model_cfg.flow.variance == "learned_range" else None
        )
        return mean_theta, v_theta

    def _sigma_to_patch(
        self,
        sigma_theta: Float[Tensor, "batch 1 1 height width"]
    ) -> Float[Tensor, "batch num_patches"]:
        patch_sigma = avg_pool2d(
            sigma_theta.squeeze(1),
            kernel_size=self.model.cfg.patch_size,
            count_include_pad=False,
        )
        return patch_sigma.reshape(-1, prod(self.model.patch_grid_size))

    @staticmethod
    def _masked_mean(
        values: Float[Tensor, "batch dim height width"],
        active: Bool[Tensor, "batch 1 height width"],
    ) -> Float[Tensor, "batch"]:
        """Mean over active scalar dimensions; values may have 1 or d_data channels."""
        total = (values * active).flatten(1).sum(dim=1)
        count = values.shape[1] * active.flatten(1).sum(dim=1)
        return total / count.clamp(min=1)

    def _log(self, record: dict) -> dict:
        """Average numeric metrics across ranks, then log on rank 0.
        Must be called by all ranks (collective). Returns the reduced record."""
        record = all_reduce_mean_scalars(record, self.device)
        if self.rank != 0:
            return record
        record = {"iteration": self.iteration, **record}
        line = json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in record.items()})
        # write through the bar so log lines scroll above it instead of corrupting it
        if self._pbar is not None:
            self._pbar.write(line)
        else:
            print(line)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self.use_wandb:
            import wandb
            phase = record.get("phase", "train")
            wandb.log(
                {f"{phase}/{k}": v for k, v in record.items() if isinstance(v, (int, float))},
                step=self.iteration,
            )
        return record

    # ------------------------------------------------------------ data access

    def _collate_conditions(
        self,
        indices: list[int],
        num_given_cells: list[int] | None = None,
        dataset: Dataset | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (image, mask, masked) on the training device."""
        dataset = self.train_dataset if dataset is None else dataset
        items = [
            dataset.__getitem__(
                idx,
                **({} if num_given_cells is None else {"num_given_cells": num_given_cells[i]})
            )
            for i, idx in enumerate(indices)
        ]
        image = torch.stack([item["image"] for item in items]).to(self.device)
        mask = torch.stack([item["mask"] for item in items]).to(self.device)
        return image, mask, (1 - mask) * image

    # -------------------------------------------------------------- rollouts

    @torch.no_grad()
    def collect(self) -> RolloutBatch:
        roll_cfg = self.rl.rollout
        lo, hi = roll_cfg.num_fill
        indices = np.random.randint(0, len(self.train_dataset), roll_cfg.num_conditions)
        num_given = np.random.randint(lo, hi + 1, roll_cfg.num_conditions)
        _, mask, masked = self._collate_conditions(indices.tolist(), num_given.tolist())

        group_ids = torch.arange(roll_cfg.num_conditions).repeat_interleave(roll_cfg.group_size)
        mask = mask.repeat_interleave(roll_cfg.group_size, dim=0)
        masked = masked.repeat_interleave(roll_cfg.group_size, dim=0)
        z_init = torch.randn(
            (mask.size(0), self.model.d_data, *self.image_shape), device=self.device
        )

        roll = self.rollout_sampler.sample_trajectory(self.model, z_init, mask, masked)
        scores = self.reward_fn(roll.sample.to(self.device))
        roll.reward = scores["reward"]
        roll.compute_group_advantages(
            group_ids,
            eps=self.rl.update.adv_eps,
            skip_degenerate=self.rl.update.skip_degenerate_groups,
        )
        roll.metrics.update({
            "reward_mean": scores["reward"].mean().item(),
            "accuracy": scores["accuracy"].mean().item(),
            "distance": scores["distance"].mean().item(),
            "active_steps_per_traj": roll.active.float().sum(dim=0).mean().item(),
        })
        return roll

    # ---------------------------------------------------------------- update

    def _grpo_microbatch_loss(
        self,
        roll: RolloutBatch,
        b_idx: Int64[Tensor, "n"],
        s_idx: Int64[Tensor, "n"],
        stats: dict,
    ) -> Float[Tensor, ""]:
        upd, d_data = self.rl.update, self.model.d_data
        z_t = roll.z_seq[s_idx, b_idx].to(self.device, torch.float32)
        z_next = roll.z_seq[s_idx + 1, b_idx].to(self.device, torch.float32)
        t = self._patch_to_pixel(roll.t_patch[s_idx, b_idx].to(self.device, torch.float32))
        t_next = self._patch_to_pixel(roll.t_next_patch[s_idx, b_idx].to(self.device, torch.float32))
        old_logp = roll.old_logp[s_idx, b_idx].to(self.device, torch.float32)
        advantage = roll.advantage[b_idx.to(roll.advantage.device)].to(self.device)
        active = (t > t_next).logical_and(t_next > 0)

        with self._autocast():
            mean_theta, v_theta, sigma_theta = self.model.forward(
                z_t.unsqueeze(1), t.unsqueeze(1), sample=True, use_ema=False
            )
        mean_theta = mean_theta.float()
        sigma_theta = sigma_theta.float() if sigma_theta is not None else None
        # Transition variance is frozen during RL (Decisions.md D4)
        v_detached = v_theta.detach().float() if v_theta is not None else None
        p_new = self.model.flow.conditional_p(
            mean_theta, z_t.unsqueeze(1), t.unsqueeze(1), t_next.unsqueeze(1),
            self.rl.rollout.alpha, self.rl.rollout.temperature, v_theta=v_detached
        )
        logp_map = -p_new.nll(z_next.unsqueeze(1)).squeeze(1)
        logp_new = self._masked_mean(logp_map, active)
        ratio = torch.exp(logp_new - old_logp)
        clipped = ratio.clamp(1 - upd.clip_range, 1 + upd.clip_range)
        surrogate = torch.minimum(ratio * advantage, clipped * advantage)

        if upd.noise_aware_weighting:
            with torch.no_grad():
                weight = self._masked_mean(p_new.std.squeeze(1), active)
                weight = weight / weight.mean().clamp(min=1.e-8)
            surrogate = weight * surrogate

        loss = -surrogate.mean()
        stats["ratio"].append(ratio.detach().mean().item())
        stats["clip_frac"].append(
            ((ratio - 1).abs() > upd.clip_range).float().mean().item()
        )
        stats["pg_loss"].append(loss.item())

        if self.kl_beta > 0 or upd.kl_target is not None:
            with torch.no_grad():
                ref_mean, ref_v = self._predict(
                    self.ref_denoiser, z_t.unsqueeze(1), t.unsqueeze(1)
                )
                p_ref = self.model.flow.conditional_p(
                    ref_mean, z_t.unsqueeze(1), t.unsqueeze(1), t_next.unsqueeze(1),
                    self.rl.rollout.alpha, self.rl.rollout.temperature, v_theta=ref_v
                )
            kl_map = p_new.kl(p_ref).squeeze(1)
            kl = self._masked_mean(kl_map, active).mean()
            loss = loss + self.kl_beta * kl
            stats["kl"].append(kl.detach().item())

        if self.rl.order_policy.enabled:
            event_ids = roll.order_event_idx[s_idx, b_idx]
            has_event = event_ids >= 0
            if has_event.any():
                rows = has_event.nonzero(as_tuple=True)[0]
                events = event_ids[rows]
                patch_sigma = self._sigma_to_patch(sigma_theta)[rows.to(self.device)]
                logp_order = TrajectoryRecordingSampler.order_logp(
                    patch_sigma,
                    roll.order_unknown[events].to(self.device),
                    roll.order_patch[events].to(self.device),
                    self.rl.order_policy.temperature,
                )
                old_logp_order = roll.order_old_logp[events].to(self.device, torch.float32)
                adv_order = advantage[rows.to(self.device)]
                ratio_o = torch.exp(logp_order - old_logp_order)
                clipped_o = ratio_o.clamp(1 - upd.clip_range, 1 + upd.clip_range)
                surrogate_o = torch.minimum(ratio_o * adv_order, clipped_o * adv_order)
                order_loss = -surrogate_o.mean()
                loss = loss + self.rl.order_policy.loss_weight * order_loss
                stats["order_loss"].append(order_loss.detach().item())

        return loss

    def _aux_loss(self, stats: dict) -> Float[Tensor, ""] | None:
        """Supervised sigma-NLL (keeps the uncertainty head calibrated) and
        optional flow-matching anchor on real dataset batches (Decisions.md D3/D13)."""
        upd = self.rl.update
        if upd.sigma_aux_weight <= 0 and upd.flow_anchor_weight <= 0:
            return None
        model = self.model
        indices = np.random.randint(0, len(self.train_dataset), upd.aux_batch_size)
        image, mask, _ = self._collate_conditions(indices.tolist())

        t, weight = model.time_sampler(image.size(0), 1, self.device)
        t = torch.kron(t, model.float_kernel)
        weight = torch.kron(weight, model.float_kernel)
        if model.cfg.conditioning.mask:
            t = t * mask
            weight = weight * mask
        x = image.unsqueeze(1)
        t = t.unsqueeze(2).expand_as(x[..., :1, :, :])
        weight = weight.unsqueeze(2)
        eps = model.flow.sample_eps(x)
        z_t = model.flow.get_zt(t, eps=eps, x=x)

        with self._autocast():
            mean_theta, _, sigma_theta = model.forward(z_t, t, sample=True, use_ema=False)
        mean_theta = mean_theta.float()
        sigma_theta = sigma_theta.float() if sigma_theta is not None else None
        if model.cfg.model.parameterization == "eps":
            target = eps
        else:
            target = model.flow.get_ut(t, eps=eps, x=x)

        loss = torch.zeros((), device=self.device)
        if upd.sigma_aux_weight > 0 and sigma_theta is not None:
            pred_theta = DiagonalGaussian(mean_theta.detach(), std=sigma_theta)
            sigma_loss = (weight * pred_theta.nll(target)).mean()
            loss = loss + upd.sigma_aux_weight * sigma_loss
            stats["sigma_aux"].append(sigma_loss.detach().item())
        if upd.flow_anchor_weight > 0:
            anchor = (weight * mse_loss(mean_theta, target, reduction="none")).mean()
            loss = loss + upd.flow_anchor_weight * anchor
            stats["flow_anchor"].append(anchor.detach().item())
        return loss

    def _apply_optimizer_step(self) -> None:
        """Sync (multi-GPU) + clip + step + EMA update on accumulated gradients."""
        all_reduce_grads(self.model.denoiser)
        if self.rl.update.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.denoiser.parameters(), self.rl.update.grad_clip
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.model.ema_denoiser is not None:
            self.model.ema_denoiser.update_parameters(self.model.denoiser)

    def _build_pairs(self, roll: RolloutBatch) -> tuple[Tensor, Tensor]:
        """Timestep subsampling (Decisions.md D8): per trajectory, a random
        fraction of its active steps; order-decision steps optionally forced in.
        Zero-advantage trajectories carry no signal and are dropped."""
        upd = self.rl.update
        advantage = roll.advantage.cpu()
        b_list, s_list = [], []
        for b in range(roll.num_trajectories):
            if advantage[b].abs() < 1.e-12:
                continue
            steps = roll.active[:, b].nonzero(as_tuple=True)[0]
            if steps.numel() == 0:
                continue
            num = ceil(upd.step_fraction * steps.numel())
            chosen = steps[torch.randperm(steps.numel())[:num]]
            if upd.include_order_steps and self.rl.order_policy.enabled:
                order_steps = (roll.order_event_idx[:, b] >= 0).nonzero(as_tuple=True)[0]
                chosen = torch.unique(torch.cat([chosen, order_steps]))
            b_list.append(torch.full_like(chosen, b))
            s_list.append(chosen)
        if not b_list:
            return torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64)
        return torch.cat(b_list), torch.cat(s_list)

    def update(self, roll: RolloutBatch) -> dict:
        upd = self.rl.update
        stats = {k: [] for k in (
            "ratio", "clip_frac", "pg_loss", "kl", "order_loss", "sigma_aux", "flow_anchor"
        )}
        num_pairs = 0
        num_optimizer_steps = 0
        for _ in range(upd.inner_epochs):
            b_idx, s_idx = self._build_pairs(roll)
            num_pairs += b_idx.numel()
            perm = torch.randperm(b_idx.numel())
            b_idx, s_idx = b_idx[perm], s_idx[perm]
            # Accumulate microbatch gradients into few optimizer steps per epoch:
            # AdamW normalizes gradient scale, so taking a step per microbatch
            # amounts to hundreds of noise-driven steps per rollout batch and
            # makes the policy drift off the pretrained manifold (Decisions.md D16).
            # Every rank performs exactly optimizer_steps_per_epoch (+1 aux) steps
            # regardless of its local microbatch count, keeping the multi-GPU
            # gradient all-reduces in lockstep (Decisions.md D17).
            #
            # Chunk by PAIR index (not microbatch count) so each optimizer step
            # sees a fixed set of pairs and its gradient is exactly the grand
            # mean over that set, independent of update_batch_size. This makes
            # update_batch_size a pure throughput knob (Decisions.md D19).
            pair_chunks = np.array_split(
                np.arange(b_idx.numel()), upd.optimizer_steps_per_epoch
            )
            self.optimizer.zero_grad(set_to_none=True)
            for pchunk in pair_chunks:
                n = len(pchunk)
                if n == 0:
                    self._apply_optimizer_step()       # keep ranks in lockstep
                    num_optimizer_steps += 1
                    continue
                for start in range(0, n, upd.update_batch_size):
                    sub = torch.from_numpy(pchunk[start : start + upd.update_batch_size])
                    loss = self._grpo_microbatch_loss(
                        roll, b_idx[sub], s_idx[sub], stats
                    )
                    # weight by the microbatch's pair fraction so the chunk's
                    # accumulated gradient is the exact grand mean over n pairs
                    (loss * len(sub) / n).backward()
                self._apply_optimizer_step()
                num_optimizer_steps += 1
            aux = self._aux_loss(stats)
            if aux is not None:
                aux.backward()
                self._apply_optimizer_step()
                num_optimizer_steps += 1
        result = {k: float(np.mean(v)) for k, v in stats.items() if v}
        result["num_pairs"] = num_pairs
        result["optimizer_steps"] = num_optimizer_steps
        # Adaptive KL controller: nudge kl_beta to keep the measured KL near the
        # target, preventing the unbounded drift that over-optimizes past the
        # good region (cf. Decisions.md D24). Disabled when kl_target is None.
        if upd.kl_target is not None and "kl" in result:
            ratio_to_target = result["kl"] / max(upd.kl_target, 1.e-8)
            if ratio_to_target > 1.5:
                self.kl_beta = min(self.kl_beta * upd.kl_adapt_rate, upd.kl_beta_max)
            elif ratio_to_target < 1 / 1.5:
                self.kl_beta = max(self.kl_beta / upd.kl_adapt_rate, upd.kl_beta_min)
        result["kl_beta"] = self.kl_beta
        return result

    # ------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Each rank evaluates a strided shard of the sample indices; the
        cross-rank average happens in _log. Returns local-shard means.
        If eval.dump_samples is set, also writes per-sample (index, accuracy,
        distance) to output_dir for paired comparison across runs (D25)."""
        ev = self.rl.eval
        lo, hi = ev.num_fill
        all_scores = []
        per_sample = []     # (index, accuracy, distance) for dumping
        shard = list(range(self.rank, ev.num_samples, self.world_size))
        batch_starts = range(0, len(shard), ev.batch_size)
        if ev.progress_bar:
            batch_starts = tqdm(
                batch_starts,
                desc=f"eval ({ev.num_samples} samples, {ev.max_steps} steps)",
                unit="batch",
            )
        for start in batch_starts:
            indices = shard[start : start + ev.batch_size]
            num_given = [
                int(np.random.default_rng(idx).integers(lo, hi + 1)) for idx in indices
            ]
            _, mask, masked = self._collate_conditions(
                [idx % len(self.test_dataset) for idx in indices],
                num_given,
                dataset=self.test_dataset,
            )
            z_init = torch.stack([
                torch.randn(
                    (self.model.d_data, *self.image_shape),
                    generator=torch.Generator(self.device).manual_seed(idx),
                    device=self.device,
                ) for idx in indices
            ])
            out = self.eval_sampler(self.model, z_t=z_init, mask=mask, masked=masked)
            scores = self.reward_fn(out["sample"])
            all_scores.append(scores)
            if ev.dump_samples:
                for j, idx in enumerate(indices):
                    per_sample.append((
                        idx, scores["accuracy"][j].item(), scores["distance"][j].item()
                    ))
            if ev.progress_bar:
                acc = torch.cat([s["accuracy"] for s in all_scores])
                batch_starts.set_postfix(acc=f"{acc.mean().item():.3f}", n=acc.numel())
        if ev.dump_samples:
            self._dump_eval_samples(per_sample)
        return {
            key: torch.cat([s[key] for s in all_scores]).mean().item()
            for key in all_scores[0]
        }

    def _dump_eval_samples(self, per_sample: list) -> None:
        """Gather per-sample eval results across ranks and write them on rank 0,
        keyed by the (deterministic) sample index so two runs can be compared
        per-puzzle (paired McNemar test, cf. src/rl/paired_eval.py)."""
        if self.world_size > 1:
            import torch.distributed as dist
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, per_sample)
            if self.rank != 0:
                return
            per_sample = [row for part in gathered for row in part]
        per_sample.sort(key=lambda r: r[0])
        path = self.output_dir / f"eval_samples_it{self.iteration}.jsonl"
        with path.open("w") as f:
            for idx, acc, dist in per_sample:
                f.write(json.dumps({"index": idx, "correct": int(acc), "distance": dist}) + "\n")
        print(f"Wrote {len(per_sample)} per-sample eval results to {path}")

    # ---------------------------------------------------------- checkpointing

    def save_checkpoint(self) -> None:
        if self.rank != 0:
            return
        state = {
            "iteration": self.iteration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_beta": self.kl_beta,
            "best_eval_acc": self._best_eval_acc,
        }
        torch.save(state, self.checkpoint_dir / "rl_state.pt")
        # plain state_dict for the existing test pipeline (checkpointing.load=*.pth)
        torch.save(self.model.state_dict(), self.checkpoint_dir / "policy_latest.pth")

    def maybe_save_best(self, eval_acc: float) -> bool:
        """Keep the best-by-eval-accuracy policy so transient peaks are not lost
        when the policy later drifts (cf. Decisions.md D24). Rank 0 only."""
        if eval_acc <= self._best_eval_acc:
            return False
        self._best_eval_acc = eval_acc
        if self.rank == 0:
            torch.save(self.model.state_dict(), self.checkpoint_dir / "policy_best.pth")
        return True

    def load_checkpoint(self) -> bool:
        path = self.checkpoint_dir / "rl_state.pt"
        if not path.exists():
            return False
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state["iteration"]
        self.kl_beta = state.get("kl_beta", self.kl_beta)
        self._best_eval_acc = state.get("best_eval_acc", self._best_eval_acc)
        return True

    # ------------------------------------------------------------------ loop

    def fit(self) -> None:
        if self.load_checkpoint():
            print(f"Resumed RL training from iteration {self.iteration}")
        if self.rank == 0:
            rc = self.rl.rollout
            global_conditions = rc.num_conditions * self.world_size
            print(
                f"[GRPO] world_size={self.world_size} | "
                f"num_conditions/GPU={rc.num_conditions} -> global={global_conditions} | "
                f"group_size={rc.group_size} | rollout batch/GPU={rc.num_conditions * rc.group_size} | "
                f"global rollouts/iter={global_conditions * rc.group_size}. "
                f"More GPUs only speed up an ITERATION if you keep global conditions "
                f"fixed by lowering num_conditions/GPU (cf. Decisions.md D21)."
            )
        if self.rank == 0 and self.rl.progress_bar:
            self._pbar = tqdm(
                total=self.rl.num_iterations,
                initial=self.iteration,
                desc="GRPO training",
                unit="it",
                dynamic_ncols=True,
            )
        while self.iteration < self.rl.num_iterations:
            if self.rl.eval.every > 0 and self.iteration % self.rl.eval.every == 0:
                eval_rec = self._log({"phase": "eval", **self.evaluate()})
                self._last_eval_acc = eval_rec.get("accuracy", float("nan"))
                self.maybe_save_best(self._last_eval_acc)
            roll = self.collect()
            # Rollout and update run sequentially; return the rollout's cached
            # allocator blocks so the update gets maximal contiguous headroom
            # (guards against fragmentation OOM; cf. Decisions.md D20).
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            update_stats = self.update(roll)
            train_rec = self._log({
                "phase": "train",
                **roll.metrics,
                "advantage_abs": roll.advantage.abs().mean().item(),
                **update_stats,
            })
            self.iteration += 1
            if self._pbar is not None:
                self._pbar.update(1)
                self._pbar.set_postfix(
                    acc=f"{train_rec.get('accuracy', float('nan')):.3f}",
                    rew=f"{train_rec.get('reward_mean', float('nan')):.3f}",
                    kl=f"{train_rec.get('kl', float('nan')):.4f}",
                    eval_acc=f"{self._last_eval_acc:.3f}",
                    refresh=False,
                )
            if self.iteration % self.rl.checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint()
        if self.rl.eval.every > 0:
            final_eval = self._log({"phase": "eval", **self.evaluate()})
            self.maybe_save_best(final_eval.get("accuracy", float("nan")))
        if self._pbar is not None:
            self._pbar.close()
