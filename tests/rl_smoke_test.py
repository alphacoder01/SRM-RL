"""CPU smoke test for the GRPO RL pipeline (no datasets/checkpoints required).

Builds a tiny randomly initialized SRM and verifies:
 1. trajectory recording (shapes, active masking, schedule consistency),
 2. log-prob recomputation consistency: with unchanged weights, recomputed
    step log-probs match the rollout-time behavior log-probs (ratio == 1),
 3. a full GRPO update step changes the policy (Stage 1 and Stage 2),
 4. auxiliary supervised losses and the eval path run end to end.

Run from the repo root: DEBUG=false python -m tests.rl_smoke_test
"""
from types import SimpleNamespace

import numpy as np
import torch

# Newer torchvision releases dropped torchvision.io.write_video, which the
# repo's LocalLogger imports at module level; shim it for the test environment.
import torchvision.io
if not hasattr(torchvision.io, "write_video"):
    torchvision.io.write_video = lambda *args, **kwargs: None

from src.model.denoiser.class_embedding.parameters import ClassEmbeddingParametersCfg
from src.model.denoiser.embedding.sinusodial import EmbeddingSinusodialCfg
from src.model.denoiser.unet import UNetDenoiserCfg
from src.model.flow.rectified_flow import RectifiedFlowCfg
from src.model.time_sampler.mean_beta import MeanBetaCfg
from src.model.wrapper import (
    LossCfg, ModelCfg, OptimizerCfg, TrainCfg, Wrapper, WrapperCfg
)
from src.rl.config import OrderPolicyCfg, RLCfg, RLEvalCfg, RolloutCfg, UpdateCfg
from src.rl.grpo import GRPOTrainer
from src.type_extensions import ConditioningCfg

IMAGE_SHAPE = (28, 28)
PATCH_SIZE = 7
GRID = (4, 4)


class SyntheticGridDataset:
    """Minimal stand-in for the Sudoku dataset: random images + per-cell masks."""

    num_classes = None

    def __len__(self) -> int:
        return 64

    def __getitem__(self, idx: int, num_given_cells: int | None = None) -> dict:
        rng = np.random.default_rng(idx)
        image = torch.from_numpy(
            rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype(np.float32)
        )
        num_cells = GRID[0] * GRID[1]
        if num_given_cells is None:
            num_given_cells = int(rng.integers(0, num_cells))
        mask = np.ones(GRID, dtype=np.float32)
        given = rng.choice(num_cells, num_given_cells, replace=False)
        mask[given // GRID[1], given % GRID[1]] = 0
        mask = np.kron(mask, np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
        return {"index": idx, "image": image, "mask": torch.from_numpy(mask)[None]}


class RandomReward:
    def __call__(self, samples: torch.Tensor) -> dict:
        n = samples.size(0)
        reward = torch.randn(n, device=samples.device)
        return {
            "reward": reward,
            "accuracy": (reward > 0).float(),
            "distance": reward.abs(),
        }


def build_model() -> Wrapper:
    cfg = WrapperCfg(
        conditioning=ConditioningCfg(label=False, mask=True),
        model=ModelCfg(
            denoiser=UNetDenoiserCfg(
                name="unet",
                hid_dims=[32, 32],
                attention=[False, True],
                num_blocks=1,
                t_emb_dim=64,
                time_embedding=EmbeddingSinusodialCfg(name="sinusodial", d_emb=32),
                class_embedding=ClassEmbeddingParametersCfg(name="parameters"),
            ),
            flow=RectifiedFlowCfg(name="rectified", variance="learned_range"),
            time_sampler=MeanBetaCfg(name="mean_beta"),
            learn_sigma=True,
            ema=True,
        ),
        loss=LossCfg(),
        optimizer=OptimizerCfg(name="AdamW", lr=1.e-4),
        patch_size=PATCH_SIZE,
        train=TrainCfg(step_offset=0),
    )
    return Wrapper(cfg, d_data=1, image_shape=IMAGE_SHAPE)


def build_trainer(tmp_dir, order_enabled: bool) -> GRPOTrainer:
    model = build_model()
    rl = RLCfg(
        pretrained_checkpoint="<unused>",
        num_iterations=1,
        checkpoint_every=1,
        rollout=RolloutCfg(
            num_conditions=2,
            group_size=3,
            num_fill=[0, 8],
            max_steps=48,
        ),
        update=UpdateCfg(
            update_batch_size=4,
            step_fraction=0.5,
            kl_beta=0.01,
            noise_aware_weighting=True,
            sigma_aux_weight=0.01,
            flow_anchor_weight=0.1,
            aux_batch_size=2,
        ),
        order_policy=OrderPolicyCfg(enabled=order_enabled, temperature=0.1),
        eval=RLEvalCfg(every=1, num_samples=2, batch_size=2, num_fill=[0, 8], max_steps=24),
    )
    cfg = SimpleNamespace(
        rl=rl,
        dataset=SimpleNamespace(image_shape=IMAGE_SHAPE, grayscale=True),
        mnist_classifier="<unused>",
    )
    dataset = SyntheticGridDataset()
    trainer = GRPOTrainer(cfg, model, dataset, dataset, tmp_dir)
    trainer.reward_fn = RandomReward()
    return trainer


def check_rollout(trainer: GRPOTrainer) -> None:
    roll = trainer.collect()
    S, B = roll.num_steps, roll.num_trajectories
    assert roll.z_seq.shape[0] == S + 1
    assert roll.t_patch.shape == (S, B, GRID[0] * GRID[1])
    assert torch.equal(roll.sample, roll.z_seq[-1])

    # Known cells must stay at t == 0; unknown cells must end at t == 0
    known = trainer.rollout_sampler.full_mask_to_sequence_mask(roll.mask) <= 0.5
    assert (roll.t_patch[:, known] == 0).all(), "known cells must have t == 0"
    assert (roll.t_next_patch[-1] == 0).all(), "all cells must be clean at the end"

    # Schedules must be monotonically non-increasing per patch
    assert (roll.t_next_patch <= roll.t_patch + 1e-6).all()

    # Behavior log-prob recomputation: with unchanged weights ratio == 1
    b_idx, s_idx = roll.active.T.nonzero(as_tuple=True)
    stats = {k: [] for k in (
        "ratio", "clip_frac", "pg_loss", "kl", "order_loss", "sigma_aux", "flow_anchor"
    )}
    roll.advantage = torch.ones(B)  # nonzero so nothing is skipped
    loss = trainer._grpo_microbatch_loss(roll, b_idx[:16], s_idx[:16], stats)
    assert torch.isfinite(loss)
    ratio_err = abs(stats["ratio"][0] - 1.0)
    assert ratio_err < 1e-4, f"recomputed/behavior log-prob mismatch, ratio off by {ratio_err}"
    assert stats["clip_frac"][0] == 0.0
    # KL to the (initially identical) policy weights must be ~0... but the
    # reference is a copy of the EMA-initialized denoiser == policy, so:
    assert stats["kl"][0] < 1e-8, f"KL to identical reference should be 0, got {stats['kl'][0]}"

    if trainer.rl.order_policy.enabled:
        assert roll.order_patch.numel() > 0
        assert (roll.order_old_logp <= 0).all()
        # every trajectory schedules each unknown patch exactly once
        events_per_traj = (roll.order_event_idx >= 0).sum(dim=0)
        unknown_per_traj = (~known).sum(dim=1)
        assert torch.equal(events_per_traj, unknown_per_traj.cpu())
        # order-logp recomputation consistency: standardized logits in the
        # update must match those used at rollout time, so recomputed order
        # log-probs equal the stored behavior log-probs (D23). Recompute under
        # the (unchanged) policy from a fresh sigma forward.
        from src.rl.rollout import TrajectoryRecordingSampler
        ev = (roll.order_event_idx >= 0).T.nonzero(as_tuple=True)
        b_ev, s_ev = ev
        eids = roll.order_event_idx[s_ev, b_ev]
        z = roll.z_seq[s_ev, b_ev].to(trainer.device, torch.float32)
        t = trainer._patch_to_pixel(roll.t_patch[s_ev, b_ev].to(trainer.device, torch.float32))
        _, _, sigma = trainer.model.forward(z.unsqueeze(1), t.unsqueeze(1), sample=True, use_ema=False)
        patch_sigma = trainer._sigma_to_patch(sigma.float())
        re_logp = TrajectoryRecordingSampler.order_logp(
            patch_sigma, roll.order_unknown[eids].to(trainer.device),
            roll.order_patch[eids].to(trainer.device), trainer.rl.order_policy.temperature,
        )
        max_err = (re_logp.cpu() - roll.order_old_logp[eids]).abs().max().item()
        assert max_err < 1e-4, f"order-logp recomputation mismatch, off by {max_err}"
    print(f"  rollout OK: steps={S}, trajectories={B}, "
          f"active_pairs={roll.active.sum().item()}, events={roll.order_patch.numel()}")


def check_update(trainer: GRPOTrainer) -> None:
    roll = trainer.collect()
    params_before = torch.cat([
        p.detach().flatten().clone() for p in trainer.model.denoiser.parameters()
    ])
    stats = trainer.update(roll)
    params_after = torch.cat([
        p.detach().flatten().clone() for p in trainer.model.denoiser.parameters()
    ])
    assert stats["num_pairs"] > 0
    assert not torch.equal(params_before, params_after), "update must change the policy"
    assert np.isfinite(stats["pg_loss"]) and np.isfinite(stats["sigma_aux"])
    print(f"  update OK: {stats}")


def check_adaptive_kl_and_best(trainer: GRPOTrainer) -> None:
    # adaptive KL: a tiny target should push kl_beta up; best-checkpoint saving
    trainer.rl.update.kl_target = 1.e-9       # measured KL >> target -> beta rises
    beta0 = trainer.kl_beta
    roll = trainer.collect()
    stats = trainer.update(roll)
    assert "kl_beta" in stats and stats["kl_beta"] > beta0, "adaptive KL should raise beta"
    assert trainer.maybe_save_best(0.5) and (trainer.checkpoint_dir / "policy_best.pth").exists()
    assert not trainer.maybe_save_best(0.4), "lower acc must not overwrite best"
    assert trainer.maybe_save_best(0.6)
    print(f"  adaptive-KL/best OK: kl_beta {beta0:.3f}->{stats['kl_beta']:.3f}, best_acc={trainer._best_eval_acc}")


def check_eval(trainer: GRPOTrainer) -> None:
    metrics = trainer.evaluate()
    assert set(metrics) == {"reward", "accuracy", "distance"}
    print(f"  eval OK: {metrics}")


def check_distributed(world: int, rank: int) -> None:
    """Run under torchrun (gloo on CPU): verify ranks collect different
    rollouts but end every update with identical parameters."""
    from pathlib import Path
    import tempfile
    import torch.distributed as dist

    torch.manual_seed(0)        # identical model init on all ranks
    np.random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(Path(tmp), order_enabled=False)
        # diverge per-rank rollout conditions and sampling noise
        np.random.seed(100 + rank)
        torch.manual_seed(1234 + rank)
        roll = trainer.collect()

        # rollouts must differ across ranks
        reward_sum = roll.reward.sum()
        gathered_rewards = [torch.zeros_like(reward_sum) for _ in range(world)]
        dist.all_gather(gathered_rewards, reward_sum)
        assert not torch.allclose(gathered_rewards[0], gathered_rewards[1]), \
            "ranks collected identical rollouts (seeding broken)"

        stats = trainer.update(roll)
        assert stats["optimizer_steps"] == trainer.rl.update.optimizer_steps_per_epoch + 1

        # parameters (and EMA) must stay bit-synced across ranks
        for module in (trainer.model.denoiser, trainer.model.ema_denoiser.module):
            vec = torch.nn.utils.parameters_to_vector(module.parameters())
            checks = torch.stack([vec.sum(), vec.abs().sum()])
            gathered = [torch.zeros_like(checks) for _ in range(world)]
            dist.all_gather(gathered, checks)
            for g in gathered[1:]:
                assert torch.allclose(g, gathered[0], atol=1.e-6), \
                    "parameters diverged across ranks"

        # sharded eval + collective metric reduction must not deadlock
        trainer._log({"phase": "eval", **trainer.evaluate()})
    if rank == 0:
        print(f"  distributed OK: world={world}, optimizer_steps={stats['optimizer_steps']}")
        print("Distributed smoke test passed.")


def main() -> None:
    from pathlib import Path
    import tempfile

    from src.rl.distributed import maybe_init_distributed

    rank, world, _ = maybe_init_distributed()
    if world > 1:
        check_distributed(world, rank)
        return

    torch.manual_seed(0)
    np.random.seed(0)
    for order_enabled in (False, True):
        stage = 2 if order_enabled else 1
        print(f"Stage {stage} (order policy {'on' if order_enabled else 'off'}):")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = build_trainer(Path(tmp), order_enabled)
            check_rollout(trainer)
            check_update(trainer)
            if not order_enabled:
                check_eval(trainer)
                check_adaptive_kl_and_best(trainer)
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
