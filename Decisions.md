# RL Fine-Tuning of SRMs: Design Decisions

This document records every major decision taken while implementing GRPO-style RL
fine-tuning (Flow-GRPO adapted to Spatial Reasoning Models) on top of the SRM codebase.
Ordered roughly by when the decision was made.

---

## D1. Task & reward: MNIST Sudoku with the existing classifier-based verifier

**Decision:** Fine-tune on the MNIST Sudoku task and reuse the repo's verifier
(per-cell MNIST classifier + Sudoku rule check) as the reward function, refactored
into a standalone `SudokuReward` module (`src/rl/reward.py`) independent of the
Lightning evaluation classes.

**Reward shape:** `r = accuracy_bonus * 1[dist == 0] - distance_weight * dist`, where
`dist` is the existing L1 row/column/block histogram violation metric from
`MnistSudokuEvaluation.classify`. The dense `dist` term provides shaping; the binary
term anchors the actual objective. Group normalization (D7) makes the absolute scale
irrelevant; only the monotone shaping matters.

**Why:** the verifier already exists and is exactly what the paper reports as its
metric; consistency with given cells is structurally guaranteed because the sampler
repaints known patches every step, so the reward only needs to check rule validity.

## D2. RL algorithm: Flow-GRPO-style PPO with group-relative advantages

**Decision:** On-policy GRPO: for each masked-Sudoku condition, sample a group of G
rollouts, compute group-relative advantages `(r - mean_g) / (std_g + eps)`, and apply a
PPO-clipped surrogate per denoising step with importance ratios from exact Gaussian
transition log-probs. KL regularization to a frozen reference copy of the pretrained
model.

**Why no ODE→SDE conversion (unlike Flow-GRPO):** SRM's generative process already
supports stochastic sampling with arbitrary noise schedules (η/alpha = 1). Every
transition is an explicit `DiagonalGaussian` returned by `Flow.conditional_p`, so
per-step log-probs are exact and differentiable through the predicted mean. We
therefore require `alpha > 0` for RL rollouts (asserted in code).

## D3. The uncertainty head (σ_θ): two-stage treatment

The σ head plays two roles: it selects generation *order* at inference (greedy
lowest-σ), and it shares the denoiser backbone, so RL gradients on the mean will
drift it. Additionally, its supervised NLL loss needs ground-truth ε, which does not
exist for rollout states (masked cells have multiple valid solutions).

**Stage 1 (default, implemented):**
- Order selection stays the deterministic greedy `argmin σ_θ` and is treated as part
  of the *environment*, not the policy. Deterministic given the state ⇒ contributes
  ratio 1 ⇒ no term in the GRPO objective. Valid because rollouts are regenerated
  every iteration (on-policy) and inner epochs are kept small.
- σ is kept *calibrated* by continuing its original supervised NLL loss
  (`mean_theta.detach()`, same as pretraining, weight `sigma_aux_weight`) on
  **dataset batches with the pretraining MeanBeta time sampler** — NOT on rollouts
  (no ground-truth ε there). Without this, RL improves the mean while the ordering
  silently degrades.

**Stage 2 (implemented behind `rl.order_policy.enabled`):**
- Order selection becomes a stochastic action: categorical over unknown patches with
  logits `-σ_patch / temperature`, sampled during rollouts; its log-prob enters the
  GRPO objective with the same per-trajectory advantage and PPO clipping. This lets
  RL optimize the generation order directly (the paper's stated future direction).
- In Stage 2, σ stops being calibrated uncertainty and becomes order-policy logits;
  `sigma_aux_weight` should be lowered/zeroed (config decision left to experiments).
- Restriction: stochastic order requires `top_k == 1` (asserted). Logits use `-σ/τ`
  (monotone in σ; squaring would only change the temperature semantics).

## D4. Transition variance is frozen during RL

**Decision:** The policy distribution's variance (from `learned_range` interpolation
`v_θ` and the analytic σ-schedule) is **detached** during RL updates; gradients flow
only through the predicted mean. The VLB loss that trains `v_θ` in pretraining is not
applied during RL.

**Why:** letting RL shape the variance invites variance-collapse reward hacking and
ill-conditioned Gaussian ratios. This matches Flow-GRPO, where the SDE noise scale is
fixed by the schedule.

## D5. Which transitions count as actions (active-pixel masking)

**Decision:** Per step, the log-prob (and KL) is computed only over *active* pixels:
`(t > t_next) AND (t_next > 0)`, which implies the patch is unknown (masked) and inside
its denoising block. Excluded:
- known cells (t = 0 always; repainted every step),
- not-yet-started patches (t = t_next = 1, zero-variance identity transition),
- finished patches (t = t_next = 0),
- the final sub-step of each patch (t_next = 0), which the sampler takes
  deterministically via the posterior mean (`sampler.py` uses `conditional_p.mean`
  when `t_next == 0`), so it has no density.

**Log-prob normalization:** per-step log-probs are **averaged over active scalar
dimensions** (pixels × channels), not summed, so ratio magnitudes are comparable
across steps with different active-set sizes and across patch sizes. The clip range
is calibrated to this convention.

## D6. Trajectory recording strategy

**Decision:** A dedicated `TrajectoryRecordingSampler` (subclass of
`SequentialAdaptiveSampler`, `src/rl/rollout.py`) re-implements the sampling loop and
explicitly records, per step: the patch-level `t` and `t_next` actually used, the
post-repaint latent `z`, the behavior log-prob over active pixels, and order-decision
events (step, batch element, chosen patch, unknown-mask snapshot, categorical
log-prob if stochastic).

- We store `t`/`t_next` explicitly at patch resolution ([steps, B, 81]) rather than
  relying on the invariant that the final scheduling matrix equals what was read at
  each step (true, but fragile — depends on prototype[0] == 1). Explicit > clever.
- `z` sequence is stored at full resolution on CPU in float32 by default
  (~2 GB for 32 trajectories × 243 steps on Sudoku; configurable device/dtype). We
  store post-repaint states; repaint does not modify active pixels, so log-probs are
  unaffected.
- Behavior log-probs are computed at rollout time from the exact same
  `conditional_p` used for sampling (no recomputation mismatch).

## D7. Group construction & degenerate groups

**Decision:** A group = G rollouts (different seeds) of the *same condition* (same
masked Sudoku, same mask). Advantage = `(r - mean_g)/(std_g + eps)`. Groups where
`std_g ≈ 0` (all-success or all-fail) get advantage 0 and are skipped — they carry no
learning signal under GRPO and skipping acts as a free curriculum. With the
pretrained model at ~50% hard accuracy, most groups are informative.

## D8. Denoising reduction & timestep subsampling

**Decision (rollouts):** train-time rollouts use a reduced step budget
(`rl.rollout.max_steps`, default 243 ≈ 3 steps/patch for hard Sudoku with top_k=1,
no overlap) instead of the paper's 1000; evaluation uses the full budget
(`rl.eval.max_steps`, default 1000). Flow-GRPO's "denoising reduction". The README
recommends first measuring baseline accuracy vs. step budget and adjusting.

**Decision (updates):** within each inner epoch we train on a random fraction of
the active steps of each trajectory (`rl.update.step_fraction`, default 0.25), the
DanceGRPO-style timestep subsampling, since each trained step costs one full UNet
forward. Order-decision steps can be force-included (`include_order_steps`) so the
Stage-2 order loss sees every decision.

**Noise-aware step weighting** (TempFlow-GRPO-inspired): optional
(`rl.update.noise_aware_weighting`, default **off** for a vanilla baseline): weights
each step's surrogate by the mean transition std over active pixels (normalized per
microbatch), concentrating credit on the stochastic (high-noise) steps.

## D9. Reference model, EMA, and initialization

**Decision:**
- The policy is initialized from the **EMA weights** of the pretrained checkpoint
  (EMA is what the paper evaluates).
- The frozen **reference model for KL** is a deep copy of those EMA weights.
- During RL, the wrapper's existing EMA machinery keeps running over the policy
  (`ema_denoiser.update_parameters` after each optimizer step); **eval uses the EMA**
  policy, rollouts use the raw policy (`use_ema=False`) so behavior == trained policy.
- KL term: analytic diagonal-Gaussian KL `KL(π_new || π_ref)` per active pixel
  (means and variances from each model's own `conditional_p`), coefficient
  `rl.update.kl_beta` (default 0.01; set 0 to disable and save the reference forward
  pass).

## D10. Custom training loop instead of PyTorch Lightning

**Decision:** The GRPO trainer (`src/rl/grpo.py`) is a plain PyTorch loop, not a
LightningModule. The rollout→reward→multi-epoch-update cycle does not fit Lightning's
dataloader-driven `training_step` contract without contortions (rollouts inside
training steps, fake dataloaders). Logging is stdout + JSONL
(`outputs_rl/.../metrics.jsonl`) + optional wandb (reuses the repo's `wandb` config
section). Checkpoints are saved both as a resumable RL state (`rl_state.pt`) and as a
plain `state_dict` `.pth` that the existing `test.sh` pipeline can load via
`checkpointing.load=<path>` (suffix-based `load_state_dict` path in `src/_main.py`).

**Scope:** single-GPU for v1. Multi-GPU rollout sharding is a later optimization.

## D11. Reuse of the experiment config system

**Decision:** RL runs compose the *same* hydra groups as pretraining via a new root
config `config/rl_main.yaml` (a superset of `config/main.yaml` with an `rl:` section),
so `+experiment=ms1000_28` pulls in exactly the pretrained model/dataset configuration
and guarantees the architecture matches the checkpoint. Entry point: `src/main_rl.py`,
launcher: `train_rl.sh`. The typed config is `RLRootCfg(RootCfg)` parsed with dacite
(extra `rl:` key is ignored by the base parse).

**Sampler hyperparameters are explicit** in the RL config (`max_steps`, `alpha`,
`overlap`, `top_k`): note that the released `config/sampler/seq_adaptive.yaml` leaves
`max_steps`/`alpha` at dataclass defaults (100 / 0.0), which does not match the
paper's 1000-step stochastic setup — we do not inherit that ambiguity.

## D12. Reward-hacking defenses

**Decision:** three concentric guards, all configurable:
1. KL anchor to the reference model (D9).
2. Optional flow-matching anchor loss (`rl.update.flow_anchor_weight`, default 0) —
   the pretraining MSE loss on dataset batches, applied alongside the σ aux loss.
3. The verifier is a small MLP and *will* be attacked by optimization pressure over
   time; the README recommends monitoring with an independently trained classifier.
   (Not implemented in v1: classifier ensembling.)

## D13. Aux supervised batches reuse pretraining machinery

**Decision:** σ-NLL aux loss (and optional flow anchor) run on real dataset batches
drawn from the same train dataset with `model.time_sampler` (MeanBeta / Uniform-t̄)
and the same per-patch loss-weight pipeline (`torch.kron` expansion, mask zeroing)
as `Wrapper.training_step`. The logic is re-implemented in a focused function rather
than calling `training_step` (which would also compute the VLB and flow losses with
their own weights and logging).

## D14. Supported model class

**Decision:** v1 supports patch-based SRMs only (`patch_size != None`). The
image-level inpainting diffusion baseline (`patch_size: null`, mask-concat
conditioning) is out of scope for RL — the interesting RL surface (ordering,
per-patch sequentialization) only exists for SRMs. Asserted at trainer construction.

## D15. Verification status

Code was smoke-tested CPU-only with a tiny randomly initialized UNet (script
`tests/rl_smoke_test.py`): rollout recording, active masking, log-prob recomputation
consistency (recomputed log-probs match rollout-time log-probs for unchanged weights),
group advantages, one full GRPO update step (Stage 1 and Stage 2), and aux losses.
Full-scale training requires GPUs and the released datasets/checkpoints, which are
not available in this development environment.

## D16. Post-mortem of the first training run: gradient accumulation is mandatory

**Observation (first 60 iterations, default config):** train-rollout accuracy
collapsed 0.55 → ~0.0, distance 2.7 → 20+, KL to the reference grew linearly
0.002 → 0.043, and the supervised σ-NLL on real data rose steadily — the policy
walked off the pretrained manifold. Eval accuracy looked stable only because it
uses the slow EMA (decay 0.9999), which lags the degrading raw policy.

**Root cause:** the original update loop took an optimizer step on *every*
8-pair microbatch — ~218 sequential AdamW steps per rollout batch, all driven by
advantages from only 4 conditions. AdamW's preconditioning normalizes gradient
magnitude, so even near-zero noisy gradients move parameters by ~lr per step;
hundreds of such steps per iteration are a noise-driven random walk, and
within-epoch off-policy drift (mean ratio settling at ~0.998) compounds it.
Flow-GRPO-style training takes ~1–4 optimizer steps per rollout round.

**Fixes:**
1. Gradient accumulation across microbatches: `update.optimizer_steps_per_epoch`
   (default 4) controls how many optimizer steps each inner epoch takes; each
   step now averages ~400+ pairs instead of 8.
2. `rollout.num_conditions` default 4 → 8: with few conditions, all pairs in a
   batch share a handful of advantage values; doubling conditions doubles the
   effective sample size of the gradient.
3. `update.kl_beta` 0.01 → 0.04 and `update.flow_anchor_weight` 0.0 → 0.1:
   the observed drift was exactly the failure mode these terms guard against
   (D9/D12); the run showed the previous defaults were too weak.

Degraded runs should be restarted from the pretrained checkpoint with a fresh
run id (resuming loads the contaminated weights and EMA from `rl_state.pt`).

## D17. Multi-GPU training and mixed precision

**Multi-GPU (torchrun, no DDP wrapper):** `train_rl.sh` auto-launches
`torchrun --standalone --nproc_per_node=<gpus>` when more than one GPU is
visible. Design choices:
- **Conditions are per-rank** (`rl.rollout.num_conditions` is per GPU, matching
  the repo's per-GPU `data_loader.batch_size` convention); each group of G
  rollouts lives entirely on one rank, so group-relative advantages need no
  communication.
- **Manual gradient all-reduce instead of a DDP wrapper:** ranks can have
  different numbers of microbatches (active-step counts vary), which breaks
  DDP's backward-hook synchronization. Instead, each inner epoch splits the
  local microbatch list into exactly `optimizer_steps_per_epoch` chunks
  (np.array_split, possibly empty), so every rank executes the same fixed
  number of all-reduce + step collectives per epoch — lockstep by construction.
  Zero grads are materialized for parameters untouched by a rank's local loss
  (e.g. order-loss terms present on one rank only) to keep collectives matched.
- Identical gradients after all-reduce + identical AdamW state keep weights
  (and therefore EMA) bit-synced; verified by the distributed smoke test
  (`torchrun --standalone --nproc_per_node=2 -m tests.rl_smoke_test`, gloo/CPU).
- Eval shards sample indices `rank::world_size`; metric records are averaged
  across ranks inside `_log` (a collective all ranks must enter); file/wandb
  logging and checkpointing happen on rank 0 only. Seeds are offset by rank so
  rollout conditions and noise differ across GPUs.

**Mixed precision (`rl.precision`, default "bf16"):** denoiser forwards in
rollouts, GRPO updates, and aux losses run under bfloat16 autocast (CUDA only);
network outputs are cast back to float32 before any Gaussian transition math,
so log-probs, ratios, and KLs keep full precision. Behavior and recomputed
log-probs use the same autocast path, so the ratio == 1 consistency at epoch
start is preserved exactly. Eval sampling stays full precision (it is rare and
is the number compared against the paper).

**Optional rollout compilation (`rl.rollout.compile`, default off):** rollout
forwards use the model's `forward_compiled` path; the rollout batch shape is
fixed across all `max_steps` evaluations, so compilation amortizes well. Off by
default because compile warmup costs minutes and pays off only for long runs.

## D18. Second run was stable but under-trained: loosen the optimization

**Observation (run2, 2 GPUs, 94 iterations, post-D16/D17 defaults):** no
divergence — KL to reference flat at ~0.001, `sigma_aux` stable around -0.3
(no off-manifold drift), train accuracy noisy around the 0.5 baseline. But also
*no learning*: eval accuracy stayed pure-noise around 0.5 (0.50 / 0.50 / 0.61 /
0.44 at iters 0/25/50/75), `pg_loss` ~1e-6, `ratio` ~0.9999, `clip_frac` ~0.

**Diagnosis:** the D16 fix over-corrected. Total optimization per run was tiny
(94 iters x 4 steps x lr 1e-5 = 376 AdamW steps on a 118M UNet), and the policy
barely left the pretrained checkpoint (KL 0.001 vs the kl_beta coefficient of
0.04 — ~40x of unused headroom; run1 only diverged past KL ~0.04 *without*
anchors). The gradient signal was healthy (advantage_abs ~0.88, degenerate
groups ~1/16); there simply was almost no step being taken.

**Changes (defaults):**
- `update.inner_epochs` 1 -> 2: reuse each (expensive) rollout batch twice,
  PPO-style; the importance ratio + clip handle the mild within-batch
  off-policyness.
- `update.optimizer_steps_per_epoch` 4 -> 8: each step still averages hundreds
  of (trajectory, step) pairs, so SNR stays high; safe given the strong anchors.
- `update.lr` 1e-5 -> 3e-5.

Net ~8x more effective optimization per iteration. With the KL/flow anchors and
per-eval monitoring as guardrails, watch that KL settles in the ~0.01-0.05 band
(not climbing unbounded) and `sigma_aux` stays near its initial ~-0.3; if KL
runs away or eval/sigma_aux degrade, dial `lr` or `optimizer_steps_per_epoch`
back down. These are the first knobs to tune per-checkpoint, not fixed truths.

## D19. Throughput: the eager update phase dominates wall-clock

**Profile (run2, ~10 h):** in forward-pass-equivalents, the GRPO update phase is
~60% of total compute, rollout ~35%, eval ~3%. The update ran at
`update_batch_size = 8` — far below what a 118M UNet at 252x252 saturates
(pretraining used batch 28) — and post-D18's `inner_epochs = 2` plus the
per-microbatch KL reference forward made it the clear bottleneck.

**Changes:**
- `update.update_batch_size` 8 -> 32: a pure throughput knob (4x fewer, larger
  forward/backward calls on the dominant phase, GPU utilization comparable to
  pretraining). No memory concern at batch 32 (pretraining trained batch 28 with
  optimizer state on the same GPUs).
- Fixed the gradient-accumulation chunking to be **pair-indexed** instead of
  microbatch-count-indexed. Previously each optimizer step's set of pairs (and
  thus its gradient) shifted with `update_batch_size`, so changing the batch
  size changed the optimization (batch8-vs-batch32 single-iteration update
  cosine was only 0.72). Now each optimizer step covers a fixed contiguous pair
  chunk and accumulates microbatches weighted by their pair fraction, so its
  gradient is the exact grand mean over the chunk regardless of batch size
  (cosine 0.988, matched magnitude — residual is fp summation order). This makes
  `update_batch_size` safe to tune for speed and also removes a latent
  optimization dependence on it.

**Other levers (not new defaults; per-run overrides):**
- `rl.eval.max_steps=243` (or 162): eval accuracy is saturated well below 1000
  steps (verified earlier), so 4x-cheaper evals during training; keep a final
  1000-step eval for the paper-comparable number.
- `rl.rollout.storage_device=cuda`: keeps the recorded latent trajectory on GPU
  (exact, fp32), removing 243 host<->device syncs per rollout and the reload in
  the update — worth it when GPU memory allows (~4 GB for the z buffer at the
  default Sudoku rollout size).
- `rl.rollout.compile=true` (already used) compiles the rollout forward; the
  eager update forward remains a future compile target (needs fixed microbatch
  shape, now satisfied by the pair-chunked loop, to avoid recompiles).

## D20. OOM at update_batch_size=32 on A100-80GB; corrected default and headroom guard

**Observation:** D19's `update_batch_size=32` OOM'd on 2xA100-80GB even with
`storage_device=cpu`. run2 (batch 8) had completed fine, so the OOM is the
update forward+backward, not the rollout.

**Why the ceiling is below pretraining's batch 28:** the RL update holds more
per-sample memory than supervised pretraining did — a second (reference) forward
for the KL term, several full-resolution DiagonalGaussian tensors
(mean/var/logp/kl maps), and three resident model copies (policy + EMA +
frozen reference) vs pretraining's two. So batch 32 exceeded 80 GB where
pretraining's batch 28 fit.

**Fixes:**
- `update.update_batch_size` 32 -> 16 default: comfortably under the pretraining
  batch, still ~2x fewer microbatches/kernels than the original 8, so it keeps a
  real speedup. Documented as the primary OOM knob (lower to 8 if still OOM,
  raise toward 24 with headroom). Because the accumulation is batch-invariant
  (D19), changing it never changes the optimization.
- `torch.cuda.empty_cache()` between `collect()` and `update()` each iteration:
  the two phases are sequential, so returning the rollout's cached allocator
  blocks gives the update maximal contiguous headroom and guards against
  fragmentation OOM. One call per iteration, negligible cost.

**OOM playbook (documented in README):** lower `update_batch_size` first; then
reduce rollout memory via `rollout.group_size` / `rollout.num_conditions`; keep
`rollout.storage_device=cpu`.

## D21. Scaling across GPUs: shard the global batch, don't grow it

**Question:** does adding GPUs make an iteration faster? In this design, not by
default. `rollout.num_conditions` is PER GPU and each rank collects its own
rollouts and updates on its own pairs with gradient all-reduce (D17). So adding
GPUs at fixed `num_conditions` keeps per-GPU work constant and instead *grows
the global batch* (more conditions/iteration) at the same seconds-per-iteration.
That improves gradient quality / throughput, but does not reduce wall-clock per
iteration — which is usually what "make it faster" means.

**To reduce per-iteration wall-clock:** hold the global batch fixed and shard it
across ranks by lowering `num_conditions` as GPUs are added, e.g. keep 16 global
conditions via `num_conditions = 16 / num_gpus` (4 GPUs -> 4, 8 GPUs -> 2, 16
GPUs -> 1; a group must live on one rank, so 1 is the floor). Both phases scale
~linearly with per-GPU trajectories (rollout batch and update pair count), so
this gives near-linear speedup until the rollout's 243 *sequential* steps become
the floor at very small per-GPU batch.

**Decision:** keep the per-GPU semantics (no breaking change) but print a startup
summary on rank 0 — world size, conditions/GPU, derived global conditions,
rollout batch/GPU, global rollouts/iter — so the operator can see whether adding
GPUs grew the batch or sped up the iteration. The sharding recipe is documented
in the README.

**Related cost levers for the observed ~650 s/it (2x A100, 16 global conditions):**
the first interval also includes torch.compile warmup and the iteration-0
full eval; eval runs every 25 iters at 1000 steps. Recommended overrides for
speed: `rl.eval.max_steps=243` (accuracy saturates well below 1000, verified)
and `rl.eval.every=50` to cut periodic eval cost, plus `rl.update.update_batch_size=24`
when GPU headroom allows.
