"""Minimal torch.distributed helpers for multi-GPU GRPO training.

Design (cf. Decisions.md D17): no DDP wrapper. Each rank collects its own
rollouts (conditions are per-rank, groups never cross ranks) and accumulates
local gradients; gradients are manually all-reduced right before each of the
fixed number of optimizer steps per epoch, so ranks stay in lockstep even with
different numbers of microbatches.
"""
import os

import torch
import torch.distributed as dist


def maybe_init_distributed() -> tuple[int, int, torch.device]:
    """Initialize the process group from torchrun env vars if present.

    Returns (rank, world_size, device)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, dist.get_world_size(), device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return get_rank() == 0


def all_reduce_grads(module: torch.nn.Module) -> None:
    """Average gradients across ranks; materializes zero grads for parameters
    untouched by the local loss (their presence can differ between ranks, e.g.
    order-loss heads) so collectives stay matched."""
    world = get_world_size()
    if world == 1:
        return
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad)     # gloo has no AVG; SUM then scale
        p.grad /= world


def all_reduce_mean_scalars(record: dict, device: torch.device) -> dict:
    """Average the numeric entries of a metrics record across ranks.
    All ranks must call this with the same numeric keys (config-determined)."""
    if get_world_size() == 1:
        return record
    keys = sorted(k for k, v in record.items() if isinstance(v, (int, float)))
    values = torch.tensor([float(record[k]) for k in keys], device=device)
    dist.all_reduce(values)
    values /= get_world_size()
    return {**record, **dict(zip(keys, values.tolist()))}
