from dataclasses import dataclass

from jaxtyping import Bool, Float, Integer
import torch
from torch import Tensor

from ..misc.mnist_classifier import get_classifier


@dataclass
class SudokuRewardCfg:
    # reward = accuracy_bonus * 1[dist == 0] - distance_weight * dist
    accuracy_bonus: float = 1.0
    distance_weight: float = 0.1


class SudokuReward:
    """Standalone MNIST Sudoku verifier usable as an RL reward function.

    Mirrors MnistEvaluation.discretize and MnistSudokuEvaluation.classify without
    the Lightning evaluation machinery.
    """

    def __init__(
        self,
        cfg: SudokuRewardCfg,
        classifier_path: str,
        grid_size: tuple[int, int] = (9, 9),
    ) -> None:
        self.cfg = cfg
        self.classifier_path = classifier_path
        self.grid_size = grid_size

    @torch.no_grad()
    def discretize(
        self,
        samples: Float[Tensor, "batch 1 height width"]
    ) -> Integer[Tensor, "batch grid_height grid_width"]:
        classifier = get_classifier(self.classifier_path, samples.device)
        batch_size = samples.shape[0]
        tile_shape = tuple(s // g for s, g in zip(samples.shape[-2:], self.grid_size))
        tiles = samples.unfold(2, tile_shape[0], tile_shape[0])\
            .unfold(3, tile_shape[1], tile_shape[1]).reshape(-1, 1, *tile_shape)
        logits: Float[Tensor, "batch 10"] = classifier.forward(tiles)
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]
        # Replace zero predictions with second most probable number
        zero_mask = pred == 0
        pred[zero_mask] = idx[zero_mask, 1]
        return pred.reshape(batch_size, *self.grid_size)

    @torch.no_grad()
    def rule_distance(
        self,
        pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        Integer[Tensor, "batch"]
    ]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1     # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))
        grids = pred.unfold(1, sub_grid_size, sub_grid_size)\
            .unfold(2, sub_grid_size, sub_grid_size).reshape(-1, grid_size, grid_size)
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=2, index=grids, src=ones)
        dist.add_(cnt.abs_().sum(dim=(1, 2)).reshape(batch_size, -1).sum(dim=1))
        return dist == 0, dist

    @torch.no_grad()
    def __call__(
        self,
        samples: Float[Tensor, "batch 1 height width"]
    ) -> dict[str, Float[Tensor, "batch"]]:
        pred = self.discretize(samples)
        correct, dist = self.rule_distance(pred)
        correct, dist = correct.float(), dist.float()
        reward = self.cfg.accuracy_bonus * correct - self.cfg.distance_weight * dist
        return {"reward": reward, "accuracy": correct, "distance": dist}
