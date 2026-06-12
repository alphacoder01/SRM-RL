"""Entry point for GRPO RL fine-tuning of a pretrained SRM.

Composes the same hydra groups as pretraining (config/rl_main.yaml is a superset
of config/main.yaml), so `+experiment=ms1000_28` guarantees the model
configuration matches the checkpoint being fine-tuned. See Decisions.md.
"""
from pathlib import Path

import hydra
import torch
from colorama import Fore
from omegaconf import DictConfig, OmegaConf

from src.dataset import get_dataset
from src.global_cfg import set_cfg
from src.model.wrapper import Wrapper
from src.rl import GRPOTrainer, load_typed_rl_config


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="rl_main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_rl_config(cfg_dict)
    set_cfg(cfg_dict)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        import numpy as np
        np.random.seed(cfg.seed)

    if cfg.torch.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.torch.float32_matmul_precision)
    torch.backends.cudnn.benchmark = cfg.torch.cudnn_benchmark

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving RL outputs to {output_dir}."))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_data = 1 if cfg.dataset.grayscale else 3
    train_dataset = get_dataset(cfg.dataset, cfg.conditioning, "train")
    test_dataset = get_dataset(cfg.dataset, cfg.conditioning, "test")

    model = Wrapper(cfg, d_data, cfg.dataset.image_shape, train_dataset.num_classes)
    checkpoint_path = Path(cfg.rl.pretrained_checkpoint)
    assert checkpoint_path.exists(), f"Pretrained checkpoint not found: {checkpoint_path}"
    if checkpoint_path.suffix == ".ckpt":
        state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(cyan(f"Checkpoint load: {len(missing)} missing, {len(unexpected)} unexpected keys"))
    model = model.to(device)

    use_wandb = cfg.wandb.activated
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            name=f"rl_{output_dir.parent.name} ({output_dir.name})",
            tags=(cfg.wandb.tags or []) + ["rl", "grpo"],
            dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            entity=cfg.wandb.entity,
        )

    trainer = GRPOTrainer(
        cfg, model, train_dataset, test_dataset, output_dir, use_wandb=use_wandb
    )
    trainer.fit()


if __name__ == "__main__":
    main()
