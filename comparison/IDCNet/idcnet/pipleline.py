# pipleline.py
import json
import os
import random
from pathlib import Path

import lightning as lgn
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from model_abc import MCLModel

torch.set_float32_matmul_precision('medium')


def setup_seed(seed: int):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    seed: int,
    model: MCLModel,
    train_loader,
    val_loader,
    epochs: int,
    save_dir,
    resume_ckpt=None,
    monitor: str = 'val_auc',
    mode: str = 'max',
    num_device: int = 1,
    val_check_interval: float = 0.2,
    limit_val_batches: float = 1.0,
    save_top_k: int = 10,
    ckpt_name: str = '{epoch:02d}-{val_acc:.4f}--{val_auc:.4f}',
    **kwargs,
):
    setup_seed(seed)

    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_ckpt_path = save_dir / 'checkpoints'
    save_ckpt_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_ckpt_path),
        filename=ckpt_name,
        monitor=monitor,
        save_top_k=save_top_k,
        mode=mode,
    )

    trainer = lgn.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_device if torch.cuda.is_available() else 1,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=str(save_dir),
        num_sanity_val_steps=2,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=True,
    )

    if resume_ckpt is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model, train_loader, val_loader)


def test_model(
    test_loader,
    dataset_name: str,
    save_dir,
    model: MCLModel,
    ckpt_path=None,
    num_device: int = 1,
    **kwargs,
):
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_result = save_dir if ckpt_path is None else save_dir / Path(ckpt_path).stem
    save_result.mkdir(parents=True, exist_ok=True)

    trainer = lgn.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_device if torch.cuda.is_available() else 1,
        default_root_dir=str(save_result),
        enable_progress_bar=True,
    )

    trainer.test(model, dataloaders=[test_loader], ckpt_path=ckpt_path)

    metrics = trainer.callback_metrics
    metrics_out = {}

    for k, v in metrics.items():
        try:
            metrics_out[k] = v.item()
        except Exception:
            metrics_out[k] = float(v)

    with open(os.path.join(str(save_result), f"{dataset_name}_metrics.json"), 'w') as file:
        json.dump(metrics_out, file, indent=4)

    trainer.callback_metrics.clear()