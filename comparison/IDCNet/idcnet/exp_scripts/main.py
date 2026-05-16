# main.py
import logging
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader

from exp_strategy import differ_backbone_exp
import model_abc
import mydataset
import mytransforms
import pipleline


def get_train_loader(t_batch_size=16, v_batch_size=16, num_workers=4, balance=False, aug=None):
    """
    train / val = FF++ only
    """
    if aug is None:
        aug = mytransforms.Transform4()

    train_dataset = mydataset.TrainDataset1(
        mode="train",
        transform=aug,
        normalize=None,
        balanced=balance,
    )

    val_dataset = mydataset.TrainDataset1(
        mode="val",
        transform=mytransforms.DefaultTransform(),
        normalize=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=v_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader


def parse_auc(file_name):
    match = re.search(r'val_auc=([0-9]+(?:\.[0-9]+)?)', file_name)
    if match:
        return float(match.group(1))
    raise ValueError(f'Cannot find val_auc in {file_name}')


def get_ckpt(save_dir, backbone_pair, mode='resume'):
    curr_dir = save_dir / f"{backbone_pair[0]}_{backbone_pair[1]}_recw_10"
    checkpoint = curr_dir / 'checkpoints'

    if not curr_dir.exists() or not checkpoint.exists():
        return None

    ckpt_list = list(checkpoint.glob('*.ckpt'))
    if not ckpt_list:
        return None

    if mode == 'resume':
        latest_file = max(ckpt_list, key=lambda x: x.stat().st_ctime)
        logging.info(f'Loading latest ckpt {latest_file}')
        return str(latest_file)

    if mode == 'best':
        best_file = max(ckpt_list, key=lambda x: parse_auc(x.stem))
        logging.info(f'Loading best ckpt {best_file}')
        return str(best_file)

    raise ValueError(f"Invalid mode: {mode}")


def get_test_loaders(batch_size=16, num_workers=4):
    """
    external datasets only
    - Celeb
    - DFD
    - DeepfakeTIMIT
    - WildDeepfake

    mydataset.build_external_test_dict()가
    key별로 분리된 test dict를 반환한다고 가정.
    """
    test_dict = mydataset.build_external_test_dict()

    loaders = []
    dataset_names = []

    for dataset_name, one_dataset_dict in test_dict.items():
        tmp_dataset = mydataset.TestDataset1(
            test_dict={dataset_name: one_dataset_dict},
            transform=mytransforms.DefaultTransform(),
            normalize=None,
        )
        tmp_loader = DataLoader(
            tmp_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(tmp_loader)
        dataset_names.append(dataset_name)

    return loaders, dataset_names


def build_model(backbone1='ResNet50', backbone2='EfficientNetB4', lr=1e-4):
    model_strategy = differ_backbone_exp.get_specific_strategy(backbone1, backbone2)

    # strategy hyperparameters
    model_strategy.rec_w = 10.0
    model_strategy.distill_w = 0.1
    model_strategy.cls_w = 1.0
    model_strategy.threshold_bath_idx = 1800

    model_lgn = model_abc.MCLModel(
        strategy=model_strategy,
        using_cls_metric=True,
        lr=lr,
    )
    return model_lgn, model_strategy


def train_ResNet50_EfficientNetB4():
    backbone1 = 'ResNet50'
    backbone2 = 'EfficientNetB4'

    save_dir = Path(f"/home/oem/deepfake/Ourmethod/comparison/_ckpt/idcnet/{backbone1}_{backbone2}_recw_10")
    save_dir.mkdir(exist_ok=True, parents=True)

    resume_ckpt = get_ckpt(save_dir.parent, [backbone1, backbone2], mode='resume')

    model_lgn, _ = build_model(backbone1, backbone2, lr=3e-4)

    train_loader, val_loader = get_train_loader(
        t_batch_size=16,
        v_batch_size=16,
        num_workers=5,
        balance=True,
        aug=mytransforms.Transform4(),   # 가장 단순한 train transform
    )

    ckpt_name = "epoch={epoch:02d}-val_acc={val_acc:.4f}-val_auc={val_auc:.4f}"

    pipleline.train_model(
        seed=42,
        model=model_lgn,
        save_dir=save_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        num_device=1,              # DDP 제거
        val_check_interval=1.0,    # epoch마다 validation
        resume_ckpt=resume_ckpt,
        epochs=10,
        monitor="val_auc",
        mode="max",
        save_top_k=10,
        ckpt_name=ckpt_name,
        limit_val_batches=1.0,
    )


def test_ResNet50_EfficientNetB4():
    backbone_pair = ['ResNet50', 'EfficientNetB4']
    save_root = Path("/home/oem/deepfake/Ourmethod/comparison/_ckpt/idcnet")
    ckpt_path = get_ckpt(save_root, backbone_pair, mode='best')

    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found. Train first.")

    model_lgn, _ = build_model(backbone_pair[0], backbone_pair[1], lr=1e-4)

    test_loaders, dataset_names = get_test_loaders(batch_size=30, num_workers=5)

    result_save = Path("/home/oem/deepfake/Ourmethod/comparison/_result/idcnet")
    result_save.mkdir(exist_ok=True, parents=True)

    for test_loader, dataset_name in zip(test_loaders, dataset_names):
        print(f"[TEST] {dataset_name}")
        pipleline.test_model(
            test_loader=test_loader,
            dataset_name=dataset_name,
            save_dir=result_save,
            model=model_lgn,
            ckpt_path=ckpt_path,
            num_device=1,   # DDP 제거
        )


if __name__ == '__main__':
    # 1) train
    # train_ResNet50_EfficientNetB4()

    # 2) test on external datasets
    test_ResNet50_EfficientNetB4()