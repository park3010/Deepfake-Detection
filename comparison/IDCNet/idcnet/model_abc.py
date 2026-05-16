from abc import ABC, abstractmethod
from typing import Dict

import torch
import torchmetrics.classification
from lightning.pytorch import LightningModule

import my_metrics
from dataclasses import dataclass

@dataclass
class ModelOutput:
    """
    该类用于存储模型的输出结果。
    Attributes:
        loss (float): 模型的总损失。
        y_true (Any): 真实标签
        y_pred (Any): 预测标签
        loss_dict (Dict[str, float], optional): 各部分损失的字典，键为损失名称，值为对应的损失值。
    """
    total_loss: torch.Tensor
    y_true: torch.Tensor
    y_pred: torch.Tensor
    loss_dict: Dict[str, torch.Tensor] = None
    
    
class StrategyBase(ABC):
    def __init__(self, config=None):
        self.config = config
        self.device:torch.device = None
        self.lgn_base:LightningModule = None
        
    def train_step(self, batch, batch_idx)->ModelOutput:
        pass
    @abstractmethod
    def validation_step(self, batch, **kwargs)->ModelOutput:
        pass
    @abstractmethod
    def test_step(self, batch, **kwargs)->ModelOutput:
        pass
    @abstractmethod
    def get_model(self):
        pass

class MCLModel(LightningModule):
    def __init__(self, config=None, strategy=None, using_cls_metric=True, lr=1e-4):
        super().__init__()
        self.config = config
        self.strategy:StrategyBase = strategy
        self.strategy.lgn_base = self
        self.model = self.strategy.get_model()
        self.lr = lr
        self.using_cls_metric = using_cls_metric
        if using_cls_metric:
            # define metrics of binary classification
            self.val_acc = torchmetrics.classification.Accuracy(task='binary')
            self.val_auc = torchmetrics.classification.AUROC(task='binary')
            self.val_ap = torchmetrics.classification.AveragePrecision(task='binary')
            self.val_precision = torchmetrics.classification.Precision(task='binary')
            self.val_recall = torchmetrics.classification.Recall(task='binary')
            self.val_eer = my_metrics.EER()

            self.test_acc = torchmetrics.classification.Accuracy(task='binary')
            self.test_auc = torchmetrics.classification.AUROC(task='binary')
            self.test_ap = torchmetrics.classification.AveragePrecision(task='binary')
            self.test_precision = torchmetrics.classification.Precision(task='binary')
            self.test_recall = torchmetrics.classification.Recall(task='binary')
            self.test_eer = my_metrics.EER()

    def training_step(self, batch, batch_idx):
        self.strategy.device = self.device
        model_out:ModelOutput = self.strategy.train_step(batch, batch_idx)
        log_dict = {}
        for key, value in model_out.loss_dict.items():
            key = f"train_{key}"
            log_dict[key] = value
        self.log_dict(log_dict, sync_dist=True)
        return model_out.total_loss

    def validation_step(self, batch, batch_idx):
        self.strategy.device = self.device
        model_out:ModelOutput = self.strategy.validation_step(batch)
        if self.using_cls_metric:
            self.cls_metrics(model_out.y_true, model_out.y_pred[:, 1], 'val')
        log_dict = {}
        for key, value in model_out.loss_dict.items():
            key = f"val_{key}"
            log_dict[key] = value
        self.log_dict(log_dict, sync_dist=True)

    def cls_metrics(self,y, y_pred, mode):
        if mode == 'val':
            self.val_acc.update(y_pred, y)
            self.val_auc.update(y_pred, y)
            self.val_ap.update(y_pred, y)
            self.val_precision.update(y_pred, y)
            self.val_recall.update(y_pred, y)
            self.val_eer.update(y_pred, y)
        elif mode == 'test':
            self.test_acc.update(y_pred, y)
            self.test_auc.update(y_pred, y)
            self.test_ap.update(y_pred, y)
            self.test_precision.update(y_pred, y)
            self.test_recall.update(y_pred, y)
            self.test_eer.update(y_pred, y)
        else:
            raise ValueError("mode should be val or test")

    def test_step(self, batch, batch_idx):
        self.strategy.device = self.device
        model_out:ModelOutput = self.strategy.test_step(batch)
        
        if self.using_cls_metric:
            self.cls_metrics(model_out.y_true, model_out.y_pred[:, 1], 'test')
        
        for key, value in model_out.loss_dict.items():
            key = f"test_{key}"
        self.log_dict(model_out.loss_dict, sync_dist=True)

    def on_validation_epoch_end(self):
        if not self.using_cls_metric:
            return
        self.log_dict({'val_acc': self.val_acc.compute(),
                       'val_auc': self.val_auc.compute(),
                       'val_ap': self.val_ap.compute(),
                       'val_precision': self.val_precision.compute(),
                       'val_recall': self.val_recall.compute(),
                       'val_eer': self.val_eer.compute()}, sync_dist=True)
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_ap.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_eer.reset()

    def on_test_epoch_end(self):
        if not self.using_cls_metric:
            return
        self.log_dict({'test_acc': self.test_acc.compute(),
                       'test_auc': self.test_auc.compute(),
                       'test_ap': self.test_ap.compute(),
                       'test_precision': self.test_precision.compute(),
                       'test_recall': self.test_recall.compute(),
                       'test_eer': self.test_eer.compute()}, sync_dist=True)
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_ap.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_eer.reset()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer

