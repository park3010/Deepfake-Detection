
import model_abc
import torch
import torch.nn as nn
from torch.nn.functional import kl_div
import torch.nn.functional as F

from exp_strategy.resnet50 import ResNet50
from exp_strategy.efficientNet import EfficientNetB4
from networks import unet
import networks.utils as utils

# from exp_strategy.sfi_resnet import SFIResNet
# from exp_strategy.xception import Xception1

backbone_map = {
    'ResNet50':ResNet50,
    'EfficientNetB4':EfficientNetB4,
    # 'SFIResNet':SFIResNet,
    # 'Xception':Xception1
}

class UnetCrossDistill(model_abc.StrategyBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = unet.UnetCrossDistill(config)
        
        self.rec_loss = nn.MSELoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.W_dist = utils.SinkhornDistance().to(self.device)
        self.rec_w = 10.0
        self.distill_w = 1.0
        self.cls_w = 1.0
        self.threshold_bath_idx = 0
        
    def t_forward(self, x, y):
        xr, xs, v_xr, v_xs, z_xr, z_xs, pv_xr, pz_xr, pv_xs, pz_xs = self.model(x)
        
        # for vae
        loss_rec = self.rec_loss(xr, x)
        
        # cls
        loss_ce_vxr = self.cls_loss(pv_xr, y)
        loss_ce_vzr = self.cls_loss(pz_xr, y)
        loss_ce_vxs = self.cls_loss(pv_xs, y)
        loss_ce_vzs = self.cls_loss(pz_xs, y)
        loss_cls = 0.5*loss_ce_vxr + 0.25*loss_ce_vzr + 0.5*loss_ce_vxs + 0.25*loss_ce_vzs
        # distillation
        # vsd_loss = kl_div(input=self.softmax(v_x_rec.detach() / 1),
        #                   target=self.softmax(z_x_rec / 1)) + \
        #            kl_div(input=self.softmax(v_x_s.detach() / 1),
        #                   target=self.softmax(z_x_s / 1))
        
        vmd_loss = 0.5 * kl_div(F.log_softmax(pz_xs.detach(), dim=1),
                          F.softmax(pz_xr, dim=1)) + \
                   0.5 * kl_div(F.log_softmax(pz_xr.detach(),dim=1),
                          F.softmax(pz_xs,dim=1))
                   
        vcd_loss = 0.5 * kl_div(F.log_softmax(pv_xr.detach(), dim=1),
                                F.softmax(pz_xs,dim=1)) + \
                   0.5 * kl_div(F.log_softmax(pv_xs.detach(),dim=1),
                                F.softmax(pz_xr))
        
        vsd_loss = 0.5 * kl_div(F.log_softmax(pv_xr.detach(), dim=1),
                          F.softmax(pz_xr, dim=1)) + \
                   0.5 * kl_div(F.log_softmax(pv_xs.detach(),dim=1),
                          F.softmax(pz_xs,dim=1))
        
        if self.threshold_bath_idx > 0:
            loss_cls = 0.5*loss_ce_vxs + 0.25*loss_ce_vzs
            vsd_loss = vsd_loss * 0
            vcd_loss = vcd_loss * 0
            vmd_loss = vmd_loss * 0
            self.threshold_bath_idx = self.threshold_bath_idx - 1
        
        
        # conventional_ML = self.W_dist(self.softmax(pv_xr), self.softmax(pv_xs))
        total_loss = self.rec_w * loss_rec + loss_cls * self.cls_w +  self.distill_w * (vsd_loss +   vcd_loss +  vmd_loss)
        
        loss_dict = {'rec_loss': loss_rec,
                     'cls_loss': loss_cls,
                     'vsd_loss': vsd_loss,
                     'vcd_loss': vcd_loss,
                     "vmd_loss": vmd_loss,
                     'total_loss': total_loss,
                     }
        
        mode_out = model_abc.ModelOutput(total_loss = total_loss,
                               y_true=y,
                               y_pred=pz_xs,
                               loss_dict=loss_dict)
                               
        return mode_out
    
    def train_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        mode_out = self.t_forward(x, y)
        return mode_out

    def validation_step(self, batch, **kwargs):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        mode_out = self.t_forward(x, y)
        return mode_out

    def test_step(self, batch, **kwargs):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        mode_out = self.t_forward(x, y)
        return mode_out

    def get_model(self):
        return self.model 
    
    def image_save(self, batch, batch_idx, **kwargs):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        xr, xs, v_xr, v_xs, z_xr, z_xs, pv_xr, pz_xr, pv_xs, pz_xs = self.model(x)
        return xr, xs, x
    
def get_specific_strategy(backbone1:str, backbone2:str, **kwargs):
    if backbone1 not in backbone_map or backbone2 not in backbone_map:
        raise ValueError(f"backbone1 or backbone2 not in {list(backbone_map.keys())}")
    
    model1 = backbone_map[backbone1]()
    model2 = backbone_map[backbone2]()
    
    config1 = model1.get_config()
    config2 = model2.get_config()
    
    config = {
        'backbone1': config1,
        'backbone2': config2
    }
    return UnetCrossDistill(config)