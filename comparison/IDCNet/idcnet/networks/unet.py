import torch
import torch.nn as nn

from networks.effient import Efficientnet
from .unet_parts import *

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)



class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256, mid_ch=512):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = mid_ch
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Linear(num_bottleneck, in_ch)]
        add_block += [nn.BatchNorm1d(in_ch)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(in_ch, out_ch)]

        # Extra BN layer, need to be removed
        #add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(init_weights)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x

from torch.nn import init
class VIBNet(nn.Module):
    def __init__(self, in_ch=512, z_dim=256, num_class=2):
        super(VIBNet, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch, mid_ch=z_dim)
        # classifier of VIB, maybe modified later.
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class), nn.Softmax(dim=1)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        # self.classifier.apply(self.weights_init_classifier)
        
    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.normal_(m.weight.data, std=0.001)
            init.constant_(m.bias.data, 0.0)
    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        return z_given_v,p_y_given_z
    
class Baseline(nn.Module):
    def __init__(self, config = None):
        super(Baseline, self).__init__()
        self.bae = config['network']
        dim = config['feat_dim']
        classifier = [nn.Linear(dim, 2),nn.Softmax(dim=1)]
        self.classifier = nn.Sequential(*classifier)
        # self.classifier.apply(self.weights_init_classifier)
        
    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            nn.init.constant_(m.bias.data, 0.0)
            
    def forward(self, x):
        v = self.bae(x)
        p_y_given_v = self.classifier(v)
        return v, p_y_given_v


class UnetCrossDistill(nn.Module):
    def __init__(self, config=None):
        super(UnetCrossDistill, self).__init__()
        """
        config:
        {
            'backbone1': {
                'name': 'efficientnet-b0',
                network: None,
                'feat_dim': 512
            'backbone2': {
                'name': 'efficientnet-b0',
                network: None,
                'feat_dim': 512
        }
        """
        
        print(config)
        backbone1_config = config['backbone1']
        backbone2_config = config['backbone2']
        
        backbone1_feat_dim = backbone1_config['feat_dim']
        backbone2_feat_dim = backbone2_config['feat_dim']
        
        
        self.unet = UNet(n_channels=3, out_channels=3)
        self.enc1 = Baseline(config=config['backbone1'])
        self.enc2 = Baseline(config=config['backbone2'])
        
        self.ib1 = VIBNet(in_ch=backbone1_feat_dim, z_dim=256, num_class=2)
        self.ib2 = VIBNet(in_ch=backbone2_feat_dim, z_dim=256, num_class=2)
    
    def forward(self, x):
        xr = self.unet(x)
        xs = x - xr
        
        v_xr, pv_xr = self.enc1(xr)
        z_xr, pz_xr = self.ib1(v_xr)
        
        v_xs, pv_xs = self.enc2(xs)
        z_xs, pz_xs = self.ib2(v_xs)
        
        return xr, xs, v_xr, v_xs, z_xr, z_xs, pv_xr, pz_xr, pv_xs, pz_xs



class UnetCrossDistillInterface(nn.Module):
    def __init__(self):
        super(UnetCrossDistillInterface, self).__init__()
        self.unet = UNet(n_channels=3, out_channels=3)
    
    def load_ckpt(self, ckpt_path):
        static_dict = torch.load(ckpt_path)['state_dict']
        old_dict = self.unet.state_dict().copy()
        for k, v in static_dict.items():
            if k.startswith('model.unet'):
                old_dict[k.replace('model.unet.', '')] = v
        self.unet.load_state_dict(old_dict)
        
    def forward(self, x):
        return self.unet(x)

