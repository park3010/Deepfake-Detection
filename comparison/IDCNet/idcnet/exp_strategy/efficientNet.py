import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from networks.effient import Efficientnet

class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        self.efficientnet = Efficientnet()
    
    def forward(self, x):
        x = self.efficientnet.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x
    
    def get_config(self):
        print(f'efficientNetB4 loaded successfully')
        config = {
                'name': 'EfficientNetB4',
                'network': self,
                'feat_dim': 512
        }
        return config
    
if __name__ == '__main__':
    model = EfficientNetB4()
    print(model.get_config())
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)