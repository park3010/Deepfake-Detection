import torch
import torch.nn as nn
import efficientnet_pytorch as efn

class Efficientnet(nn.Module):
    def __init__(self, pretrain='efficientnet-b4', sbi=None, dropout_rate=0.2, drop_connect_rate=0):
        super(Efficientnet, self).__init__()
        self.model = efn.EfficientNet.from_pretrained(pretrain,
                                                      weights_path='/home/oem/deepfake/Ourmethod/comparison/IDCNet/idcnet/weight/adv-efficientnet-b4-44fb3a87.pth',
                                                      dropout_rate=dropout_rate,
                                                      drop_connect_rate=drop_connect_rate)

        if pretrain == 'efficientnet-b4':
            self.conv = nn.Conv2d(1792, 512, 1)
        elif pretrain == 'efficientnet-b1':
            self.conv = nn.Conv2d(1280, 512, 1)
        elif pretrain == 'efficientnet-b3':
            self.conv = nn.Conv2d(1536, 512, 1)
        elif pretrain == 'efficientnet-b5':
            self.conv = nn.Conv2d(2048, 512, 1)
        elif pretrain == 'efficientnet-b6':
            self.conv = nn.Conv2d(2304, 512, 1)
        else:
            raise ValueError('pretrain is not supported')

        # self.channel_adjust_conv = nn.Conv2d(2424, 512, 1)

    def features(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)

        return x
    
if __name__ == '__main__':
    model = Efficientnet(pretrain='efficientnet-b4')
    inp = torch.randn(1, 3, 512, 512)
    out = model(inp)
    print(out.shape)
    feat = model.features(inp)
    print(feat.shape)