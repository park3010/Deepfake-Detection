import torch
import torch.nn as nn
import torchvision

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False) 
        
        self.features = None  # 用于存储特征的变量
        # 定义 hook 函数
        def forward_hook(module, input, output):
            self.features = output
        # 注册 hook 到 avgpool 层，该层输出是全连接层之前的特征
        self.resnet.avgpool.register_forward_hook(forward_hook)
    def get_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, inp):
        feat = self.get_features(inp)
        return feat
    
    def get_config(self, ckpt_path='/home/oem/deepfake/Ourmethod/comparison/IDCNet/idcnet/weight/resnet50-0676ba61.pth'):
        # load the pre-trained model
        static_dict = torch.load(ckpt_path)
        old_dict = self.resnet.state_dict().copy()
        for k, v in static_dict.items():
            if k in old_dict:
                old_dict[k] = v
        self.resnet.load_state_dict(old_dict)
        print(f'{ckpt_path} loaded successfully')
        config = {
                'name': 'ResNet34',
                'network': self,
                'feat_dim': 2048
        }
        return config

if __name__ == '__main__':
    resnet = ResNet50()
    config = resnet.get_config()
    print(config)
    inp = torch.rand(2, 3, 224, 224)
    out = resnet(inp)
    print(out.shape)