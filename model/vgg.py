import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
__all__ = ['VGG13','OpenVGG13']

class VGG(nn.Module):
    
    def __init__(self, num_classes=10, backbone_fc=False):
        super(VGG, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(in_channels=256,
            #           out_channels=256,
            #           kernel_size=(3, 3),
            #           stride=(1, 1),
            #           padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(in_channels=512,
            #           out_channels=512,
            #           kernel_size=(3, 3),
            #           stride=(1, 1),
            #           padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(in_channels=512,
            #           out_channels=512,
            #           kernel_size=(3, 3),
            #           stride=(1, 1),
            #           padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )
        if backbone_fc:
            self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            )

    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        if hasattr(self, 'classifier'):
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

class OpenVGG(nn.Module):
    
    def __init__(self, num_classes=10, backbone_fc=False):
        super(OpenVGG, self).__init__()
        self.block_11 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )
        self.block_12 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_21 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_22 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )    
        
        self.block_31 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_41 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_51 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )
        self.block_32 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_42 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_52 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

    def forward(self, img, spec):
        x1 = self.block_11(img)
        x2 = self.block_12(spec)
        x1 = self.block_21(x1)
        x2 = self.block_22(x2)
        x1 = self.block_31(x1)
        x2 = self.block_32(x2)
        x1 = self.block_41(x1)
        x2 = self.block_42(x2)
        x1 = self.block_51(x1)
        x2 = self.block_52(x2)
        return x1,x2

def VGG13(num_classes=10, backbone_fc=False):
    return VGG(num_classes, backbone_fc=backbone_fc)

def OpenVGG13(num_classes=10, backbone_fc=False):
    return OpenVGG(num_classes, backbone_fc=backbone_fc)
