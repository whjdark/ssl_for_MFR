'''
Author: whj
Date: 2022-02-28 13:27:06
LastEditors: whj
LastEditTime: 2022-03-05 14:49:45
Description: file content
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


    
class SCNN(nn.Module):
    def __init__(self, num_classes=24, pretraining=True):
        super(SCNN, self).__init__()
        self.num_classes = num_classes
        self.pretraining = pretraining
        
        self.features = models.vgg11(pretrained=self.pretraining).features
        self.adppool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = models.vgg11(pretrained=self.pretraining).classifier
        self.classifier._modules['6'] = nn.Linear(4096, num_classes)
        
        if self.pretraining:
            nn.init.normal_(self.classifier._modules['6'].weight, 0, 0.01)
            nn.init.constant_(self.classifier._modules['6'].bias, 0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.features(x)
        y = self.adppool(y)
        
        return self.classifier(y.view(y.shape[0],-1))    


class MCNN(nn.Module):
    def __init__(self, model, num_classes=24, num_cuts=12):
        super(MCNN, self).__init__()
        self.num_classes = num_classes
        self.num_cuts = num_cuts

        self.features = model.features
        self.adppool = model.adppool
        self.classifier = model.classifier

    def forward(self, x):
        x = x.reshape(-1,3,x.shape[3],x.shape[4])
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_cuts),self.num_cuts,y.shape[-3],y.shape[-2],y.shape[-1]))#(8,12,512,7,7)
        y = self.adppool(torch.max(y,1)[0])
        y = self.classifier(y.view(y.shape[0],-1))
        return y

