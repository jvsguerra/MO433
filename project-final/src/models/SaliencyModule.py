import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101


class SPM(object):
    """ Saliency Prediction Module (SPM) """
    def __init__(self, in_channels=256, out_channels=1):
        super(SPM, self).__init__()
        self.ResNet = deeplabv3_resnet101(pretrained=True)
        self.ResNet.classifier[4] = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.ResNet.aux_classifier[4] = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        
    def build(self):
        return self.ResNet
