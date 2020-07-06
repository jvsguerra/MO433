import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        
        # Encoder f(.)
        self.f = nn.Sequential(*self.f)

        # Projection head g(.)
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        # Encoder pass
        h = self.f(x)
        h = torch.flatten(h, start_dim=1)
        # Projection Head
        z = self.g(h)
        # Normalize h and z
        h = F.normalize(h, dim=-1)
        z = F.normalize(z, dim=-1)
        return h, z