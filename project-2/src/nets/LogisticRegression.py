from torch.nn import functional as F
import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(65536, 1)
    def forward(self, x):
        return F.sigmoid(self.linear(x))
