import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import numpy as np

class Model(nn.Module):
    def __init__(self, num_class, **kwargs):
        super().__init__()

        self.inception_v3 = models.inception_v3(pretrained=True)
        self.inception_v3.fc = nn.Linear(2048, num_class)

    def forward(self, x_rgb):
        out = self.inception_v3(x_rgb)
        return out
