import torch
import torch.nn as nn
from torchvision.models import resnet18


class Base_net(nn.Module):
    def __init__(self, image_size=1024, pretrain=False, classes=6):
        super(Base_net, self).__init__()
        self.model = resnet18()
        self.fc = nn.Linear(1000, classes)

    def forward(self, x):
        return self.fc(self.model(x))
