import torch
import torch.nn as nn


def widthN_to_bsN(x, N):
    # x: (bs, c, h, w*N) -> (N*bs, c, h, w)
    bs, c, h, w = x.shape
    w = w // N
    x = torch.stack(x.split(w, dim=-1), dim=1).view(-1, c, h, w)
    return x

def bsN_to_widthN(x, N):
    # x: (N*bs, c, h, w) -> (bs, c, h, w*N)
    bs, c, h, w = x.shape
    bs = bs // N
    x = torch.stack(x.split(N, dim=0)).permute(0, 2, 3, 1, 4).reshape(bs, c, h, -1)
    return x


class Resnet18(nn.Module):
    def __init__(self, pretrain=True, classes=6, N=25):
        super(Resnet18, self).__init__()
        self.N = N

        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        feat_channel = list(model.children())[-1].in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, 64),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        f = self.encoder(widthN_to_bsN(x, self.N))
        return self.classifier(bsN_to_widthN(f, self.N))
