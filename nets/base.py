import torch
import torch.nn as nn
import torchvision

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

        model = torchvision.models.resnet18(pretrained=pretrain)
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        feat_channel = list(model.children())[-1].in_features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, classes),
        )

    def forward(self, x):
        f = self.encoder(widthN_to_bsN(x, self.N))
        return self.classifier(bsN_to_widthN(f, self.N))


class Resnet18_asigm(nn.Module):
    def __init__(self, pretrain=True, classes=6, N=25):
        super(Resnet18_asigm, self).__init__()
        self.N = N
        self.classes = classes

        model = torchvision.models.resnet18(pretrained=pretrain)
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        feat_channel = list(model.children())[-1].in_features
        self.penultimate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, classes),
            nn.Softmax(dim=1),
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, classes),
        )

    def forward(self, x):
        f = self.encoder(widthN_to_bsN(x, self.N))
        f = bsN_to_widthN(f, self.N)
        p = self.penultimate(f)
        a = self.final(f)
        return (self.classes - 1) * (a * p).sum(dim=1).sigmoid()


class Resnet50_asigm(nn.Module):
    def __init__(self, pretrain=True, classes=6, N=25):
        super(Resnet50_asigm, self).__init__()
        self.N = N
        self.classes = classes

        model = torchvision.models.resnet50(pretrained=pretrain)
        self.encoder = nn.Sequential(*list(model.children())[:-2])
        feat_channel = list(model.children())[-1].in_features
        self.penultimate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, classes),
            nn.Softmax(dim=1),
        )
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channel, classes),
        )

    def forward(self, x):
        f = self.encoder(widthN_to_bsN(x, self.N))
        f = bsN_to_widthN(f, self.N)
        p = self.penultimate(f)
        a = self.final(f)
        return (self.classes - 1) * (a * p).sum(dim=1).sigmoid()
