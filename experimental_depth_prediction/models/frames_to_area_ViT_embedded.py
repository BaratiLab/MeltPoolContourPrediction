import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from experimental_depth_prediction.models.vit.model import ViT
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class VitFineTune(nn.Module):
    def __init__(self,  num_features = 4, embed_only = True, pretrained = True):
        super(VitFineTune, self).__init__()
        self.vit = ViT('B_16_imagenet1k', pretrained = True, embed_only = embed_only)
        if embed_only:
            self.fc5 = nn.Linear(768, num_features)
        self.embed_only = embed_only
        self.resize= torchvision.transforms.Resize((384,384))
    def forward(self, x):
        x = self.vit(self.resize(x))
        if self.embed_only:
            x = x[:, 0]
        x = x.view(x.shape[0], -1)        

        if self.embed_only:
            x = self.fc5(x)
        return x

class VitFromScratch(nn.Module):
    def __init__(self,  num_features=4, image_size=64,  embed_only=True, pretrained=True, n_layers=12, ff_dim=3072, num_heads=12, dim=768):
        super(VitFromScratch, self).__init__()
        self.vit = ViT(
            pretrained=False,
            embed_only=embed_only,
            image_size=image_size,
            num_layers=n_layers,
            ff_dim=ff_dim,
            num_heads=num_heads,
            dim=dim
        )
        if embed_only:
            self.fc5 = nn.Linear(dim, num_features)
        self.embed_only = embed_only
        self.resize = torchvision.transforms.Resize((64, 64))

    def forward(self, x):
        x = self.vit(x)
        if self.embed_only:
            x = x[:, 0]
        x = x.view(x.shape[0], -1)
        if self.embed_only:
            x = self.fc5(x)
        return x
