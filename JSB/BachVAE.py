import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import joblib
from intomido.functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModifierBlock(nn.Module):
    def __init__(self, heads=8):
        super(ModifierBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim=128, num_heads=heads, batch_first=True)
        self.convQ = nn.Conv2d(1, 1, 16, padding='same')
        self.convK = nn.Conv2d(1, 1, 16, padding='same')
        self.convV = nn.Conv2d(1, 1, 4, padding='same')


    def forward(self, x):
        Q = self.convQ(x)[:, 0, :, :].transpose(2, 1)
        K = self.convK(x)[:, 0, :, :].transpose(2, 1)
        V = self.convV(x)[:, 0, :, :].transpose(2, 1)
        att = self.att(Q, V, K)[0]
        att = att.transpose(2, 1).view(*x.shape)
        return x + att


class Modifier(nn.Module):
    def __init__(self, heads=8, repetition=10):
        super(Modifier, self).__init__()
        self.blocks = []
        for i in range(repetition):
            self.blocks.append(ModifierBlock(heads))
            self.blocks.append(nn.Conv2d(1, 32, 4, padding='same'))
            self.blocks.append(nn.ReLU())
            self.blocks.append(nn.Conv2d(32, 1, 4, padding='same'))
            self.blocks.append(nn.ReLU())

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x

if __name__ == '__main__':
    model = Modifier(1)

    optim = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    dummy = torch.rand(10, 1, 128, 100)
    torch_imshow(dummy)
    torch_imshow(torch.roll(dummy, shifts=(16,), dims=(3,)))
    with torch.no_grad():
        X = model(dummy)
        torch_imshow(X)




