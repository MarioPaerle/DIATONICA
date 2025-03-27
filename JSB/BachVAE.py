import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BachBlock(nn.Module):
    def __init__(self, heads=1):
        super().__init__()
        layers = []

        for head in range(heads):
            layers.append(nn.Conv2d(1, 64, kernel_size=(16, 16), padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(64, 64, kernel_size=(8, 8), padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(64, 1, kernel_size=(4, 4), padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mod(x) + x
        return x


class BachVAE(nn.Module):
    def __init__(self):
        super(BachVAE, self).__init__()
        self.preprocesser = BachBlock(8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(16, 16), stride=(4, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=(8, 8), stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 2, kernel_size=(4, 4)),
            nn.LeakyReLU(0.2)
        )

        self.d1 = nn.Dropout(0.2)
        self.ztodec = nn.Linear(32, 1024)

        self.mean_layer = nn.Linear(240, 32)
        self.logvar_layer = nn.Linear(240, 32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 256, kernel_size=(4, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 256, kernel_size=(9, 7), stride=(2, 2)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 1, kernel_size=(16, 16), stride=(4, 2)),
            nn.LeakyReLU(0.2)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = self.d1(x)
        x = x.flatten(1)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        z = self.ztodec(z)
        z = z.view(z.shape[0], 8, 8, 16)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


model = BachNet(16) #  with 8 "heads" it performs as well as 16
if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()

optim = Adam(model.parameters(), lr=0.01)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
