import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1),
            nn.InstanceNorm2d(64 * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1),
            nn.InstanceNorm2d(64 * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 1, 1),
            nn.InstanceNorm2d(64 * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        residuals = []
        for _ in range(9):
            residuals += [
                nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1, padding_mode="reflect"),
                nn.InstanceNorm2d(64 * 4),
                nn.ReLU(True),
                nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1, padding_mode="reflect"),
                nn.InstanceNorm2d(64 * 4),
                nn.Identity(),
            ]

        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, 7, 1, 3, padding_mode="reflect"),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64 * 2, 3, 2, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(64 * 2),
            nn.ReLU(True),
            nn.Conv2d(64 * 2, 64 * 4, 3, 2, 1, padding_mode="reflect"),
            nn.InstanceNorm2d(64 * 2),
            nn.ReLU(True),
            *residuals,
            nn.ConvTranspose2d(64 * 4, 64 * 2, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(
                64 * 1,
                channels,
                7,
                1,
                3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
