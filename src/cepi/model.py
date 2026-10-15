import torch
from torch import nn
from torch.nn import functional as F


class Restoration(nn.Module):
    def __init__(self, width, cameras):
        super().__init__()
        self.camera = nn.Embedding(cameras, 4)
        self.enc = nn.Sequential(
            nn.Conv2d(7, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.low = nn.Sequential(
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1),
            nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(3 * width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 2, 3, padding=1),
        )

    def forward(self, y, exposure, gain, camera):
        fields = torch.cat(
            [
                torch.log(exposure.clamp_min(1e-8))[:, None],
                torch.log(gain.clamp_min(1e-8))[:, None],
                self.camera(camera),
            ],
            dim=1,
        )
        fields = fields[:, :, None, None].expand(-1, -1, *y.shape[-2:])
        normalized = y / (exposure * gain)[:, None, None, None]
        high = self.enc(torch.cat([normalized, fields], dim=1))
        low = self.low(F.avg_pool2d(high, 2))
        outputs = self.dec(
            torch.cat(
                [
                    high,
                    F.interpolate(
                        low, size=high.shape[-2:], mode="bilinear", align_corners=False
                    ),
                ],
                dim=1,
            )
        )
        return F.softplus(outputs[:, :1]), F.softplus(outputs[:, 1:])


def predictive(mean, variance, exposure, gain, noise, floor, variant):
    scale = (exposure * gain)[:, None, None, None]
    if variant == "wrong_exposure":
        scale = scale * 2
    if variant == "wrong_gain":
        noise = noise * 2
    observation = scale * mean
    a, b, row, quant = [noise[:, i, None, None, None] for i in range(4)]
    sensor = a * observation + b + row + quant
    if variant == "homoscedastic":
        sensor = torch.ones_like(sensor) * (b + row + quant)
    latent = 0 if variant == "deterministic" else scale.square() * variance
    return observation, (sensor + latent).clamp_min(floor)


def nll(target, mean, variance, mask):
    if not mask.any():
        raise ValueError("No valid pixels in paired observation")
    values = (target - mean).square() / (2 * variance) + 0.5 * variance.log()
    return values[mask].mean()
