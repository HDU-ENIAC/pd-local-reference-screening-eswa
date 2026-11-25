"""Final local-moment network. Inputs are windows, not continuous streams."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize_reference(x, refs):
    """Scale queries and references using only reference edge samples.

    x: (N,512); refs: (K,512). No subtraction of the waveform mean.
    The historical helper called this RMS; it actually used population STD.
    """
    x, refs = np.asarray(x), np.asarray(refs)
    if x.ndim != 2 or refs.ndim != 2 or x.shape[1] != 512 or refs.shape[1] != 512:
        raise ValueError("Expected query and reference arrays with 512 samples per row")
    if len(refs) == 0 or not np.isfinite(x).all() or not np.isfinite(refs).all():
        raise ValueError("Inputs must be finite and references nonempty")
    quiet = np.concatenate([refs[:, :32], refs[:, 448:]], axis=1).astype(float).reshape(-1)
    sigma = max(float(np.std(quiet)), 1e-4)
    return (x / sigma).astype(np.float32), (refs / sigma).astype(np.float32), sigma


class LogPowerSTFT(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.hann_window(64, periodic=True, dtype=torch.float64)
        self.register_buffer("window", (w / w.norm()).float())

    def forward(self, x):
        z = torch.stft(x, 64, hop_length=4, window=self.window, center=True,
                       pad_mode="constant", return_complex=True)[:, 1:, :128]
        return torch.log1p(z.abs().square())


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(z)) + x)


class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("input_mean", torch.tensor(0.))
        self.register_buffer("input_std", torch.tensor(1.))
        self.conv1 = nn.Conv2d(1, 64, 7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(), BasicBlock())
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
        self.adapter = nn.Conv2d(64, 24, 1)

    def forward(self, x):
        z = (x - self.input_mean) / self.input_std
        z = F.relu(self.bn1(self.conv1(z)))
        z = F.max_pool2d(z, 3, stride=(2, 1), padding=1)
        return self.adapter(self.layer1(z))


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 7, padding=3*dilation, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.mix(F.silu(self.conv(x)))


class LocalReferenceNet(nn.Module):
    """Return 512 interpolated logits; use window_scores for admission.

    State-dictionary keys match the final archived checkpoints. The positive
    channel scale is learned; reference variance is a separate input, not the
    denominator of the residual. No ridge/subspace operation is included.
    """
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.floor = nn.Parameter(torch.zeros(1, 24, 1, 1))
        self.project = nn.Conv2d(48, 24, 1)
        self.body = nn.Sequential(TemporalBlock(24, 4), TemporalBlock(24, 8))
        self.head = nn.Conv1d(24, 1, 1)
        self.register_buffer("interpolation", torch.tensor(
            [.25, .5, .75, 1., .75, .5, .25]).reshape(1, 1, 7))
        nn.init.constant_(self.head.bias, -5.)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
        return self

    def forward(self, x, refs, inverse=None):
        n, k, f, t = refs.shape
        q = self.encoder(x[:, None])
        z = self.encoder(refs.reshape(n*k, 1, f, t)).reshape(n, k, *q.shape[1:])
        if inverse is not None:
            z = z[inverse]
        mu, var = z.mean(1), z.var(1, unbiased=False)
        scale = F.softplus(self.floor) + 1e-3
        residual = (q - mu) / torch.sqrt(scale)
        h = F.silu(self.project(torch.cat([residual, torch.log1p(var)], 1))).mean(2)
        logits = self.head(self.body(h))
        dense = F.conv_transpose1d(logits, self.interpolation, stride=4, padding=3)
        return torch.cat([dense, dense[:, :, -1:].expand(-1, -1, 3)], dim=2)[:, 0]

    def window_scores(self, x, refs, inverse=None):
        return self(x, refs, inverse)[:, 32:448].amax(1)


@torch.no_grad()
def score_windows(net, x, refs, batch_size=64):
    """One condition/reference set per call. Returns uncalibrated logits."""
    x, refs, _ = normalize_reference(x, refs)
    device = next(net.parameters()).device
    tf = LogPowerSTFT().to(device)
    r = tf(torch.from_numpy(refs).to(device))[None]
    out = []
    net.eval()
    for i in range(0, len(x), batch_size):
        q = tf(torch.from_numpy(x[i:i+batch_size]).to(device))
        inverse = torch.zeros(len(q), dtype=torch.long, device=device)
        out.append(net.window_scores(q, r, inverse).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
