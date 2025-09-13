"""
Generative Adversarial Capsule Bidirectional Temporal Convolutional Neural Network (GAC-BTCNN)
----------------------------------------------------------------------------------------------
Reference implementation (PyTorch) that combines:
  • Generator (G): 1D transposed-convolutional generator for sequence-like data.
  • Discriminator / Classifier (D/C): Bidirectional TCN backbone + Capsule layer + linear head.
  • Capsule layer: dynamic routing (Sabour et al. 2017) adapted to 1D feature maps.
  • Training: Adversarial loss for G/D and supervised classification loss for D on labeled batches.

Input format
------------
This implementation targets sequence data with shape:
    (batch, channels, length)  
Example: protein PSSM segments with 20 channels across L positions.

Quick start
-----------
1) Install deps: pip install torch torchvision torchaudio numpy
2) Replace the DummySequenceDataset with your dataset (tensor or custom Dataset).
3) Run this script. It will:
   - Instantiate G, D/C
   - Train adversarially + classification on a toy dataset
   - Print losses and a sample evaluation metric

Notes
-----
• The Bi-TCN here is realized by a pair of causal TCN stacks run on the sequence forward and backward,
  whose outputs are concatenated.  
• The Capsule layer converts convolutional feature maps into a small set of higher-level capsules via routing.  
• The GAN setup is vanilla (non-saturating), but you can swap in WGAN-GP or other objectives easily.  
• For real projects, tune channels, depths, routing iters, and losses.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Utility: weight init
# -------------------------------

def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# -------------------------------
# Bidirectional TCN building blocks
# -------------------------------

class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        # Causal crop to maintain causality
        out = self.conv(x)
        crop = (self.kernel_size - 1) * self.dilation
        if crop > 0:
            out = out[:, :, :-crop]
        out = F.relu(self.bn(out))
        out = self.dropout(out)
        return F.relu(out + self.res(x[:, :, :out.size(2)]))

class TCNStack(nn.Module):
    def __init__(self, in_ch: int, chs: Tuple[int, ...] = (64, 64, 128, 128), kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_ch
        for i, c in enumerate(chs):
            layers.append(CausalConvBlock(prev, c, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
            prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BiTCN(nn.Module):
    """Bidirectional TCN = forward TCN + backward TCN (on time-reversed input)."""
    def __init__(self, in_ch: int, chs: Tuple[int, ...] = (64, 64, 128, 128), kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn_f = TCNStack(in_ch, chs, kernel_size, dropout)
        self.tcn_b = TCNStack(in_ch, chs, kernel_size, dropout)

    def forward(self, x):
        f = self.tcn_f(x)
        b = self.tcn_b(torch.flip(x, dims=[-1]))
        b = torch.flip(b, dims=[-1])
        return torch.cat([f, b], dim=1)  # concat channels

# -------------------------------
# Capsule layer (routing by agreement)
# -------------------------------

class PrimaryCaps1D(nn.Module):
    def __init__(self, in_ch: int, num_caps: int = 32, cap_dim: int = 8, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.conv = nn.Conv1d(in_ch, num_caps * cap_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        out = self.conv(x)  # (B, num_caps*cap_dim, L')
        B, C, L = out.shape
        out = out.view(B, self.num_caps, self.cap_dim, L)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, L, num_caps, cap_dim)
        # squash nonlinearity
        out = squash(out)
        return out  # (B, L, num_caps, cap_dim)

def squash(s, dim=-1, eps=1e-8):
    norm2 = (s ** 2).sum(dim=dim, keepdim=True)
    scale = norm2 / (1.0 + norm2)
    return scale * s / torch.sqrt(norm2 + eps)

class DenseCaps1D(nn.Module):
    """Dense capsule layer with dynamic routing.
    input: (B, L, N_in, D_in) -> output: (B, N_out, D_out)
    collapses spatial L by averaging predictions.
    """
    def __init__(self, n_in: int, d_in: int, n_out: int, d_out: int, iters: int = 3):
        super().__init__()
        self.n_in, self.d_in, self.n_out, self.d_out, self.iters = n_in, d_in, n_out, d_out, iters
        self.W = nn.Parameter(0.01 * torch.randn(1, n_in, n_out, d_out, d_in))

    def forward(self, x):
        # x: (B, L, n_in, d_in)
        B, L, n_in, d_in = x.shape
        assert n_in == self.n_in and d_in == self.d_in
        x_exp = x.mean(dim=1, keepdim=False)  # (B, n_in, d_in) aggregate over length
        x_exp = x_exp.unsqueeze(2).unsqueeze(0)  # (1, B, n_in, 1, d_in)
        x_exp = x_exp.permute(1, 2, 0, 3, 4)    # (B, n_in, 1, 1, d_in)
        # u_hat: (B, n_in, n_out, d_out)
        W = self.W.repeat(B, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x_exp[..., None]).squeeze(-1)

        b = torch.zeros(B, self.n_in, self.n_out, device=x.device)
        for _ in range(self.iters):
            c = F.softmax(b, dim=-1)  # (B, n_in, n_out)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # (B, n_out, d_out)
            v = squash(s, dim=-1)
            b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v  # (B, n_out, d_out)

# -------------------------------
# Discriminator / Classifier (BiTCN + Capsules)
# -------------------------------

class DiscriminatorCapsBiTCN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 2, tcn_chs=(64,64,128,128), dropout=0.1,
                 primary_caps=32, primary_dim=8, dense_caps=16, dense_dim=16):
        super().__init__()
        self.backbone = BiTCN(in_ch, chs=tcn_chs, kernel_size=3, dropout=dropout)
        out_ch = 2 * tcn_chs[-1]
        self.primary_caps = PrimaryCaps1D(out_ch, num_caps=primary_caps, cap_dim=primary_dim, kernel_size=3, stride=2)
        self.dense_caps = DenseCaps1D(n_in=primary_caps, d_in=primary_dim, n_out=dense_caps, d_out=dense_dim, iters=3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_caps * dense_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, L)
        h = self.backbone(x)
        pc = self.primary_caps(h)           # (B, L', P, D)
        dc = self.dense_caps(pc)            # (B, N, D)
        logits = self.classifier(dc)
        return logits

# -------------------------------
# Generator (1D)
# -------------------------------

class Generator1D(nn.Module):
    def __init__(self, z_dim: int, out_ch: int, out_len: int, base_ch: int = 128):
        super().__init__()
        self.out_len = out_len
        # Project and reshape to (base_ch, out_len/8)
        self.proj_len = out_len // 8
        self.fc = nn.Linear(z_dim, base_ch * self.proj_len)
        self.tconv1 = nn.ConvTranspose1d(base_ch, base_ch//2, 4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose1d(base_ch//2, base_ch//4, 4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose1d(base_ch//4, out_ch, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(base_ch//2)
        self.bn2 = nn.BatchNorm1d(base_ch//4)

    def forward(self, z):
        x = self.fc(z)  # (B, base_ch*proj_len)
        x = x.view(z.size(0), -1, self.proj_len)  # (B, base_ch, L/8)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = torch.tanh(self.tconv3(x))  # output in [-1,1]
        return x[:, :, : self.out_len]

# -------------------------------
# Toy dataset (replace with real sequences)
# -------------------------------

class DummySequenceDataset(Dataset):
    """Generates synthetic (channels=20, length=256) sequences and binary labels.
    Replace with your real dataset (e.g., PSSM tensors)."""
    def __init__(self, n: int = 2048, length: int = 256, channels: int = 20):
        super().__init__()
        self.length = length
        self.channels = channels
        self.X = torch.randn(n, channels, length)
        # Inject a simple pattern for class 1
        self.y = torch.randint(0, 2, (n,))
        for i in range(n):
            if self.y[i] == 1:
                pos = random.randint(0, length-16)
                self.X[i, :5, pos:pos+16] += 1.5

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# Training utils
# -------------------------------

@dataclass
class TrainConfig:
    z_dim: int = 128
    out_len: int = 256
    in_ch: int = 20
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    batch_size: int = 32
    epochs: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def adversarial_training_loop():
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    # Data
    train_ds = DummySequenceDataset(n=2048, length=cfg.out_len, channels=cfg.in_ch)
    val_ds = DummySequenceDataset(n=256, length=cfg.out_len, channels=cfg.in_ch)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    # Models
    G = Generator1D(cfg.z_dim, cfg.in_ch, cfg.out_len).to(device)
    D = DiscriminatorCapsBiTCN(cfg.in_ch, num_classes=2).to(device)
    G.apply(kaiming_init)
    D.apply(kaiming_init)

    # Optims
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    # Losses
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    def d_step(real_x, labels):
        B = real_x.size(0)
        # Supervised classification loss on real
        logits_real = D(real_x)
        cls_loss = ce(logits_real, labels)

        # Adversarial loss (real vs fake)
        z = torch.randn(B, cfg.z_dim, device=device)
        fake_x = G(z).detach()
        # D-head: we reuse class logits; make a real/fake head via pooled feature: simplest -> use max logit as score
        # Alternatively, use a dedicated real/fake linear head; here we approximate with class-1 logit.
        # To be explicit, we add a small real/fake head:
        # For simplicity, compute a scalar realism by margin between top-1 and top-2 logits.
        rf_real = (logits_real.topk(2, dim=1).values[:, 0] - logits_real.topk(2, dim=1).values[:, 1]).unsqueeze(1)
        logits_fake = D(fake_x)
        rf_fake = (logits_fake.topk(2, dim=1).values[:, 0] - logits_fake.topk(2, dim=1).values[:, 1]).unsqueeze(1)
        # Build labels for adversarial head
        y_real = torch.ones(B, 1, device=device)
        y_fake = torch.zeros(B, 1, device=device)
        adv_loss = bce(rf_real, y_real) + bce(rf_fake, y_fake)

        loss = cls_loss + adv_loss
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        return loss.item(), cls_loss.item(), adv_loss.item()

    def g_step(B):
        z = torch.randn(B, cfg.z_dim, device=device)
        fake_x = G(z)
        logits_fake = D(fake_x)
        rf_fake = (logits_fake.topk(2, dim=1).values[:, 0] - logits_fake.topk(2, dim=1).values[:, 1]).unsqueeze(1)
        y_real = torch.ones(B, 1, device=device)
        adv_loss = bce(rf_fake, y_real)
        opt_g.zero_grad()
        adv_loss.backward()
        opt_g.step()
        return adv_loss.item()

    def evaluate(loader):
        D.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = D(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        D.train()
        return correct / max(1, total)

    for epoch in range(1, cfg.epochs + 1):
        for i, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            d_loss, cls_loss, adv_loss_d = d_step(x, y)
            g_loss = g_step(x.size(0))
            if i % 50 == 0:
                print(f"Epoch {epoch:02d} Iter {i:04d} | D:{d_loss:.3f} (cls {cls_loss:.3f}, adv {adv_loss_d:.3f}) | G:{g_loss:.3f}")
        acc = evaluate(val_loader)
        print(f"[Epoch {epoch:02d}] Val Acc: {acc*100:.2f}%")


if __name__ == "__main__":
    adversarial_training_loop()
