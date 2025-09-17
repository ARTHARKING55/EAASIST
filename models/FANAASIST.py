import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# FAN Block
class FANBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_p = nn.Linear(in_dim, out_dim // 4)
        self.linear_g = nn.Linear(in_dim, out_dim - (out_dim // 2))
        self.act = nn.GELU()
        self.gate = nn.Parameter(torch.randn(1))

    def forward(self, x):
        p = self.linear_p(x)
        g = self.act(self.linear_g(x))
        gate = torch.sigmoid(self.gate)
        return torch.cat([gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g], dim=-1)

# Heterogeneos Fusion FAN Block
class HtrgFANBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fan1 = FANBlock(in_dim, out_dim)
        self.fan2 = FANBlock(in_dim, out_dim)
        self.master_proj = nn.Linear(in_dim, out_dim)  
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=True)
        
    def forward(self, x1, x2, master=None):
        x1 = self.fan1(x1)
        x2 = self.fan2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        master = self.master_proj(master)
        out = self._apply_BN(x)
        return x1, x2, master

    def _apply_BN(self, x):
        B, N, C = x.shape
        x = self.bn(x.view(-1, C))
        return self.act(x.view(B, N, C))

class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: float = 0.3):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p)
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        scores = self.sigmoid(self.proj(Z))
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores, h, k):
        B, N, C = h.shape
        K = max(int(N * k), 1)
        _, idx = torch.topk(scores, K, dim=1)
        idx = idx.expand(-1, -1, C)
        h = h * scores
        return torch.gather(h, 1, idx)

class CONV(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports 1 input channel")
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.hsupp = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        self.band_pass = self._create_filters()

    def _create_filters(self):
        NFFT = 512
        f = np.linspace(0, self.sample_rate / 2, int(NFFT / 2) + 1)
        mel = 2595 * np.log10(1 + f / 700)
        mel_min, mel_max = mel.min(), mel.max()
        mel_bands = np.linspace(mel_min, mel_max, self.out_channels + 1)
        hz_bands = 700 * (10**(mel_bands / 2595) - 1)

        filters = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(self.out_channels):
            fmin, fmax = hz_bands[i], hz_bands[i + 1]
            hHigh = 2*fmax/self.sample_rate * np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = 2*fmin/self.sample_rate * np.sinc(2*fmin*self.hsupp/self.sample_rate)
            filters[i] = torch.tensor(np.hamming(self.kernel_size) * (hHigh - hLow))
        return filters

    def forward(self, x, mask=False):
        filt = self.band_pass.clone().to(x.device)
        if mask:
            A = random.randint(0, 20)
            A0 = random.randint(0, filt.shape[0] - A)
            filt[A0:A0+A] = 0
        filters = filt.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=0)

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.selu = nn.SELU(inplace=True)
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2, 3), padding=(0, 1))
        self.mp = nn.MaxPool2d((1, 3))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], (1, 3), padding=(0, 1))

    def forward(self, x):
        identity = x
        out = self.conv1(x if self.first else self.selu(x))
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        return self.mp(out + identity)

# FAN-AASIST Model
class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        filts, gat_dims = d_args["filts"], d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        self.conv_time = CONV(filts[0], d_args["first_conv"])
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            *[Residual_block(f, first=(i==0)) for i, f in enumerate(filts[1:])]
        )
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = FANBlock(filts[-1][-1], gat_dims[0])
        self.GAT_layer_T = FANBlock(filts[-1][-1], gat_dims[0])
        self.HtrgGAT_layers = nn.ModuleList([
            HtrgFANBlock(gat_dims[0], gat_dims[1]),
            HtrgFANBlock(gat_dims[1], gat_dims[1]),
            HtrgFANBlock(gat_dims[0], gat_dims[1]),
            HtrgFANBlock(gat_dims[1], gat_dims[1])
        ])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0])
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0])
        self.pool_h = nn.ModuleList([
            GraphPool(pool_ratios[2], gat_dims[1]),
            GraphPool(pool_ratios[2], gat_dims[1]),
            GraphPool(pool_ratios[2], gat_dims[1]),
            GraphPool(pool_ratios[2], gat_dims[1])
        ])
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        self.drop = nn.Dropout(0.5)
        self.drop_way = nn.Dropout(0.2)

    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug).unsqueeze(1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.selu(self.first_bn(x))
        e = self.encoder(x)
        e_S = torch.max(torch.abs(e), dim=3)[0].transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        e_T = torch.max(torch.abs(e), dim=2)[0].transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        master1 = self.master1.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layers[0](out_T, out_S, master1)
        out_S1, out_T1 = self.pool_h[0](out_S1), self.pool_h[1](out_T1)
        T_aug, S_aug, m_aug = self.HtrgGAT_layers[1](out_T1, out_S1, master1)
        out_T1, out_S1, master1 = out_T1 + T_aug, out_S1 + S_aug, master1 + m_aug

        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T2, out_S2, master2 = self.HtrgGAT_layers[2](out_T, out_S, master2)
        out_S2, out_T2 = self.pool_h[2](out_S2), self.pool_h[3](out_T2)
        T_aug, S_aug, m_aug = self.HtrgGAT_layers[3](out_T2, out_S2, master2)
        out_T2, out_S2, master2 = out_T2 + T_aug, out_S2 + S_aug, master2 + m_aug

        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))
        T_max, T_avg = out_T.abs().max(dim=1)[0], out_T.mean(dim=1)
        S_max, S_avg = out_S.abs().max(dim=1)[0], out_S.mean(dim=1)
        last_hidden = self.drop(torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1))
        return self.out_layer(last_hidden)
