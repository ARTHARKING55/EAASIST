import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Union


# Mamba Block 
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=3, d_expand=2):
        super().__init__()
        self.inner_dim = d_model * d_expand
        self.linear_proj = nn.Linear(d_model, self.inner_dim * 2)
        self.conv = nn.Conv1d(self.inner_dim, self.inner_dim, kernel_size=d_conv, padding=d_conv // 2, groups=d_expand)
        self.linear_out = nn.Linear(self.inner_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_proj = self.linear_proj(x)
        u, v = x_proj.chunk(2, dim=-1)
        u = u.transpose(1, 2)
        u = self.conv(u)
        u = u.transpose(1, 2)
        x = self.linear_out(F.silu(u * v))
        return self.norm(x)

# BiMamba Block
class BidirectionalMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_forward = MambaBlock(d_model)
        self.mamba_backward = MambaBlock(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        x_fwd = self.mamba_forward(x)
        x_bwd = self.mamba_backward(torch.flip(x, dims=[1]))
        x_bwd = torch.flip(x_bwd, dims=[1])
        x = torch.cat([x_fwd, x_bwd], dim=-1)
        return self.fuse(x)
        
# Heteerogeneous Fusion Mamba Block
class HtrgMambaBlock(nn.Module):
    def __init__(self, in_dim, out_dim, master_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.master_dim = master_dim if master_dim is not None else out_dim

        self.mamba_1 = BidirectionalMamba(in_dim)
        self.mamba_2 = BidirectionalMamba(in_dim)

        self.out_proj_1 = nn.Linear(in_dim, out_dim)
        self.out_proj_2 = nn.Linear(in_dim, out_dim)

        self.master_proj = nn.Linear(self.master_dim, out_dim)

        self.master_update = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x1, x2, master):
        x1_mamba = self.mamba_1(x1)
        x2_mamba = self.mamba_2(x2)

        x1_out = self.out_proj_1(x1_mamba)  
        x2_out = self.out_proj_2(x2_mamba)

        branch_avg = (x1_out.mean(1) + x2_out.mean(1)) / 2  
        master_proj = self.master_proj(master.squeeze(1))   

        fused = branch_avg + master_proj                    
        master_update = self.master_update(fused)           

        return self.norm(x1_out), self.norm(x2_out), master_update.unsqueeze(1)  


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1, stride=1, padding=0, dilation=1, bias=False, groups=1, mask=False):
        super().__init__()
        if in_channels != 1:
            raise ValueError("SincConv only supports one input channel.")
        self.out_channels = out_channels
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        NFFT = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)
        self.mel = filbandwidthsf
        self.band_pass = torch.zeros(out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / sample_rate) * np.sinc(2 * fmax * self.hsupp / sample_rate)
            hLow = (2 * fmin / sample_rate) * np.sinc(2 * fmin * self.hsupp / sample_rate)
            hideal = hHigh - hLow
            self.band_pass[i, :] = torch.Tensor(np.hamming(self.kernel_size)) * torch.Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None)

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm2d(nb_filts[0]) if not first else nn.Identity()
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.selu = nn.SELU(inplace=False)
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.downsample = (nb_filts[0] != nb_filts[1])
        self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1)) if self.downsample else nn.Identity()
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.selu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return self.mp(out)


# BiMambaAASIST Model
class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        self.conv_time = CONV(out_channels=filts[0], kernel_size=d_args["first_conv"], in_channels=1)
        self.first_bn = nn.BatchNorm2d(1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=False)
        self.encoder = nn.Sequential(*[Residual_block(nb_filts=f, first=(i == 0)) for i, f in enumerate(filts[1:])])
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = BidirectionalMamba(filts[-1][-1])
        self.GAT_layer_T = BidirectionalMamba(filts[-1][-1])
        self.HtrgGAT_layer_ST11 = HtrgMambaBlock(gat_dims[0], gat_dims[1], master_dim=gat_dims[0])
        self.HtrgGAT_layer_ST12 = HtrgMambaBlock(gat_dims[1], gat_dims[1], master_dim=gat_dims[1])
        self.HtrgGAT_layer_ST21 = HtrgMambaBlock(gat_dims[0], gat_dims[1], master_dim=gat_dims[0])
        self.HtrgGAT_layer_ST22 = HtrgMambaBlock(gat_dims[1], gat_dims[1], master_dim=gat_dims[1])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        e = self.encoder(x)
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        return self.out_layer(last_hidden)
