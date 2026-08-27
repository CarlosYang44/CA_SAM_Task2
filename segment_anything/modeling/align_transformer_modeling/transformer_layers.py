# Modified from PyTorch nn.Transformer

import math
from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                attn_mask: bool = None,
                key_padding_mask: bool = None) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x

        r = x
        x = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return r + self.dropout(x)


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
class CrossAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim,
                                                nhead,
                                                dropout=dropout,
                                                batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: bool = None,
                *,
                need_weights: bool = False) -> (torch.Tensor, torch.Tensor): # type: ignore
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x
        x, weights = self.cross_attn(q,
                                     k,
                                     v,
                                     attn_mask=attn_mask,
                                     need_weights=need_weights,
                                     average_attn_weights=False)

        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights


class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        # pixel: [B, C, H, W]
        # pixel_flat: [B, H*W, C]
        bs, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()    # [B, C, H, W]

        x = self.conv(pixel_flat)   # [B, C, H, W]
        return x


class OutputFFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


class CAResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

        t = int((abs(math.log2(out_dim)) + 1) // 2)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1) # 
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        if self.residual:
            if in_dim == out_dim:
                self.downsample = nn.Identity()
            else:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        r = x
        x = self.conv1(F.relu(x))   # [B, C, H, W]
        x = self.conv2(F.relu(x))   # [B, C, H, W]

        b, c = x.shape[:2]
        w = self.pool(x).view(b, 1, c)                              # B*1*C
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1).sigmoid()  # B*C*1*1

        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w

        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x
    

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))