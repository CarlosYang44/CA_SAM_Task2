# feature_pool.py
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import torch.nn as nn
import math

@torch.no_grad()
def extract_feature_for_vae(
    image_embeddings: torch.Tensor,
    mode: str = "gap",
    *,
    cls_token: torch.Tensor = None,
    patch_tokens: torch.Tensor = None,
    attn_temp: float = 1.0
) -> torch.Tensor:
    assert isinstance(image_embeddings, torch.Tensor), "image_embeddings must be Tensor"


    if mode == "cls":
        assert cls_token is not None, "mode='cls' requires cls_token=[B,T,C_cls]"

        f = cls_token.mean(dim=1)  # [B, C_cls]
        return f


    x = image_embeddings
    if x.dim() == 2:

        if mode in ("gap", "mean"):
            return x
        elif mode == "flatten":

            B, C = x.shape
            f = x.view(B, C)
            reducer = torch.nn.Linear(f.size(1), 512, bias=False).to(f.device)
            return reducer(f)
        elif mode == "attn_pool":

            return x
        else:
            raise ValueError(f"Unknown vae_feat mode: {mode}")

    if x.dim() != 4:
        raise ValueError(f"Expect image_embeddings of shape [B,C,H,W] or [B,C], got {list(x.shape)}")


    B, C, H, W = x.shape

    if mode in ("gap", "mean"):
        return F.adaptive_avg_pool2d(x, output_size=1).view(B, C)  # [B, C]

    if mode == "flatten":
        f = x.view(B, C * H * W)  # [B, C*H*W]
        reducer = torch.nn.Linear(f.size(1), 512, bias=False).to(f.device)
        return reducer(f)

    if mode == "attn_pool":






        sal = torch.linalg.vector_norm(x, ord=2, dim=1)  # across channel C
        sal = sal.view(B, -1) / max(1e-6, float(C))
        # softmax with temperature
        attn = F.softmax(sal / max(1e-6, float(attn_temp)), dim=1)  # [B, H*W]
        attn = attn.view(B, 1, H, W)                                # [B, 1, H, W]

        weighted = (x * attn).sum(dim=[2, 3])                       # [B, C]
        return weighted

    raise ValueError(f"Unknown vae_feat mode: {mode}")


class SelfAttnPool(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = None,
        num_tokens: int = 1,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        use_abs_pos: bool = True,
    ):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.use_abs_pos = use_abs_pos


        self.pool_token = nn.Parameter(torch.zeros(1, num_tokens, in_dim))
        nn.init.trunc_normal_(self.pool_token, std=0.02)


        self.ln1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        # FFN
        self.ln2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, ffn_mult * in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * in_dim, in_dim),
            nn.Dropout(dropout),
        )


        self.proj_out = nn.Identity() if self.out_dim == in_dim else nn.Linear(in_dim, self.out_dim)

    @staticmethod
    def _build_abs_pos(h: int, w: int, c: int, device: torch.device) -> torch.Tensor:

        c_x = c // 2
        c_y = c - c_x
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=h, device=device),
            torch.linspace(-1.0, 1.0, steps=w, device=device),
            indexing="ij",
        )  # [H,W]
        pos_x = torch.stack([torch.sin(math.pi * (i + 1) * xx) for i in range(c_x)], dim=-1)  # [H,W,cx]
        pos_y = torch.stack([torch.cos(math.pi * (i + 1) * yy) for i in range(c_y)], dim=-1)  # [H,W,cy]
        pos = torch.cat([pos_x, pos_y], dim=-1).view(1, h * w, c)  # [1,L,C]
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] -> f: [B,D]
        """
        assert x.dim() == 4, f"expect 4D [B,C,H,W], got {list(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_dim, f"in_dim mismatch: {C} vs {self.in_dim}"


        X = x.flatten(2).transpose(1, 2)  # [B, L, C], L=H*W


        if self.use_abs_pos:
            pos = self._build_abs_pos(H, W, C, x.device)  # [1,L,C]
            X = X + pos


        pool = self.pool_token.expand(B, self.num_tokens, C)  # [B,Q,C]
        X = torch.cat([pool, X], dim=1)  # [B, Q+L, C]


        Y = X + self.drop1(self.attn(self.ln1(X), self.ln1(X), self.ln1(X), need_weights=False)[0])
        Z = Y + self.ffn(self.ln2(Y))


        Z_pool = Z[:, : self.num_tokens, :]  # [B,Q,C]
        if self.num_tokens == 1:
            g = Z_pool[:, 0, :]  # [B,C]
        else:
            g = Z_pool.mean(dim=1)


        f = self.proj_out(g)  # [B,D]
        return f


def build_self_attn_pool(args) -> SelfAttnPool:
    heads = getattr(args, "self_attn_heads", 4)
    num_tokens = getattr(args, "self_attn_num_tokens", 1)
    out_dim = getattr(args, "self_attn_out_dim", getattr(args, "vae_in_dim", 256))
    dropout = getattr(args, "self_attn_dropout", 0.1)
    ffn_mult = getattr(args, "self_attn_ffn_mult", 4)
    pool = SelfAttnPool(
        in_dim=256, out_dim=out_dim, num_tokens=num_tokens,
        num_heads=heads, ffn_mult=ffn_mult, dropout=dropout, use_abs_pos=True
    )
    return pool




class LearnableAttnPool(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = None, heads: int = 1, temp: float = 1.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim or in_dim)
        self.heads = int(heads)
        self.register_buffer("temp", torch.tensor(float(max(1e-6, temp))), persistent=True)


        self.q = nn.Parameter(torch.randn(self.heads, self.in_dim) * (self.in_dim ** -0.5))

        self.wk = nn.Linear(self.in_dim, self.in_dim, bias=False)
        self.wv = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expect [B,C,H,W], got {list(x.shape)}"
        B, C, H, W = x.shape
        assert C == self.in_dim, f"in_dim mismatch: got {C} vs {self.in_dim}"
        N = H * W


        x_seq = x.view(B, C, N).permute(0, 2, 1)

        K = self.wk(x_seq)                     # [B, N, C]
        V = self.wv(x_seq)                     # [B, N, out_dim]


        q = self.q / (self.in_dim ** 0.5)


        scores = torch.einsum("hc,bnc->bhn", q, K) / self.temp
        attn = torch.softmax(scores, dim=-1)   # [B, heads, N]


        pooled = torch.einsum("bhn,bnd->bhd", attn, V)


        return pooled.mean(dim=1)

def build_learnable_attn_pool(args) -> LearnableAttnPool:
    heads   = int(getattr(args, "learn_heads", 1))
    out_dim = int(getattr(args, "learn_out_dim", getattr(args, "vae_in_dim", 256)))
    temp    = float(getattr(args, "learn_temp", 1.0))
    return LearnableAttnPool(in_dim=256, out_dim=out_dim, heads=heads, temp=temp)
