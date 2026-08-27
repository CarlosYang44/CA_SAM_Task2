# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from typing import Dict
from segment_anything.modeling.align_transformer_modeling import (
    AlignMLP, AlignCNN, AlignTransformer, AlignTransformerPlus
)

def build_align_module_by_args(args) -> nn.Module:
    if args.method == "cnn":
        return AlignCNN(num_layers=args.num_cnn, dim=256)
    elif args.method == "mlp":
        return AlignMLP(num_layers=1, dim_in=256, dim_ff=256)
    elif args.method == "transformer":
        return AlignTransformer(num_blocks=5, input_dim=256, embed_dim=512, num_heads=8,
                                num_queries=16, pixel_pe_scale=32, pixel_pe_temperature=128)
    elif args.method == "transplus":
        return AlignTransformerPlus()
    else:
        raise ValueError(f"Unknown method: {args.method}")

class ZeroAdapter(nn.Module):
    def __init__(self, mode: str = "zeros"):
        super().__init__()
        assert mode in ("zeros", "identity")
        self.mode = mode

    def forward(self, image_embeddings_ori: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return image_embeddings_ori
        # zeros
        return torch.zeros_like(image_embeddings_ori, device=image_embeddings_ori.device)

def build_zero_adapter(mode: str = "zeros") -> nn.Module:
    return ZeroAdapter(mode=mode).eval()

def load_adapters_for_seen_tasks(args, device: str) -> Dict[str, nn.Module]:
    adapters: Dict[str, nn.Module] = {}
    root = getattr(args, "adapters_ckpt_dir", None)
    if not root or (not os.path.isdir(root)):
        return adapters

    for task_name in os.listdir(root):
        sub = os.path.join(root, task_name)
        if not os.path.isdir(sub):
            continue

        ckpt = os.path.join(sub, f"align_{args.method}_{args.num_cnn}.pth")
        if not os.path.isfile(ckpt):
            continue
        m = build_align_module_by_args(args).to(device)
        sd = torch.load(ckpt, map_location=device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        adapters[task_name] = m
    return adapters
