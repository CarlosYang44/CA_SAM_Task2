# -*- coding: utf-8 -*-
import os
import torch
from typing import Dict
from .vae_router import TaskVAE

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_vaes_for_seen_tasks(args, device: str) -> Dict[str, TaskVAE]:
    vaes: Dict[str, TaskVAE] = {}
    root = getattr(args, "router_ckpt_dir", None)
    if not root or (not os.path.isdir(root)):
        return vaes

    for task_name in os.listdir(root):
        sub = os.path.join(root, task_name)
        if not os.path.isdir(sub):
            continue
        ckpt = os.path.join(sub, "vae.pth")
        if not os.path.isfile(ckpt):
            continue
        vae = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(device)
        sd = torch.load(ckpt, map_location=device)
        vae.load_state_dict(sd, strict=True)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        vaes[task_name] = vae
    return vaes

def save_current_task_assets(args, task_name: str, align_module, task_vae) -> None:
    # Adapter
    aroot = getattr(args, "adapters_ckpt_dir", None)
    if aroot:
        adir = os.path.join(aroot, task_name)
        ensure_dir(adir)
        apath = os.path.join(adir, f"align_{args.method}_{args.num_cnn}.pth")
        torch.save(align_module.state_dict(), apath)

    # VAE
    vroot = getattr(args, "router_ckpt_dir", None)
    if vroot and (task_vae is not None):
        vdir = os.path.join(vroot, task_name)
        ensure_dir(vdir)
        vpath = os.path.join(vdir, "vae.pth")
        torch.save(task_vae.state_dict(), vpath)
