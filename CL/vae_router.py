# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Tuple, List, Union
from tqdm import tqdm


try:

    from .feature_pool import SelfAttnPool
except Exception:

    from feature_pool import SelfAttnPool

class TaskVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim


        hidden = max(128, in_dim // 2)
        self.enc_fc1 = nn.Linear(in_dim, hidden)
        self.enc_fc2_mu = nn.Linear(hidden, latent_dim)
        self.enc_fc2_logvar = nn.Linear(hidden, latent_dim)


        self.dec_fc1 = nn.Linear(latent_dim, hidden)
        self.dec_fc2 = nn.Linear(hidden, in_dim)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.enc_fc1(f))
        mu = self.enc_fc2_mu(h)
        logvar = self.enc_fc2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.dec_fc1(z))
        recon = self.dec_fc2(h)
        return recon

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(f)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class TaskVAEWithPool(nn.Module):
    def __init__(self, pool: SelfAttnPool, vae: TaskVAE):
        super().__init__()
        self.pool = pool
        self.vae = vae

    def forward_from_feature_map(self, x: torch.Tensor):
        assert x.dim() == 4, f"expect [B,C,H,W], got {list(x.shape)}"
        f = self.pool(x)                      # [B,D]
        recon, mu, logvar = self.vae(f)
        return recon, mu, logvar, f

    def forward_from_vector(self, f: torch.Tensor):
        return self.vae(f)

    def forward(self, f: torch.Tensor):

        return self.forward_from_vector(f)


class TaskVAEWithLearnablePool(TaskVAEWithPool):
    pass


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # [B, D] -> [B]
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def elbo_loss(recon: torch.Tensor, f: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> torch.Tensor:

    recon_mse = torch.mean((recon - f) ** 2, dim=1)   # [B]
    kl = _kl_divergence(mu, logvar)                   # [B]
    elbo = recon_mse + beta * kl
    return elbo  # [B]


def _default_optimizer(vae: nn.Module, lr: float):
    return torch.optim.Adam(vae.parameters(), lr=lr)


def _epoch_iterator(feature_iter_fn):
    """Return a fresh epoch iterator for a re-iterable source or an iterator factory."""
    return feature_iter_fn() if callable(feature_iter_fn) else feature_iter_fn


def _unwrap_feature_batch(batch):
    """Accept a Tensor batch or the one-element tuple yielded by TensorDataset."""
    if isinstance(batch, (tuple, list)):
        if len(batch) != 1:
            raise ValueError(
                "VAE feature batches must be a Tensor or a one-element tuple/list."
            )
        batch = batch[0]
    if not torch.is_tensor(batch):
        raise TypeError(f"VAE feature batch must be a Tensor, got {type(batch)!r}")
    return batch


def train_task_vae(
    args,
    vae: TaskVAE,
    feature_iter_fn,
    epochs: int,
    lr: float,
    device: str,
    logger=None,
    log_every: int = 50
) -> None:
    vae.train().to(device)
    opt = _default_optimizer(vae, lr)
    beta = float(getattr(args, "vae_beta", 1.0))

    for ep in range(epochs):
        total_loss, n_sample = 0.0, 0
        running = 0.0
        count = 0

        epoch_iter = _epoch_iterator(feature_iter_fn)


        try:
            total_steps = len(epoch_iter)
        except Exception:
            total_steps = None

        pbar = tqdm(epoch_iter, total=total_steps, desc=f"[VAE][Train] epoch {ep+1}/{epochs}")

        for f in pbar:
            f = _unwrap_feature_batch(f)
            f = f.to(device, non_blocking=True)
            recon, mu, logvar = vae(f)
            elbo = elbo_loss(recon, f, mu, logvar, beta=beta)  # [B]
            loss = elbo.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = f.size(0)
            total_loss += float(loss.detach().item()) * bs
            n_sample += bs


            running = 0.9 * running + 0.1 * float(loss.detach().item())
            count += 1
            if (count % log_every) == 0:
                pbar.set_postfix({"elbo": f"{running:.4f}"})
                if logger is not None:
                    logger.info(f"[VAE][Train][ep {ep+1}] step={count} elbo(avg)={running:.6f}")

        if n_sample == 0:
            raise RuntimeError(
                f"[VAE][Train] epoch {ep+1}/{epochs} produced no samples. "
                "Check that feature_iter_fn is re-iterable or returns a fresh iterator per epoch."
            )

        avg = total_loss / n_sample
        if logger is not None:
            logger.info(f"[VAE][Train] epoch={ep+1}/{epochs} avg_elbo={avg:.6f}")
        else:
            print(f"[VAE][Train] epoch={ep+1}/{epochs} avg_elbo={avg:.6f}")

def train_task_vae_with_pool(
    args,
    vae: nn.Module,
    feature_iter_fn,
    epochs: int,
    lr: float,
    device: str = "cuda",
    beta: float = 1.0,
    logger=None,
    log_every: int = 50,
):
    vae = vae.to(device)
    opt = _default_optimizer(vae, lr=lr)

    for ep in range(epochs):
        total_loss, n_sample = 0.0, 0
        running, count = 0.0, 0

        epoch_iter = _epoch_iterator(feature_iter_fn)

        try:
            total_steps = len(epoch_iter)
        except Exception:
            total_steps = None

        pbar = tqdm(epoch_iter, total=total_steps, desc=f"[VAE-e2e][Train] epoch {ep+1}/{epochs}")

        for f in pbar:

            f = _unwrap_feature_batch(f)
            f = f.to(device, non_blocking=True)
            if f.dim() == 4 and hasattr(vae, "forward_from_feature_map"):
                recon, mu, logvar, f_vec = vae.forward_from_feature_map(f)   # f_vec: [B,D]
                elbo = elbo_loss(recon, f_vec, mu, logvar, beta=beta)        # [B]
            else:
                recon, mu, logvar = vae(f)
                elbo = elbo_loss(recon, f,    mu, logvar, beta=beta)        # [B]

            loss = elbo.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = f.size(0)
            total_loss += float(loss.detach().item()) * bs
            n_sample += bs

            running = 0.9 * running + 0.1 * float(loss.detach().item())
            count += 1
            if (count % log_every) == 0:
                pbar.set_postfix({"elbo": f"{running:.4f}"})
                if logger is not None:
                    logger.info(f"[VAE-e2e][Train][ep {ep+1}] step={count} elbo(avg)={running:.6f}")

        if n_sample == 0:
            raise RuntimeError(
                f"[VAE-e2e][Train] epoch {ep+1}/{epochs} produced no samples. "
                "Check that feature_iter_fn is re-iterable or returns a fresh iterator per epoch."
            )

        avg = total_loss / n_sample
        if logger is not None:
            logger.info(f"[VAE-e2e][Train] epoch={ep+1}/{epochs} avg_elbo={avg:.6f}")
        else:
            print(f"[VAE-e2e][Train] epoch={ep+1}/{epochs} avg_elbo={avg:.6f}")

@torch.no_grad()
def score_elbo_all_tasks(
    f: torch.Tensor,
    vae_dict: Dict[str, TaskVAE],
    beta: float,
    device: str
) -> Tuple[torch.Tensor, List[str]]:
    assert len(vae_dict) > 0, "vae_dict is empty"
    names = sorted(list(vae_dict.keys()))
    f = f.to(device, non_blocking=True)
    scores = []
    for name in names:
        vae = vae_dict[name].to(device).eval()
        recon, mu, logvar = vae(f)
        elbo = elbo_loss(recon, f, mu, logvar, beta=beta)  # [B]
        scores.append(elbo.unsqueeze(1))
    scores = torch.cat(scores, dim=1)  # [B, T]
    return scores, names


@torch.no_grad()
def route_with_vae(
    f: torch.Tensor,
    vae_dict: Dict[str, TaskVAE],
    beta: float,
    tau: Union[float, Dict[str, float]],
    device: str
):
    scores, names = score_elbo_all_tasks(f, vae_dict, beta=beta, device=device)  # [B, T]
    min_vals, min_idx = torch.min(scores, dim=1)  # [B], [B]
    if isinstance(tau, dict):
        tau_by_task = torch.tensor(
            [float(tau.get(name, 1.0)) for name in names],
            dtype=scores.dtype,
            device=scores.device,
        )
        is_unknown = min_vals > tau_by_task[min_idx]
    else:
        is_unknown = min_vals > float(tau)
    return min_idx, is_unknown, names, scores

@torch.no_grad()
def suggest_tau_from_valset(
    val_loader,
    vae_dict,
    device="cuda",
    beta: float = 1.0,
    k: float = 2.0,
    logger=None,
    sam=None,
    args=None,
):
    import numpy as np
    from tqdm import tqdm

    all_min_elbos = []

    for bi in tqdm(val_loader, desc="[TauSuggest] Collect ELBO"):

        if isinstance(bi, dict):
            if "vae_feat" in bi:
                f = bi["vae_feat"].to(device, non_blocking=True)
            elif "image" in bi:
                assert (sam is not None) and (args is not None), \
                    "val_loader provides raw images, so sam and args are required to extract features"
                imgs = bi["image"].to(device, non_blocking=True)
                if args.vae_feat in ("cls", "attn_pool"):
                    token_len = args.cls_token_len
                    embed_dim = args.vae_in_dim
                    base_cls = getattr(sam.image_encoder, "cls_token", None)
                    if base_cls is None:
                        base_cls = torch.randn(1, token_len, embed_dim, device=device)
                    out = sam.image_encoder(imgs, cls_token=base_cls)
                    if isinstance(out, tuple) and len(out) == 3:
                        conv_feat, _, cls_tok = out
                    else:
                        conv_feat, cls_tok = out, None
                    f = extract_feature_for_vae(
                        conv_feat, mode=args.vae_feat, cls_token=cls_tok
                    ).to(device, non_blocking=True)
                else:
                    emb = sam.image_encoder(imgs)
                    f = extract_feature_for_vae(emb, mode=args.vae_feat).to(device, non_blocking=True)
            else:
                raise RuntimeError("dict batches must contain 'vae_feat' or 'image'")
        elif isinstance(bi, (list, tuple)):

            assert len(bi) >= 1 and torch.is_tensor(bi[0]), "unexpected batch format"
            f = bi[0].to(device, non_blocking=True)
        elif torch.is_tensor(bi):
            f = bi.to(device, non_blocking=True)
        else:
            raise RuntimeError(f"Unsupported batch type: {type(bi)}")


        scores, _ = score_elbo_all_tasks(f, vae_dict, beta=beta, device=device)
        min_vals, _ = torch.min(scores, dim=1)
        all_min_elbos.extend(min_vals.detach().cpu().numpy())

    all_min_elbos = np.array(all_min_elbos, dtype=np.float64)
    mu, sigma = float(all_min_elbos.mean()), float(all_min_elbos.std())
    p95, vmax = float(np.percentile(all_min_elbos, 95)), float(all_min_elbos.max())
    tau = mu + k * sigma

    msg = (f"[TauSuggest] mean={mu:.4f}, std={sigma:.4f}, p95={p95:.4f}, "
           f"max={vmax:.4f}, suggested tau={tau:.4f} (mu+{k}σ)")
    (logger.info if logger else print)(msg)
    return tau, {"mean": mu, "std": sigma, "p95": p95, "max": vmax, "tau_suggested": tau}

# @torch.no_grad()
# def suggest_tau_from_valset(
#     val_loader,
#     vae_dict,
#     device="cuda",
#     beta: float = 1.0,
#     k: float = 2.0,
#     logger=None,
#     sam=None,
#     args=None,
# ):
#     """


#     """
#     import numpy as np
#     from tqdm import tqdm

#     all_min_elbos = []

#     for bi in tqdm(val_loader, desc="[TauSuggest] Collect ELBO"):

#         if isinstance(bi, dict):
#             if "vae_feat" in bi:
#                 f = bi["vae_feat"].to(device, non_blocking=True)
#             elif "image" in bi:
#                 assert (sam is not None) and (args is not None), \

#                 imgs = bi["image"].to(device, non_blocking=True)
#                 if args.vae_feat in ("cls", "attn_pool"):
#                     token_len = args.cls_token_len
#                     embed_dim = args.vae_in_dim
#                     base_cls = getattr(sam.image_encoder, "cls_token", None)
#                     if base_cls is None:
#                         base_cls = torch.randn(1, token_len, embed_dim, device=device)
#                     out = sam.image_encoder(imgs, cls_token=base_cls)
#                     if isinstance(out, tuple) and len(out) == 3:
#                         conv_feat, _, cls_tok = out
#                     else:
#                         conv_feat, cls_tok = out, None
#                     f = extract_feature_for_vae(
#                         conv_feat, mode=args.vae_feat, cls_token=cls_tok
#                     ).to(device, non_blocking=True)
#                 else:
#                     emb = sam.image_encoder(imgs)
#                     f = extract_feature_for_vae(emb, mode=args.vae_feat).to(device, non_blocking=True)
#             else:

#         elif isinstance(bi, (list, tuple)):
#             assert len(bi) >= 1 and torch.is_tensor(bi[0]), "unexpected batch format"
#             f = bi[0].to(device, non_blocking=True)
#         elif torch.is_tensor(bi):
#             f = bi.to(device, non_blocking=True)
#         else:
#             raise RuntimeError(f"Unsupported batch type: {type(bi)}")


#         scores, _ = score_elbo_all_tasks(f, vae_dict, beta=beta, device=device)
#         min_vals, _ = torch.min(scores, dim=1)
#         all_min_elbos.extend(min_vals.detach().cpu().numpy())

#     all_min_elbos = np.array(all_min_elbos, dtype=np.float64)
#     mu    = float(all_min_elbos.mean())
#     sigma = float(all_min_elbos.std())
#     p95   = float(np.percentile(all_min_elbos, 95))
#     p99   = float(np.percentile(all_min_elbos, 99))
#     p995  = float(np.percentile(all_min_elbos, 99.5))
#     p998  = float(np.percentile(all_min_elbos, 99.8))
#     p999  = float(np.percentile(all_min_elbos, 99.9))
#     vmax  = float(all_min_elbos.max())
#     tau   = mu + k * sigma


#     msg = (
#         f"[TauSuggest] mean={mu:.6f}, std={sigma:.6f}, "
#         f"p95={p95:.6f}, p99={p99:.6f}, p99.5={p995:.6f}, p99.8={p998:.6f}, p99.9={p999:.6f}, "
#         f"max={vmax:.6f}, suggested tau={tau:.6f} (mu+{k}σ)"
#     )
#     (logger.info if logger else print)(msg)

#     stats = {
#         "mean": mu,
#         "std": sigma,
#         "p95": p95,
#         "p99": p99,
#         "p99_5": p995,
#         "p99_8": p998,
#         "p99_9": p999,
#         "max": vmax,
#         "tau_suggested": tau,
#     }
#     return tau, stats
