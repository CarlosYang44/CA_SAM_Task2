#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, csv, copy, datetime, json
import argparse
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F

from segment_anything import sam_model_registry
from segment_anything.modeling.align_transformer_modeling import (
    AlignMLP, AlignCNN, AlignTransformer, AlignTransformerPlus
)
from utils import get_logger, FocalDiceloss_IoULoss
from data_loader import get_loader

# ---- VAE / Router helpers ----
from CL.feature_pool import extract_feature_for_vae
from CL.vae_router import TaskVAE, train_task_vae, elbo_loss
from CL.eval_moda_vae import evaluate_moda_vae

try:
    from CL.adapter_hub import load_adapters_for_seen_tasks, build_zero_adapter
except Exception:
    load_adapters_for_seen_tasks = None
    build_zero_adapter = lambda mode="identity": torch.nn.Identity()
try:
    from CL.router_assets import load_vaes_for_seen_tasks, save_current_task_assets
except Exception:
    load_vaes_for_seen_tasks, save_current_task_assets = None, None

DEFAULT_TASK_TAUS = {
    "ACDC": 0.0813,
    "EBHI-SEG": 0.0690,
    "56Nx": 0.2258,
    "DN": 0.1646,
    "Polyp": 0.1494,
    "MSD_Prostate": 0.1915,
    "MSD_Spleen": 0.1285,
    "promise12": 0.1483,
    "STS-2D": 0.0662,
}


# ---------------- argparse ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=str, default="workdir")
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--all_datasets", type=str, required=True,
                    help="Full task chain, comma separated. CSV columns follow this order.")
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--sam_checkpoint", type=str, default="./pretrain_model/sam_vit_b_01ec64.pth")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--model_type", type=str, default="vit_b")
    ap.add_argument("--method", type=str, default="cnn", choices=["cnn","mlp","transformer","transplus"])
    ap.add_argument("--num_cnn", type=int, default=3)
    ap.add_argument("--multimask", default=False, action='store_true')

    # loader
    ap.add_argument("--train_batch_size", type=int, default=6)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--mask_num", type=int, default=5)
    ap.add_argument("--metrics", nargs="+", default=["iou","dice","biou"])
    ap.add_argument("--dataset_scale", type=float, default=1.0, help="Dataset sampling ratio. The default 1.0 uses the full set.")
    ap.add_argument("--cls_token_len", type=int, default=1, help="cls_token length")


    ap.add_argument("--adapters_ckpt_dir", type=str, required=True,
                    help="Root directory for previous alignment weights (<root>/<task>/*.pth or pth files containing the dataset name)")
    ap.add_argument("--router_ckpt_dir", type=str, required=True,
                    help="VAE root directory. Files are stored as <root>/<task>/vae.pth.")


    ap.add_argument("--vae_feat", type=str, default="gap", choices=["gap","mean","flatten","cls","attn_pool"])
    ap.add_argument("--vae_in_dim", type=int, default=256)
    ap.add_argument("--vae_latent_dim", type=int, default=64)
    ap.add_argument("--vae_epochs", type=int, default=10)
    ap.add_argument("--vae_lr", type=float, default=5e-4)
    ap.add_argument("--vae_beta", type=float, default=16.5, help="VAE KL loss weight beta")
    ap.add_argument("--router_threshold", type=float, default=1.0,
                    help="ELBO threshold tau. min-ELBO > tau is treated as an unknown task.")
    ap.add_argument("--zero_adapter_mode", type=str, default="identity", choices=["zeros","identity"])
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--fold_seed", type=int, default=2025)
    ap.add_argument("--tau_k_std", type=float, default=2.0)


    ap.add_argument("--cl_matrix_csv", type=str, default=None, help="Wide-table CSV for IoU. Created when missing.")
    ap.add_argument("--cl_matrix_biou_csv", type=str, default=None, help="Wide-table CSV for BIoU. Created when missing.")
    ap.add_argument("--skip_train_vae", action="store_true",
                    help="Load the current task VAE checkpoint instead of retraining it")
    return ap.parse_args()


# ---------------- tiny helpers ----------------
def build_align_module(args):
    if args.method == "cnn":
        return AlignCNN(num_layers=args.num_cnn, dim=256)
    if args.method == "mlp":
        return AlignMLP(num_layers=1, dim_in=256, dim_ff=256, activation=F.gelu)
    if args.method == "transformer":
        return AlignTransformer(num_blocks=5, input_dim=256, embed_dim=512, num_heads=8,
                                num_queries=16, pixel_pe_scale=32, pixel_pe_temperature=128)
    if args.method == "transplus":
        return AlignTransformerPlus()
    raise ValueError(args.method)


def find_adapter_ckpt(root: str, ds: str) -> Optional[str]:



    pattern1 = os.path.join(root, f"*{ds}*", "align_*.pth")
    files = glob.glob(pattern1)
    if files:

        for f in files:
            bn = os.path.basename(f).lower()
            if "align" in bn or "adapter" in bn:
                return f
        return files[0]


    pattern2 = os.path.join(root, ds, "*.pth")
    files = glob.glob(pattern2)
    if files:
        for f in files:
            bn = os.path.basename(f).lower()
            if "align" in bn or "adapter" in bn:
                return f
        return files[0]


    pattern3 = os.path.join(root, f"**/*{ds}*.pth")
    files = glob.glob(pattern3, recursive=True)
    if files:
        for f in files:
            bn = os.path.basename(f).lower()
            if "align" in bn or "adapter" in bn:
                return f
        return files[0]


    f = os.path.join(root, f"adapter_{ds}.pth")
    return f if os.path.isfile(f) else None


def append_row(csv_path: str, row_idx: int, ds_list: List[str], values: Dict[str, Optional[float]]):
    header = ["Task"] + ds_list + ["Avg"]
    need_header = not os.path.isfile(csv_path)
    row_vals = [values.get(ds, None) for ds in ds_list]
    valid = [v for v in row_vals if v is not None]
    avg = float(np.mean(valid)) if valid else None
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow([row_idx] + [("" if v is None else float(v)) for v in row_vals] + [("" if avg is None else float(avg))])


def _extract_vae_feature(args, sam, imgs):
    if args.vae_feat == "cls":
        token_len = args.cls_token_len
        embed_dim = args.vae_in_dim
        base_cls = getattr(sam.image_encoder, "cls_token", None)
        if base_cls is None:
            base_cls = torch.randn(1, token_len, embed_dim, device=args.device)

        out = sam.image_encoder(imgs, cls_token=base_cls)
        if isinstance(out, tuple) and len(out) == 3:
            conv_feat, _, cls_tok = out
        elif isinstance(out, tuple) and len(out) == 2:
            conv_feat, _ = out
            cls_tok = None
        else:
            conv_feat = out
            cls_tok = None
        assert cls_tok is not None, "use_moda_encoder=True with vae_feat=cls requires the encoder to return cls_token"
        return extract_feature_for_vae(conv_feat, mode="cls", cls_token=cls_tok)
    if args.vae_feat == "attn_pool":
        emb = sam.image_encoder(imgs)
        return extract_feature_for_vae(emb, mode="attn_pool", attn_temp=1.0)
    emb = sam.image_encoder(imgs)
    return extract_feature_for_vae(emb, mode=args.vae_feat)


@torch.no_grad()
def _collect_vae_features(args, sam, loader):
    feats = []
    pbar = tqdm(loader, desc=f"[Tau][Feat] {args.dataset_name}", ncols=120)
    for bi in pbar:
        imgs = bi["image"].to(args.device, non_blocking=True)
        feat = _extract_vae_feature(args, sam, imgs)
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def _train_temp_vae(args, feats_train):
    vae = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(feats_train)
    dl = DataLoader(ds, batch_size=max(1, int(args.train_batch_size)), shuffle=True, drop_last=False)

    train_task_vae(
        args=args,
        vae=vae,
        feature_iter_fn=dl,
        epochs=int(args.vae_epochs),
        lr=float(args.vae_lr),
        device=args.device,
        logger=None,
    )
    return vae.eval()


def _estimate_tau_kfold(args, sam, train_loader, logger=None):
    feats_all = _collect_vae_features(args, sam, train_loader)
    n = int(feats_all.size(0))
    if n == 0:
        tau = float(getattr(args, "router_threshold", 1.0))
        return {
            "tau_suggested": tau,
            "tau_p95": tau,
            "tau_p97": tau,
            "tau_p99": tau,
            "tau_mean": tau,
            "tau_std": 0.0,
            "tau_max": tau,
            "tau_mu_k_sigma": tau,
            "k_folds": 0,
            "n_samples": 0,
            "fold_seed": int(args.fold_seed),
            "tau_k_std": float(args.tau_k_std),
        }

    k = max(1, int(args.k_folds))
    k = min(k, n)
    rng = np.random.RandomState(int(args.fold_seed))
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)

    s_vals = []
    for j in range(k):
        val_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j]) if k > 1 else folds[j]
        f_train = feats_all[torch.as_tensor(train_idx, dtype=torch.long)]
        f_val = feats_all[torch.as_tensor(val_idx, dtype=torch.long)]
        temp_vae = _train_temp_vae(args, f_train)
        with torch.no_grad():
            f_val_dev = f_val.to(args.device)
            recon, mu, logvar = temp_vae(f_val_dev)
            elbos = elbo_loss(recon, f_val_dev, mu, logvar, beta=float(args.vae_beta))
        s_vals.extend(elbos.detach().cpu().tolist())
        if logger is not None:
            logger.info(
                f"[Tau-KFold][{j+1}/{k}] val_size={len(val_idx)} "
                f"mean={float(elbos.mean().item()):.6f} std={float(elbos.std().item()):.6f} "
                f"min={float(elbos.min().item()):.6f} max={float(elbos.max().item()):.6f}"
            )

    s = np.asarray(s_vals, dtype=np.float64)
    mu = float(s.mean())
    sigma = float(s.std())
    p95 = float(np.percentile(s, 95))
    p97 = float(np.percentile(s, 97))
    p99 = float(np.percentile(s, 99))
    vmax = float(s.max())
    tau_mu_k_sigma = mu + float(args.tau_k_std) * sigma
    stats = {
        "tau_suggested": p97,
        "tau_p95": p95,
        "tau_p97": p97,
        "tau_p99": p99,
        "tau_mean": mu,
        "tau_std": sigma,
        "tau_max": vmax,
        "tau_mu_k_sigma": tau_mu_k_sigma,
        "k_folds": k,
        "n_samples": n,
        "fold_seed": int(args.fold_seed),
        "tau_k_std": float(args.tau_k_std),
    }
    if logger is not None:
        logger.info(
            "[TauStats] mean=%.6f, std=%.6f, p95=%.6f, p97=%.6f, p99=%.6f, max=%.6f, (mu+%sσ)=%.6f",
            mu, sigma, p95, p97, p99, vmax, str(args.tau_k_std), tau_mu_k_sigma
        )
        logger.info("[TauSuggest] Use p97 as suggested tau=%.6f", p97)
    print(f"[TauSuggest] Suggested τ={p97:.6f}")
    return stats


def _save_tau_stats(args, task_name: str, stats: dict):
    if not getattr(args, "router_ckpt_dir", None):
        return None
    tau_dir = os.path.join(args.router_ckpt_dir, task_name)
    os.makedirs(tau_dir, exist_ok=True)
    tau_path = os.path.join(tau_dir, "tau.json")
    payload = {"task_name": task_name, "dataset_name": task_name, **stats}
    with open(tau_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return tau_path


def _load_tau_stats(args, task_name: str):
    if not getattr(args, "router_ckpt_dir", None):
        return None
    tau_path = os.path.join(args.router_ckpt_dir, task_name, "tau.json")
    if not os.path.isfile(tau_path):
        return None
    try:
        with open(tau_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _tau_value_from_stats(stats, fallback: float) -> float:
    if not stats:
        return float(fallback)
    for key in ("tau_p97", "tau_suggested", "suggested_tau", "tau"):
        if key in stats and stats[key] is not None:
            try:
                return float(stats[key])
            except Exception:
                pass
    return float(fallback)


# ---------------- main ----------------
def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.router_ckpt_dir, exist_ok=True)

    # logger
    log_file = os.path.join(args.work_dir, f"vae_route_{args.dataset_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log")
    logger = get_logger(log_file)
    logger.info(f"== Start for dataset: {args.dataset_name} ==")


    all_ds = [s.strip() for s in args.all_datasets.split(",") if s.strip()]
    if args.dataset_name not in all_ds:
        raise ValueError("dataset_name must be included in all_datasets")
    stage_idx = all_ds.index(args.dataset_name)
    seen_full = all_ds[:stage_idx+1]   # 0..current
    args.router_thresholds = {
        task: DEFAULT_TASK_TAUS.get(task, 1.0)
        for task in all_ds
    }

    # ---- SAM & freeze ----
    sam = sam_model_registry[args.model_type](args).to(args.device)
    for p in sam.parameters(): p.requires_grad = False

    # ---- load adapter for current dataset ----
    align_module = build_align_module(args).to(args.device)
    ck = find_adapter_ckpt(args.adapters_ckpt_dir, args.dataset_name)
    adapter_loaded = ck is not None
    if ck is None:
        logger.warning(f"[Adapter] Could not find weights for {args.dataset_name}; falling back to the identity adapter")
        align_module = build_zero_adapter(mode="identity").to(args.device)
    else:
        state = torch.load(ck, map_location=args.device)

        if isinstance(state, dict) and not any(k.startswith("0.") for k in state.keys()):
            try:
                align_module.load_state_dict(state, strict=True)
            except Exception:
                cleaned = {k.replace("module.", ""): v for k, v in state.items()}
                align_module.load_state_dict(cleaned, strict=False)
        else:
            align_module.load_state_dict(state, strict=False)
        logger.info(f"[Adapter] Loaded: {ck}")
    align_module.eval()

    # ---- loaders for current dataset (for VAE features) ----
    args_train = copy.deepcopy(args); args_train.test_mode = False; args_train.batch_size = args.train_batch_size
    args_test  = copy.deepcopy(args); args_test.test_mode  = True;  args_test.batch_size  = args.eval_batch_size

    train_loader = get_loader(args_train, all_training_sets=[args.dataset_name])

    vae = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)
    vae_dir = os.path.join(args.router_ckpt_dir, args.dataset_name); os.makedirs(vae_dir, exist_ok=True)
    vae_path = os.path.join(vae_dir, "vae.pth")
    if args.skip_train_vae:
        if not os.path.isfile(vae_path):
            raise FileNotFoundError(f"--skip_train_vae requested, but VAE checkpoint was not found: {vae_path}")
        vae.load_state_dict(torch.load(vae_path, map_location=args.device))
        vae.eval()
        logger.info(f"[VAE] Loaded -> {vae_path}")
    else:
        logger.info(f"[VAE] Train {args.dataset_name}: in={args.vae_in_dim}, z={args.vae_latent_dim}, epochs={args.vae_epochs}, lr={args.vae_lr}")
        vae_feats = _collect_vae_features(args, sam, train_loader)
        from torch.utils.data import DataLoader, TensorDataset
        vae_loader = DataLoader(
            TensorDataset(vae_feats),
            batch_size=max(1, int(args.train_batch_size)),
            shuffle=True,
            drop_last=False,
        )
        train_task_vae(args=args, vae=vae, feature_iter_fn=vae_loader, epochs=int(args.vae_epochs),
                       lr=float(args.vae_lr), device=args.device, logger=logger)
        torch.save(vae.state_dict(), vae_path)
        logger.info(f"[VAE] Saved -> {vae_path}")


    task_tau_stats = None
    if not args.skip_train_vae:
        task_tau_stats = _estimate_tau_kfold(args, sam, train_loader, logger=logger)
        tau_path = _save_tau_stats(args, args.dataset_name, task_tau_stats)
        if tau_path:
            logger.info(f"[Tau] Saved -> {tau_path}")



    if save_current_task_assets is not None and adapter_loaded:
        try:
            save_current_task_assets(args, task_name=args.dataset_name, align_module=align_module, task_vae=vae)
        except Exception as e:
            logger.warning(f"save_current_task_assets failed: {e}")

    # ---- Build adapters_dict / vae_dict for seen tasks ----
    adapters_dict: Dict[str, torch.nn.Module] = {}
    vae_dict: Dict[str, torch.nn.Module] = {}


    if load_adapters_for_seen_tasks is not None:
        try:
            adapters_dict = load_adapters_for_seen_tasks(args, args.device)
        except Exception as e:
            logger.warning(f"load_adapters_for_seen_tasks failed: {e}")
    if load_vaes_for_seen_tasks is not None:
        try:
            vae_dict = load_vaes_for_seen_tasks(args, args.device)
        except Exception as e:
            logger.warning(f"load_vaes_for_seen_tasks failed: {e}")

    adapters_dict = {ds: adapters_dict[ds] for ds in seen_full if ds in adapters_dict}
    vae_dict = {ds: vae_dict[ds] for ds in seen_full if ds in vae_dict}


    for ds in seen_full:
        # adapter
        if ds not in adapters_dict or adapters_dict[ds] is None:
            ck_ds = find_adapter_ckpt(args.adapters_ckpt_dir, ds)
            if ck_ds:
                m = build_align_module(args).to(args.device)
                try:
                    st = torch.load(ck_ds, map_location=args.device)
                    try:
                        m.load_state_dict(st, strict=True)
                    except Exception:
                        st2 = {k.replace("module.", ""): v for k, v in st.items()}
                        m.load_state_dict(st2, strict=False)
                    m.eval()
                    adapters_dict[ds] = m
                    logger.info(f"[Adapter] Loaded for {ds}: {ck_ds}")
                except Exception as e:
                    logger.warning(f"[Adapter] Failed to load {ds} from {ck_ds}: {e}")
        # vae
        if ds not in vae_dict or vae_dict[ds] is None:
            vp = os.path.join(args.router_ckpt_dir, ds, "vae.pth")
            if os.path.isfile(vp):
                v = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)
                try:
                    v.load_state_dict(torch.load(vp, map_location=args.device))
                    v.eval()
                    vae_dict[ds] = v
                    logger.info(f"[VAE] Loaded for {ds}: {vp}")
                except Exception as e:
                    logger.warning(f"[VAE] Failed to load {ds} from {vp}: {e}")


    adapters_dict[args.dataset_name] = align_module.eval()
    vae_dict[args.dataset_name] = vae.eval()
    for p in adapters_dict[args.dataset_name].parameters(): p.requires_grad = False
    for p in vae_dict[args.dataset_name].parameters(): p.requires_grad = False

    missing_vaes = [ds for ds in seen_full if ds not in vae_dict]
    if missing_vaes:
        raise FileNotFoundError(
            "Missing VAE checkpoints for visible tasks: " + ", ".join(missing_vaes)
        )

    args.router_thresholds = {}
    for task in seen_full:
        stats = task_tau_stats if (task == args.dataset_name and task_tau_stats is not None) else _load_tau_stats(args, task)
        if stats is None:
            stats = _load_tau_stats(args, task)
        args.router_thresholds[task] = _tau_value_from_stats(stats, DEFAULT_TASK_TAUS.get(task, 1.0))

    # ---- Evaluate routing on each seen dataset ----
    iou_idx  = args.metrics.index("iou")  if "iou"  in args.metrics else None
    biou_idx = args.metrics.index("biou") if "biou" in args.metrics else None
    vals_iou: Dict[str, Optional[float]] = {}
    vals_biou: Dict[str, Optional[float]] = {}

    for ds in seen_full:
        args_eval = copy.deepcopy(args)
        args_eval.test_mode = True
        args_eval.batch_size = args.eval_batch_size
        args_eval.dataset_name = ds
        args.router_threshold = float(args.router_thresholds.get(ds, 1.0))

        test_loader = get_loader(args_eval, all_training_sets=[ds])
        loss_r, metrics_r = evaluate_moda_vae(
            args, sam, adapters_dict, test_loader, FocalDiceloss_IoULoss(), epoch=0,
            vae_dict=vae_dict, zero_adapter_mode=args.zero_adapter_mode
        )

        vi = metrics_r[iou_idx]  if (iou_idx  is not None) else None
        vb = metrics_r[biou_idx] if (biou_idx is not None) else None
        vals_iou[ds]  = vi
        vals_biou[ds] = vb
        print(f"[Eval-VAE-Route][{ds}] loss={loss_r:.4f} iou={vi} biou={vb}")

    # ---- Write CSV rows (IoU & BIoU) ----
    out_dir = os.path.join(args.work_dir, "cl_metrics")
    os.makedirs(out_dir, exist_ok=True)
    iou_csv  = args.cl_matrix_csv      or os.path.join(out_dir, "vae_router_iou.csv")
    biou_csv = args.cl_matrix_biou_csv or os.path.join(out_dir, "vae_router_biou.csv")


    def pad(vals_dict):
        return {ds: (vals_dict.get(ds, None) if ds in seen_full else None) for ds in all_ds}

    append_row(iou_csv,  stage_idx, all_ds, pad(vals_iou))
    append_row(biou_csv, stage_idx, all_ds, pad(vals_biou))
    print(f"[CSV] IoU  row -> {iou_csv}")
    print(f"[CSV] BIoU row -> {biou_csv}")
    print("== DONE ==")


if __name__ == "__main__":
    args = parse_args()
    main(args)
