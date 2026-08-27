# -*- coding: utf-8 -*-

import os
import copy
import time
import json
import torch
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from segment_anything import sam_model_registry
from segment_anything.modeling.align_transformer_modeling import AlignMLP, AlignCNN, AlignTransformer, AlignTransformerPlus

from utils import FocalDiceloss_IoULoss, get_logger, DistillLoss
from metrics import SegMetrics
from data_loader import get_loader


from CL.feature_pool import extract_feature_for_vae
from CL.vae_router import TaskVAE, train_task_vae, elbo_loss
from CL.router_assets import load_vaes_for_seen_tasks, save_current_task_assets
from CL.adapter_hub import load_adapters_for_seen_tasks
from CL.eval_moda_vae import evaluate_moda_vae
try:
    from CL.adapter_hub import load_adapters_for_seen_tasks
except Exception:
    load_adapters_for_seen_tasks = None
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir")
    parser.add_argument("--run_name", type=str, default="ACDC")
    parser.add_argument("--epochs", type=int, default=24)


    parser.add_argument("--batch_size", type=int, default=None, help="Generic batch size for compatibility")
    parser.add_argument("--train_batch_size", type=int, default=6, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--mask_num", type=int, default=5)
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice', 'biou'])
    parser.add_argument("--data_dir", type=str, default="./data", help="train data path")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default="./pretrain_model/sam_vit_b_01ec64.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--align_checkpoint", type=str, default=None)
    parser.add_argument("--num_cnn", type=int, default=3)
    parser.add_argument("--method", type=str, default="cnn")
    parser.add_argument("--multimask", default=False, action='store_true')
    parser.add_argument("--num_datasets", type=int, default=1)
    parser.add_argument("--dataset_scale", type=float, default=1.0)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--distill", default=False, action='store_true')
    parser.add_argument("--save_pred", default=False, action='store_true')
    parser.add_argument("--test_mode", default=False, action='store_true')
    parser.add_argument("--save_root", type=str, default=None, help="If provided, logs default to save_root/logs and checkpoints to save_root/checkpoints/<run_name>/")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory. Overrides save_root/logs when provided.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory. Overrides save_root/checkpoints/<run_name>/ when provided.")


    parser.add_argument("--router_type", type=str, default="none", choices=["none", "vae"], help="Router type")
    parser.add_argument("--router_threshold", type=float, default=1.0, help="VAE ELBO threshold tau. Scores above tau are treated as unknown and use the zero/identity adapter.")
    parser.add_argument("--router_ckpt_dir", type=str, default=None, help="Root directory for previous-task VAEs (<root>/<task>/vae.pth)")
    parser.add_argument("--adapters_ckpt_dir", type=str, default=None, help="Root directory for previous-task adapters (<root>/<task>/align_*.pth)")
    parser.add_argument("--zero_adapter_mode", type=str, default="identity", choices=["zeros", "identity"], help="Fallback adapter mode for unknown tasks")
    parser.add_argument("--all_datasets", type=str, default=None, help="Full task chain, comma separated, used to filter seen_full")


    parser.add_argument("--vae_feat", type=str, default="attn_pool", choices=["gap", "mean", "flatten","attn_pool"], help="Image feature extraction mode for VAE routing")
    parser.add_argument("--vae_in_dim", type=int, default=256, help="VAE input dimension. Must match the feature dimension.")
    parser.add_argument("--vae_latent_dim", type=int, default=64, help="VAE latent dimension")
    parser.add_argument("--vae_beta", type=float, default=16.5, help="VAE KL loss weight beta")
    parser.add_argument("--vae_epochs", type=int, default=10, help="Number of VAE training epochs per task")
    parser.add_argument("--vae_lr", type=float, default=5e-4, help="VAE training learning rate")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of K-fold splits for tau calibration")
    parser.add_argument("--fold_seed", type=int, default=2025, help="Random seed for tau K-fold calibration")
    parser.add_argument("--tau_k_std", type=float, default=2.0, help="k in the tau statistic mu + k*sigma")

    return parser.parse_args()


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is None:
            continue
        if key in ('image', 'label', 'ori_label'):
            device_input[key] = value.to(device)
        elif key in ('prompt', 'gt_prompt'):
            device_input[key] = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in value.items()}
        else:
            device_input[key] = value
    return device_input


def build_align_module(args):
    if args.method == "cnn":
        return AlignCNN(num_layers=args.num_cnn, dim=256)
    elif args.method == "mlp":
        return AlignMLP(num_layers=1, dim_in=256, dim_ff=256, activation=F.gelu)
    elif args.method == "transformer":
        return AlignTransformer(num_blocks=5, input_dim=256, embed_dim=512, num_heads=8,
                                num_queries=16, pixel_pe_scale=32, pixel_pe_temperature=128)
    elif args.method == "transplus":
        return AlignTransformerPlus()
    else:
        raise ValueError(f"Unknown method: {args.method}")


def _repeat_to_match_batch(x, target_B):
    if x is None:
        return None
    if x.size(0) == target_B:
        return x
    assert target_B % x.size(0) == 0, f"Can't tile from {x.size(0)} to {target_B}"
    times = target_B // x.size(0)
    return x.repeat_interleave(times, dim=0)


def _extract_vae_feature_for_train(args, sam, imgs):
    emb = sam.image_encoder(imgs)
    return extract_feature_for_vae(emb, mode=args.vae_feat, attn_temp=1.0)


@torch.no_grad()
def _collect_vae_features(args, sam, loader):
    feats = []
    pbar = tqdm(loader, desc=f"[Tau][Feat] {args.dataset_name}")
    for bi in pbar:
        imgs = bi["image"].to(args.device, non_blocking=True)
        feat = _extract_vae_feature_for_train(args, sam, imgs)
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def _train_temp_vae(args, feats_train):
    temp_vae = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)
    ds = TensorDataset(feats_train)
    dl = DataLoader(ds, batch_size=max(1, int(args.train_batch_size)), shuffle=True, drop_last=False)

    train_task_vae(
        args=args,
        vae=temp_vae,
        feature_iter_fn=dl,
        epochs=int(args.vae_epochs),
        lr=float(args.vae_lr),
        device=args.device,
        logger=None,
    )
    return temp_vae.eval()


def _estimate_tau_kfold(args, sam, train_loader, logger=None):
    feats_all = _collect_vae_features(args, sam, train_loader)
    n = int(feats_all.size(0))
    if n == 0:
        tau = 1.0
        stats = {
            "tau_suggested": tau,
            "tau_p95": tau,
            "tau_p97": tau,
            "tau_p99": tau,
            "tau_mean": tau,
            "tau_std": 0.0,
            "tau_max": tau,
            "k_folds": 0,
            "n_samples": 0,
        }
        return stats

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
    payload = {
        "task_name": task_name,
        "dataset_name": task_name,
        **stats,
    }
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


def train_one_epoch(args, sam, align_module, optimizer, train_loader, epoch, criterion_seg, criterion_distill, loggers):
    sam.eval()
    align_module.train()

    train_loader = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)

    for it, batched_input in enumerate(train_loader):
        batched_input = to_device(batched_input, args.device)


        if random.random() < 0.5:
            batched_input["prompt"]["point_coords"] = None
        else:
            batched_input["prompt"]["bboxes"] = None
            batched_input["prompt"]["boxes"] = None


        labels = batched_input["label"]
        image_embeddings_ori = sam.image_encoder(batched_input["image"])
        image_embeddings_base = align_module(image_embeddings_ori)

        loss_distill = 0.0
        if args.distill:

            criterion_distill = criterion_distill or DistillLoss(loss_type='mse')
            loss_distill = criterion_distill(image_embeddings_base, image_embeddings_ori)


        B = image_embeddings_base.shape[0]
        image_embeddings = torch.cat(
            [image_embeddings_base[i].unsqueeze(0).repeat(args.mask_num, 1, 1, 1) for i in range(B)],
            dim=0
        )


        points = (batched_input["prompt"]["point_coords"], batched_input["prompt"]["point_labels"]) \
            if batched_input["prompt"]["point_coords"] is not None else None
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=points,
            boxes=batched_input["prompt"].get("bboxes", None),
            masks=batched_input["prompt"].get("mask_inputs", None),
        )

        sparse_embeddings = _repeat_to_match_batch(sparse_embeddings, image_embeddings.size(0))
        dense_embeddings  = _repeat_to_match_batch(dense_embeddings,  image_embeddings.size(0))

        low_res_masks, iou_pred = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
        masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

        labels_4d = labels if labels.shape == masks.shape else labels.unsqueeze(1)   # [B,1,H,W]


        loss_seg = criterion_seg(masks, labels_4d, iou_pred)
        if args.distill:
            loss = 0.3 * loss_seg + 0.7 * loss_distill
        else:
            loss = loss_seg


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())


        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            train_iter_metrics[i] += batch_metrics[i]


    l = len(train_loader)
    avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
    avg_metrics = [m / l for m in train_iter_metrics] if l > 0 else [0.0] * len(args.metrics)
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate(args, sam, align_module, test_loader, criterion_seg, epoch):
    sam.eval()
    align_module.eval()

    test_pbar = tqdm(test_loader, desc=f"Eval Epoch {epoch+1}")
    test_loss, test_metrics = [], [[] for _ in args.metrics]

    for batched_input in test_pbar:
        batched_input = to_device(batched_input, args.device)
        labels = batched_input["label"]  # [K,1,H,W]

        image_embeddings = sam.image_encoder(batched_input["image"])  # [B,256,H',W']
        image_embeddings = align_module(image_embeddings)

        gtp = batched_input["gt_prompt"]
        boxes = gtp.get('bboxes', None)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=boxes, masks=None)  # [K,N0,256], [B,256,H',W']

        B = image_embeddings.shape[0]
        K = sparse_embeddings.shape[0]
        assert K % B == 0, f"K={K} not divisible by B={B}"
        times = K // B
        image_embeddings = image_embeddings.repeat_interleave(times, dim=0)  # [K,256,H',W']

        sparse_embeddings = _repeat_to_match_batch(sparse_embeddings, image_embeddings.size(0))
        dense_embeddings  = _repeat_to_match_batch(dense_embeddings,  image_embeddings.size(0))

        low_res_masks, iou_pred = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
        masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

        labels_4d = labels if labels.shape == masks.shape else labels.unsqueeze(1)   # [K,1,H,W]

        loss = criterion_seg(masks, labels_4d, iou_pred)
        test_loss.append(loss.item())

        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            test_metrics[i].append(batch_metrics[i])

    mean_loss = float(np.mean(test_loss)) if test_loss else 0.0
    mean_metrics = [float(np.mean(m)) if m else 0.0 for m in test_metrics]
    return mean_loss, mean_metrics


def main(args):

    if args.batch_size is not None:
        if args.train_batch_size is None: args.train_batch_size = args.batch_size
        if args.eval_batch_size  is None: args.eval_batch_size  = args.batch_size

    args_train = copy.deepcopy(args); args_train.test_mode = False; args_train.batch_size = args.train_batch_size
    args_test  = copy.deepcopy(args); args_test.test_mode  = True;  args_test.batch_size  = args.eval_batch_size

    train_loader = get_loader(args_train, all_training_sets=[args.dataset_name])
    test_loader  = get_loader(args_test,  all_training_sets=[args.dataset_name])

    all_ds = [s.strip() for s in args.all_datasets.split(",") if s.strip()] if getattr(args, "all_datasets", None) else [args.dataset_name]
    if args.dataset_name not in all_ds:
        raise ValueError("dataset_name must be included in all_datasets")
    stage_idx = all_ds.index(args.dataset_name)
    seen_full = all_ds[:stage_idx + 1]


    save_root = args.save_root if args.save_root else args.work_dir
    log_dir  = args.log_dir  if args.log_dir  else os.path.join(save_root, "logs")
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(save_root, "checkpoints", args.run_name)
    os.makedirs(log_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.run_name}_{args.method}_{args.num_cnn}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log")
    loggers = get_logger(log_file)


    sam = sam_model_registry[args.model_type](args).to(args.device)
    for p in sam.parameters(): p.requires_grad = False

    align_module = build_align_module(args).to(args.device)
    if getattr(args, "align_checkpoint", None) and os.path.isfile(args.align_checkpoint):
        ckpt = torch.load(args.align_checkpoint, map_location=args.device)
        align_module.load_state_dict(ckpt, strict=True)
        print(f"[Align] Loaded align checkpoint from: {args.align_checkpoint}")


    optimizer = optim.Adam([p for p in align_module.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5) if args.lr_scheduler else None


    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trainable_params = sum(p.numel() for p in align_module.parameters() if p.requires_grad)
    loggers.info(f"Training started at {now_str}")
    loggers.info(f"Learning rate: {args.lr}")
    loggers.info(f"Trainable parameters: {trainable_params}")


    criterion_seg = FocalDiceloss_IoULoss()
    criterion_dist = DistillLoss(loss_type='mse') if args.distill else None


    last_test_loss, last_test_metrics = None, None
    for epoch in range(args.epochs):
        start_t = time.time()

        train_loss, train_metrics = train_one_epoch(
            args, sam, align_module, optimizer, train_loader, epoch,
            criterion_seg, criterion_dist, loggers
        )
        test_loss, test_metrics = evaluate(args, sam, align_module, test_loader, criterion_seg, epoch)
        last_test_loss, last_test_metrics = test_loss, test_metrics

        if scheduler: scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        msg = (f"[Epoch {epoch+1}] lr={lr_now:.6f} "
               f"Train loss={train_loss:.4f} " + " ".join([f"train_{m}={train_metrics[i]:.4f}" for i, m in enumerate(args.metrics)]) +
               f" | Test loss={test_loss:.4f} " + " ".join([f"test_{m}={test_metrics[i]:.4f}" for i, m in enumerate(args.metrics)]))
        loggers.info(msg); print(msg)

        elapsed = time.time() - start_t
        print(f"Run epoch time: {elapsed:.2f}s"); loggers.info(f"Epoch {epoch+1} finished, time {elapsed:.2f}s")


    align_ckpt_path = os.path.join(ckpt_dir, f"align_{args.method}_{args.num_cnn}.pth")
    torch.save(align_module.state_dict(), align_ckpt_path)
    print(f"[Align] Saved current align checkpoint to: {align_ckpt_path}")


    task_vae = None
    task_tau_stats = None
    if args.router_type == "vae":
        task_vae = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)

        vae_feats = _collect_vae_features(args, sam, train_loader)
        vae_loader = DataLoader(
            TensorDataset(vae_feats),
            batch_size=max(1, int(args.train_batch_size)),
            shuffle=True,
            drop_last=False,
        )
        train_task_vae(
            args=args,
            vae=task_vae,
            feature_iter_fn=vae_loader,
            epochs=int(args.vae_epochs),
            lr=float(args.vae_lr),
            device=args.device,
            logger=loggers
        )
        task_tau_stats = _estimate_tau_kfold(args, sam, train_loader, logger=loggers)
        tau_path = _save_tau_stats(args, args.dataset_name, task_tau_stats)
        if tau_path:
            loggers.info(f"[Tau] Saved -> {tau_path}")


    try:
        save_current_task_assets(args, task_name=args.dataset_name, align_module=align_module, task_vae=task_vae)
    except Exception as e:
        print(f"[WARN] Save router/adapters assets failed: {e}")


    if args.router_type == "vae":
        loggers.info(f"[VAE] Start training TaskVAE for {args.dataset_name} "
                f"(feat={args.vae_feat}, in_dim={args.vae_in_dim}, z={args.vae_latent_dim}, "
                f"epochs={args.vae_epochs}, lr={args.vae_lr})")
        adapters_dict = load_adapters_for_seen_tasks(args, args.device)
        adapters_dict[args.dataset_name] = align_module.eval()
        for p in adapters_dict[args.dataset_name].parameters(): p.requires_grad = False

        vae_dict = load_vaes_for_seen_tasks(args, args.device)
        if task_vae is not None:
            vae_dict[args.dataset_name] = task_vae.eval()
            for p in task_vae.parameters(): p.requires_grad = False

        adapters_dict = {ds: adapters_dict[ds] for ds in seen_full if ds in adapters_dict}
        vae_dict = {ds: vae_dict[ds] for ds in seen_full if ds in vae_dict}

        for ds in seen_full:
            if ds not in adapters_dict or adapters_dict[ds] is None:
                ck_ds = None
                if getattr(args, "adapters_ckpt_dir", None):
                    ck_ds = os.path.join(args.adapters_ckpt_dir, ds, f"align_{args.method}_{args.num_cnn}.pth")
                    if not os.path.isfile(ck_ds):
                        ck_ds = None
                if ck_ds:
                    m = build_align_module(args).to(args.device)
                    st = torch.load(ck_ds, map_location=args.device)
                    try:
                        m.load_state_dict(st, strict=True)
                    except Exception:
                        st2 = {k.replace("module.", ""): v for k, v in st.items()}
                        m.load_state_dict(st2, strict=False)
                    m.eval()
                    adapters_dict[ds] = m
            if ds not in vae_dict or vae_dict[ds] is None:
                vp = os.path.join(args.router_ckpt_dir, ds, "vae.pth") if getattr(args, "router_ckpt_dir", None) else None
                if vp and os.path.isfile(vp):
                    v = TaskVAE(in_dim=int(args.vae_in_dim), latent_dim=int(args.vae_latent_dim)).to(args.device)
                    v.load_state_dict(torch.load(vp, map_location=args.device))
                    v.eval()
                    vae_dict[ds] = v

        args.router_thresholds = {}
        for task in seen_full:
            stats = _load_tau_stats(args, task)
            args.router_thresholds[task] = _tau_value_from_stats(
                stats, DEFAULT_TASK_TAUS.get(task, 1.0)
            )
        if task_tau_stats is not None:
            args.router_thresholds[args.dataset_name] = _tau_value_from_stats(
                task_tau_stats, DEFAULT_TASK_TAUS.get(args.dataset_name, 1.0)
            )
        args.router_threshold = float(args.router_thresholds.get(args.dataset_name, 1.0))

        loss_r, metrics_r = evaluate_moda_vae(
            args, sam, adapters_dict, test_loader, criterion_seg, epoch=args.epochs-1,
            vae_dict=vae_dict, zero_adapter_mode=args.zero_adapter_mode
        )
        msg_r = " ".join([f"{m}={metrics_r[i]:.4f}" for i, m in enumerate(args.metrics)])
        loggers.info(f"[Eval-VAE-Route][{args.dataset_name}] loss={loss_r:.4f} {msg_r}")
        print(f"[Eval-VAE-Route][{args.dataset_name}] loss={loss_r:.4f} {msg_r}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
