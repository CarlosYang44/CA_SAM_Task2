# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from torch.nn import functional as F

from .feature_pool import extract_feature_for_vae
from .vae_router import route_with_vae, score_elbo_all_tasks
from .adapter_hub import build_zero_adapter

def _to_list(x):
    if x is None:
        return []
    if torch.is_tensor(x):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, np.ndarray):
        return np.reshape(x, -1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)

    try:
        return [float(x)]
    except Exception:
        return []

def _repeat_to_match_batch(x, target_B):
    if x is None:
        return None
    if x.size(0) == target_B:
        return x
    assert target_B % x.size(0) == 0, f"Can't tile from {x.size(0)} to {target_B}"
    times = target_B // x.size(0)
    return x.repeat_interleave(times, dim=0)

def _move_gt_prompt_to_device(gtp, device):
    if gtp is None: return {}
    out = {}
    for k, v in gtp.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _default_router_tau(args) -> float:
    return 1.0


def _build_tau_vector(args, task_names, device, dtype):
    tau_map = getattr(args, "router_thresholds", None)
    default_tau = _default_router_tau(args)
    if isinstance(tau_map, dict) and len(tau_map) > 0:
        return torch.tensor(
            [float(tau_map.get(name, default_tau)) for name in task_names],
            device=device,
            dtype=dtype,
        )
    return torch.full((len(task_names),), default_tau, device=device, dtype=dtype)


def _route_scores_with_optional_task_taus(args, task_names, elbo_scores):
    min_vals, min_idx = torch.min(elbo_scores, dim=1)
    tau_vec = _build_tau_vector(args, task_names, elbo_scores.device, elbo_scores.dtype)
    chosen_tau = tau_vec[min_idx]
    is_unknown = (min_vals > chosen_tau)
    return min_idx, is_unknown, chosen_tau, min_vals


def _route_features_with_optional_task_taus(args, f, vae_dict):
    scores, task_names = score_elbo_all_tasks(
        f,
        vae_dict,
        beta=float(getattr(args, "vae_beta", 1.0)),
        device=args.device,
    )
    min_idx, is_unknown, chosen_tau, min_vals = _route_scores_with_optional_task_taus(args, task_names, scores)
    return min_idx, is_unknown, task_names, scores, chosen_tau, min_vals


@torch.no_grad()
def evaluate_moda_vae(
    args, sam, adapters_dict: Dict[str, torch.nn.Module], test_loader, criterion_seg, epoch: int,
    vae_dict: Dict[str, torch.nn.Module], zero_adapter_mode: str = "zeros"
):
    sam.eval()

    for m in adapters_dict.values():
        m.eval()
    zero_adapter = build_zero_adapter(mode=zero_adapter_mode)

    test_pbar = tqdm(test_loader, desc=f"Eval(VAE) Epoch {epoch+1}")
    test_loss, test_metrics = [], [[] for _ in args.metrics]


    debug_every = 10
    all_min_elbos = []
    per_task_elbo_sum = {}   # {task: sum}
    per_task_elbo_cnt = {}   # {task: count}
    task_names_cache = None

    def _assert_batch_match(img, sp, de, lab):
        b_img = img.size(0)
        if sp is not None: assert sp.size(0) == b_img, f"sparse B={sp.size(0)} != image B={b_img}"
        if de is not None: assert de.size(0) == b_img, f"dense  B={de.size(0)} != image B={b_img}"
        assert lab.size(0) == b_img, f"label B={lab.size(0)} != image B={b_img}"

    from metrics import SegMetrics

    for step, batched_input in enumerate(test_pbar):

        images = batched_input["image"].to(args.device)
        labels = batched_input["label"].to(args.device)

        if args.vae_feat == "cls":

            token_len = args.cls_token_len
            embed_dim = args.vae_in_dim
            base_cls = getattr(sam.image_encoder, "cls_token", None)
            if base_cls is None:
                base_cls = torch.randn(1, token_len, embed_dim, device=args.device)

            out = sam.image_encoder(images, cls_token=base_cls)

            if isinstance(out, tuple) and len(out) == 3:
                conv_feat, _, cls_tok = out
            elif isinstance(out, tuple) and len(out) == 2:
                conv_feat, _ = out
                cls_tok = None
            else:
                conv_feat = out
                cls_tok = None

            assert cls_tok is not None, "use_moda_encoder=True with vae_feat=cls requires the encoder to return cls_token"

            f = extract_feature_for_vae(conv_feat, mode="cls", cls_token=cls_tok)   # [B, 768]

            image_embeddings_ori = conv_feat                                       # [B, 256, H', W']

        elif args.vae_feat == "attn_pool":

            out = sam.image_encoder(images)
            conv_feat = out[0] if isinstance(out, tuple) else out
            #attn_temp = float(getattr(args, "attn_temp", 1.0)) ###
            f = extract_feature_for_vae(conv_feat, mode="attn_pool", attn_temp=1.0) # [B, 256]
            image_embeddings_ori = conv_feat                                        # [B, 256, H', W']

        else:

            emb = sam.image_encoder(images)
            image_embeddings_ori = emb[0] if isinstance(emb, tuple) else emb        # [B, 256, H', W']
            f = extract_feature_for_vae(image_embeddings_ori, mode=args.vae_feat)   # [B, D]



        if len(vae_dict) > 0:
            min_idx, is_unknown, task_names_sorted, elbo_scores, chosen_tau, min_vals = _route_features_with_optional_task_taus(
                args, f, vae_dict
            )
            task_names_cache = task_names_sorted


            all_min_elbos.append(min_vals.detach().cpu())


            elbo_np = elbo_scores.detach().cpu().float().numpy()  # [B, T]
            for t_idx, t_name in enumerate(task_names_sorted):
                v = float(np.mean(elbo_np[:, t_idx]))
                per_task_elbo_sum[t_name] = per_task_elbo_sum.get(t_name, 0.0) + v
                per_task_elbo_cnt[t_name] = per_task_elbo_cnt.get(t_name, 0) + 1


            mv = min_vals.detach().cpu().float().numpy()
            p25, p50, p75 = np.percentile(mv, [25, 50, 75])
            unk_ratio = float(torch.mean(is_unknown.float()).item())
            tau_disp = float(chosen_tau.float().mean().item()) if chosen_tau.numel() > 0 else _default_router_tau(args)

            per_task_mean_str = ", ".join([
                f"{t}:{float(np.mean(elbo_np[:, j])):.4f}"
                for j, t in enumerate(task_names_sorted)
            ])
            test_pbar.write(f"[VAE-LOGITS][batch {step}] per-task mean ELBO: {per_task_mean_str}")
            test_pbar.write(f"[VAE-LOGITS][batch {step}] min-ELBO p25/50/75 = {p25:.4f}/{p50:.4f}/{p75:.4f} | unk%(@tau={tau_disp:.4f})={100*unk_ratio:.1f}%")

            if (step % debug_every) == 0:
                p05, p95 = np.percentile(mv, [5, 95])
                test_pbar.write(f"[VAE-LOGITS][batch {step}] min-ELBO p05/50/95 = {p05:.4f}/{p50:.4f}/{p95:.4f}")

        else:
            task_names_sorted = []
            elbo_scores = torch.full((f.size(0), 0), 0.0, device=f.device)
            min_idx = torch.zeros(f.size(0), dtype=torch.long, device=f.device)
            is_unknown = torch.ones(f.size(0), dtype=torch.bool, device=f.device)


        gtp = _move_gt_prompt_to_device(batched_input.get("gt_prompt", {}), args.device)
        boxes = gtp.get('bboxes', None)
        if boxes is not None and torch.is_tensor(boxes):
            boxes = boxes.to(args.device, non_blocking=True)
        sparse_embeddings_full, dense_embeddings_full = sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        B = image_embeddings_ori.shape[0]
        K = sparse_embeddings_full.shape[0]
        assert K % B == 0, f"K={K} not divisible by B={B}"
        times = K // B


        groups: Dict[str, List[int]] = {}
        for b in range(B):
            if bool(is_unknown[b].item()):
                groups.setdefault("__unknown__", []).append(b)
            else:
                tname = task_names_sorted[min_idx[b].item()]
                groups.setdefault(tname, []).append(b)


        masks_collector = [None for _ in range(B * times)]
        iou_collector = [None for _ in range(B * times)]

        loss_accum = 0.0
        nloss = 0

        for key, idxs in groups.items():
            idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=args.device)
            ori_sub = image_embeddings_ori.index_select(dim=0, index=idxs_tensor)  # [b_sub, C, H', W']

            if key == "__unknown__":
                base_sub = zero_adapter(ori_sub)  # [b_sub, C, H', W']
            else:
                adapter = adapters_dict.get(key, None)
                base_sub = zero_adapter(ori_sub) if (adapter is None) else adapter(ori_sub)


            b_sub = base_sub.size(0)
            if b_sub == 0:
                continue
            img_sub_rep = torch.cat([base_sub[i].unsqueeze(0).repeat(times, 1, 1, 1) for i in range(b_sub)], dim=0)


            segs = [torch.arange(i * times, (i + 1) * times, device=args.device) for i in idxs]
            kk = torch.cat(segs, dim=0)  # [b_sub * times]
            sp_sub = sparse_embeddings_full.index_select(dim=0, index=kk)
            de_sub = _repeat_to_match_batch(dense_embeddings_full, img_sub_rep.size(0))


            low_res_masks_sub, iou_pred_sub = sam.mask_decoder(
                image_embeddings=img_sub_rep,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp_sub,
                dense_prompt_embeddings=de_sub,
                multimask_output=args.multimask,
            )
            masks_sub = F.interpolate(low_res_masks_sub, (args.image_size, args.image_size),
                                      mode="bilinear", align_corners=False)


            for c, i in enumerate(idxs):
                start = i * times
                end = start + times
                ss = c * times
                ee = ss + times
                masks_collector[start:end] = torch.unbind(masks_sub[ss:ee], dim=0)
                iou_collector[start:end] = torch.unbind(iou_pred_sub[ss:ee], dim=0)


            labels_4d = labels if labels.dim() == masks_sub.dim() else labels.unsqueeze(1)
            labels_group = torch.cat([labels_4d[i * times: (i + 1) * times] for i in idxs], dim=0)

            loss_sub = criterion_seg(masks_sub, labels_group, iou_pred_sub)
            loss_accum += float(loss_sub.item()) * (b_sub * times)
            nloss += (b_sub * times)


        masks = torch.stack(masks_collector, dim=0)
        iou_pred = torch.stack(iou_collector, dim=0)

        labels_4d = labels if labels.dim() == masks.dim() else labels.unsqueeze(1)
        _assert_batch_match(masks, sparse_embeddings_full, dense_embeddings_full, labels_4d)


        loss = loss_accum / max(1, nloss)
        test_loss.append(loss)

        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            test_metrics[i].append(batch_metrics[i])


        if len(vae_dict) > 0:
            unk_ratio = float(torch.mean(is_unknown.float()).item())
            test_pbar.set_postfix({"unk%": f"{100*unk_ratio:.1f}"})


    mean_loss = float(np.mean(test_loss)) if test_loss else 0.0
    mean_metrics = [float(np.mean(m)) if m else 0.0 for m in test_metrics]

    if all_min_elbos:
        all_min = torch.cat(all_min_elbos, dim=0).numpy()
        p01, p05, p50, p95, p99 = np.percentile(all_min, [1, 5, 50, 95, 99])
        print("[VAE-LOGITS][global] min-ELBO percentiles -> "
              f"p01={p01:.4f}, p05={p05:.4f}, p50={p50:.4f}, p95={p95:.4f}, p99={p99:.4f}")
        if task_names_cache is not None:
            means = []
            for t in task_names_cache:
                s = per_task_elbo_sum.get(t, 0.0); c = per_task_elbo_cnt.get(t, 1)
                means.append(f"{t}:{(s/c):.4f}")
            print("[VAE-LOGITS][global] per-task mean ELBO -> " + ", ".join(means))
        print("[VAE-LOGITS][hint] You can initialize --router_threshold from a high percentile "
              "of known-task min-ELBO values in this chain, such as p95 or p99, then tune it using the unknown ratio and accuracy curve.")

    return mean_loss, mean_metrics


@torch.no_grad()
def evaluate_moda_vae_with_pool(
    args,
    sam,
    adapters_dict: Dict[str, torch.nn.Module],
    test_loader,
    criterion_seg,
    epoch: int,
    vae_dict: Dict[str, torch.nn.Module],
    zero_adapter_mode: str = "zeros",
    pool: torch.nn.Module = None,
):
    sam.eval()
    for m in adapters_dict.values():
        m.eval()
    from .adapter_hub import build_zero_adapter
    zero_adapter = build_zero_adapter(mode=zero_adapter_mode)

    test_pbar = tqdm(test_loader, desc=f"Eval(VAE+Pool) Epoch {epoch+1}")
    test_loss, test_metrics = [], [[] for _ in args.metrics]


    debug_every = 10
    all_min_elbos = []
    per_task_elbo_sum, per_task_elbo_cnt = {}, {}
    task_names_cache = None

    from metrics import SegMetrics

    for step, batched_input in enumerate(test_pbar):

        images = batched_input["image"].to(args.device)
        labels = batched_input["label"].to(args.device)

        if args.vae_feat == "cls":
            token_len = args.cls_token_len
            embed_dim = args.vae_in_dim
            base_cls = getattr(sam.image_encoder, "cls_token", None)
            if base_cls is None:
                base_cls = torch.randn(1, token_len, embed_dim, device=args.device)

            out = sam.image_encoder(images, cls_token=base_cls)
            if isinstance(out, tuple) and len(out) == 3:
                conv_feat, _, cls_tok = out
            elif isinstance(out, tuple) and len(out) == 2:
                conv_feat, _ = out
                cls_tok = None
            else:
                conv_feat = out
                cls_tok = None

            assert cls_tok is not None, "use_moda_encoder=True with vae_feat=cls requires the encoder to return cls_token"
            from .feature_pool import extract_feature_for_vae
            f = extract_feature_for_vae(conv_feat, mode="cls", cls_token=cls_tok)  # [B, D=vae_in_dim]
            image_embeddings_ori = conv_feat

        elif args.vae_feat == "self_attn":

            assert pool is not None, "vae_feat=self_attn requires pool"
            out = sam.image_encoder(images)
            conv_feat = out[0] if isinstance(out, tuple) else out         # [B,256,H',W']
            f = pool(conv_feat)                                            # [B,D]
            image_embeddings_ori = conv_feat

        elif args.vae_feat == "attn_pool":
            out = sam.image_encoder(images)
            conv_feat = out[0] if isinstance(out, tuple) else out
            from .feature_pool import extract_feature_for_vae
            f = extract_feature_for_vae(conv_feat, mode="attn_pool", attn_temp=1.0)  # [B,256]
            image_embeddings_ori = conv_feat

        else:
            out = sam.image_encoder(images)
            conv_feat = out[0] if isinstance(out, tuple) else out
            from .feature_pool import extract_feature_for_vae
            f = extract_feature_for_vae(conv_feat, mode=args.vae_feat)                # [B,D]
            image_embeddings_ori = conv_feat


        if len(vae_dict) > 0:
            min_idx, is_unknown, task_names_sorted, elbo_scores, chosen_tau, min_vals = _route_features_with_optional_task_taus(
                args, f, vae_dict
            )

            all_min_elbos.append(min_vals.detach().cpu())
            if task_names_cache is None:
                task_names_cache = task_names_sorted
            for b in range(f.size(0)):
                t = "__unknown__" if bool(is_unknown[b].item()) else task_names_sorted[min_idx[b].item()]
                v = float(elbo_scores[b, min_idx[b]].item())
                per_task_elbo_sum[t] = per_task_elbo_sum.get(t, 0.0) + v
                per_task_elbo_cnt[t] = per_task_elbo_cnt.get(t, 0) + 1
        else:

            B = f.size(0)
            min_idx = torch.zeros(B, dtype=torch.long, device=args.device)
            is_unknown = torch.ones(B, dtype=torch.bool, device=args.device)
            task_names_sorted = []


        def _repeat_to_match_batch(x, target_B):
            if x is None: return None
            if x.size(0) == target_B: return x
            assert target_B % x.size(0) == 0, f"Can't tile from {x.size(0)} to {target_B}"
            times = target_B // x.size(0)
            return x.repeat_interleave(times, dim=0)

        def _move_gt_prompt_to_device(gtp, device):
            if gtp is None: return {}
            out = {}
            for k, v in gtp.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device, non_blocking=True)
                else:
                    out[k] = v
            return out

        gtp = _move_gt_prompt_to_device(batched_input.get("gt_prompt", {}), args.device)
        boxes = gtp.get('bboxes', None)
        if boxes is not None and torch.is_tensor(boxes):
            boxes = boxes.to(args.device, non_blocking=True)
        sparse_embeddings_full, dense_embeddings_full = sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        B = image_embeddings_ori.shape[0]
        K = sparse_embeddings_full.shape[0]
        assert K % B == 0, f"K={K} not divisible by B={B}"
        times = K // B


        groups: Dict[str, List[int]] = {}
        for b in range(B):
            if bool(is_unknown[b].item()):
                groups.setdefault("__unknown__", []).append(b)
            else:
                tname = task_names_sorted[min_idx[b].item()]
                groups.setdefault(tname, []).append(b)


        masks_collector = [None for _ in range(B * times)]
        iou_collector = [None for _ in range(B * times)]

        loss_accum = 0.0
        nloss = 0

        for key, idxs in groups.items():
            idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=args.device)
            ori_sub = image_embeddings_ori.index_select(dim=0, index=idxs_tensor)  # [b_sub, C, H', W']

            if key == "__unknown__":
                base_sub = zero_adapter(ori_sub)
            else:
                adapter = adapters_dict.get(key, None)
                base_sub = zero_adapter(ori_sub) if (adapter is None) else adapter(ori_sub)


            b_sub = base_sub.size(0)
            if b_sub == 0:
                continue
            img_sub_rep = torch.cat([base_sub[i].unsqueeze(0).repeat(times, 1, 1, 1) for i in range(b_sub)], dim=0)


            segs = [torch.arange(i * times, (i + 1) * times, device=args.device) for i in idxs]
            kk = torch.cat(segs, dim=0)  # [b_sub * times]
            sp_sub = sparse_embeddings_full.index_select(dim=0, index=kk)
            de_sub = _repeat_to_match_batch(dense_embeddings_full, img_sub_rep.size(0))


            low_res_masks_sub, iou_pred_sub = sam.mask_decoder(
                image_embeddings=img_sub_rep,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp_sub,
                dense_prompt_embeddings=de_sub,
                multimask_output=args.multimask,
            )
            masks_sub = F.interpolate(low_res_masks_sub, (args.image_size, args.image_size),
                                      mode="bilinear", align_corners=False)


            for c, i in enumerate(idxs):
                start = i * times
                end = start + times
                ss = c * times
                ee = ss + times
                masks_collector[start:end] = torch.unbind(masks_sub[ss:ee], dim=0)
                iou_collector[start:end] = torch.unbind(iou_pred_sub[ss:ee], dim=0)


            labels_4d = labels if labels.dim() == masks_sub.dim() else labels.unsqueeze(1)
            labels_group = labels_4d.index_select(dim=0, index=kk)
            loss_sub = criterion_seg(masks_sub, labels_group, iou_pred_sub)

            loss_accum += float(loss_sub.item()) * (b_sub * times)
            nloss += (b_sub * times)


        masks = torch.stack(masks_collector, dim=0)  # [K, 1, H, W]
        iou_pred = torch.stack(iou_collector, dim=0)

        labels_4d = labels if labels.dim() == masks.dim() else labels.unsqueeze(1)


        loss = loss_accum / max(1, nloss)
        test_loss.append(loss)

        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            test_metrics[i].append(batch_metrics[i])


        if len(vae_dict) > 0:
            unk_ratio = float(torch.mean(is_unknown.float()).item())
            test_pbar.set_postfix({"unk%": f"{100*unk_ratio:.1f}"})


    mean_loss = float(np.mean(test_loss)) if test_loss else 0.0
    mean_metrics = [float(np.mean(m)) if m else 0.0 for m in test_metrics]

    if all_min_elbos:
        all_min = torch.cat(all_min_elbos, dim=0).numpy()
        p01, p05, p50, p95, p99 = np.percentile(all_min, [1, 5, 50, 95, 99])
        print("[VAE-LOGITS][global] min-ELBO percentiles -> "
              f"p01={p01:.4f}, p05={p05:.4f}, p50={p50:.4f}, p95={p95:.4f}, p99={p99:.4f}")
        if task_names_cache is not None:
            means = []
            for t in task_names_cache:
                s = per_task_elbo_sum.get(t, 0.0); c = per_task_elbo_cnt.get(t, 1)
                means.append(f"{t}:{(s/c):.4f}")
            print("[VAE-LOGITS][global] per-task mean ELBO -> " + ", ".join(means))

    return mean_loss, mean_metrics


@torch.no_grad()
def evaluate_moda_vae_with_learnable_pool(
    args,
    sam,
    adapters_dict: Dict[str, torch.nn.Module],
    test_loader,
    criterion_seg,
    epoch: int,
    vae_dict: Dict[str, torch.nn.Module],
    zero_adapter_mode: str = "zeros",
    pool: torch.nn.Module = None,
):
    assert pool is not None, "evaluate_moda_vae_with_learnable_pool requires pool"
    sam.eval()
    for m in adapters_dict.values():
        m.eval()

    from .adapter_hub import build_zero_adapter
    zero_adapter = build_zero_adapter(mode=zero_adapter_mode)

    test_pbar = tqdm(test_loader, desc=f"Eval(VAE+LearnPool) Epoch {epoch+1}")
    test_loss, test_metrics = [], [[] for _ in args.metrics]


    debug_every = 10
    all_min_elbos = []
    per_task_elbo_sum, per_task_elbo_cnt = {}, {}
    task_names_cache = None

    from metrics import SegMetrics

    for step, batched_input in enumerate(test_pbar):
        images = batched_input["image"].to(args.device)
        labels = batched_input["label"].to(args.device)


        out = sam.image_encoder(images)
        conv_feat = out[0] if isinstance(out, tuple) else out       # [B,256,H',W']
        f = pool(conv_feat)                                         # [B,D=vae_in_dim]
        image_embeddings_ori = conv_feat


        if len(vae_dict) > 0:
            min_idx, is_unknown, task_names_sorted, elbo_scores, chosen_tau, min_vals = _route_features_with_optional_task_taus(
                args, f, vae_dict
            )

            all_min_elbos.append(min_vals.detach().cpu())
            if task_names_cache is None:
                task_names_cache = task_names_sorted
            elbo_np = elbo_scores.detach().cpu().float().numpy()
            for t_idx, t_name in enumerate(task_names_sorted):
                v = float(np.mean(elbo_np[:, t_idx]))
                per_task_elbo_sum[t_name] = per_task_elbo_sum.get(t_name, 0.0) + v
                per_task_elbo_cnt[t_name] = per_task_elbo_cnt.get(t_name, 0) + 1
        else:
            B = f.size(0)
            min_idx = torch.zeros(B, dtype=torch.long, device=args.device)
            is_unknown = torch.ones(B, dtype=torch.bool, device=args.device)
            task_names_sorted = []


        gtp = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in (batched_input.get("gt_prompt", {}) or {}).items()}
        boxes = gtp.get('bboxes', None)
        if boxes is not None and torch.is_tensor(boxes):
            boxes = boxes.to(args.device, non_blocking=True)

        sparse_embeddings_full, dense_embeddings_full = sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        B = image_embeddings_ori.shape[0]
        K = sparse_embeddings_full.shape[0]
        assert K % B == 0, f"K={K} not divisible by B={B}"
        times = K // B


        groups: Dict[str, List[int]] = {}
        for b in range(B):
            if bool(is_unknown[b].item()):
                groups.setdefault("__unknown__", []).append(b)
            else:
                tname = task_names_sorted[min_idx[b].item()]
                groups.setdefault(tname, []).append(b)

        masks_collector = [None for _ in range(B * times)]
        iou_collector = [None for _ in range(B * times)]
        loss_accum, nloss = 0.0, 0

        for key, idxs in groups.items():
            idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=args.device)
            ori_sub = image_embeddings_ori.index_select(dim=0, index=idxs_tensor)  # [b_sub, C, H', W']

            if key == "__unknown__":
                base_sub = zero_adapter(ori_sub)
            else:
                adapter = adapters_dict.get(key, None)
                base_sub = zero_adapter(ori_sub) if (adapter is None) else adapter(ori_sub)

            b_sub = base_sub.size(0)
            if b_sub == 0:
                continue

            img_sub_rep = torch.cat([base_sub[i].unsqueeze(0).repeat(times, 1, 1, 1) for i in range(b_sub)], dim=0)

            segs = [torch.arange(i * times, (i + 1) * times, device=args.device) for i in idxs]
            kk = torch.cat(segs, dim=0)
            sp_sub = sparse_embeddings_full.index_select(dim=0, index=kk)
            de_sub = (dense_embeddings_full if (dense_embeddings_full is None or dense_embeddings_full.size(0) == img_sub_rep.size(0))
                      else dense_embeddings_full.repeat_interleave(img_sub_rep.size(0) // dense_embeddings_full.size(0), dim=0))

            low_res_masks_sub, iou_pred_sub = sam.mask_decoder(
                image_embeddings=img_sub_rep,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp_sub,
                dense_prompt_embeddings=de_sub,
                multimask_output=args.multimask,
            )
            masks_sub = F.interpolate(low_res_masks_sub, (args.image_size, args.image_size),
                                      mode="bilinear", align_corners=False)

            for c, i in enumerate(idxs):
                start = i * times; end = start + times
                ss = c * times;    ee = ss + times
                masks_collector[start:end] = torch.unbind(masks_sub[ss:ee], dim=0)
                iou_collector[start:end]   = torch.unbind(iou_pred_sub[ss:ee], dim=0)

            labels_4d = labels if labels.dim() == masks_sub.dim() else labels.unsqueeze(1)
            labels_group = labels_4d.index_select(dim=0, index=kk)
            loss_sub = criterion_seg(masks_sub, labels_group, iou_pred_sub)

            loss_accum += float(loss_sub.item()) * (b_sub * times)
            nloss += (b_sub * times)

        masks = torch.stack(masks_collector, dim=0)
        iou_pred = torch.stack(iou_collector, dim=0)

        labels_4d = labels if labels.dim() == masks.dim() else labels.unsqueeze(1)
        loss = loss_accum / max(1, nloss)
        test_loss.append(loss)

        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            test_metrics[i].append(batch_metrics[i])

        if len(vae_dict) > 0:
            unk_ratio = float(torch.mean(is_unknown.float()).item())
            test_pbar.set_postfix({"unk%": f"{100*unk_ratio:.1f}"})

    mean_loss = float(np.mean(test_loss)) if test_loss else 0.0
    mean_metrics = [float(np.mean(m)) if m else 0.0 for m in test_metrics]

    if all_min_elbos:
        all_min = torch.cat(all_min_elbos, dim=0).numpy()
        p01, p05, p50, p95, p99 = np.percentile(all_min, [1, 5, 50, 95, 99])
        print("[VAE-LOGITS][global] min-ELBO percentiles -> "
              f"p01={p01:.4f}, p05={p05:.4f}, p50={p50:.4f}, p95={p95:.4f}, p99={p99:.4f}")
        if task_names_cache is not None:
            means = []
            for t in task_names_cache:
                s = per_task_elbo_sum.get(t, 0.0); c = per_task_elbo_cnt.get(t, 1)
                means.append(f"{t}:{(s/c):.4f}")
            print("[VAE-LOGITS][global] per-task mean ELBO -> " + ", ".join(means))

    return mean_loss, mean_metrics


@torch.no_grad()
def evaluate_moda_vae_with_learnable_pools(
    args,
    sam,
    adapters_dict: Dict[str, torch.nn.Module],
    test_loader,
    criterion_seg,
    epoch: int,
    vae_dict: Dict[str, torch.nn.Module],
    pool_dict: Dict[str, torch.nn.Module],
    zero_adapter_mode: str = "zeros",
):
    from .adapter_hub import build_zero_adapter
    from .vae_router import elbo_loss

    sam.eval()
    for m in adapters_dict.values():
        m.eval()
    zero_adapter = build_zero_adapter(mode=zero_adapter_mode)


    task_names = sorted(list(set(vae_dict.keys()) & set(pool_dict.keys())))
    assert len(task_names) > 0, "pool_dict and vae_dict have no task intersection; check that both were loaded"

    test_pbar = tqdm(test_loader, desc=f"Eval(VAE+PerTaskPools) Epoch {epoch+1}")
    test_loss, test_metrics = [], [[] for _ in args.metrics]


    debug_every = 10
    all_min_elbos = []
    per_task_elbo_sum, per_task_elbo_cnt = {}, {}
    from metrics import SegMetrics

    beta = float(getattr(args, "vae_beta", 1.0))

    for step, batched_input in enumerate(test_pbar):
        images = batched_input["image"].to(args.device)
        labels = batched_input["label"].to(args.device)


        out = sam.image_encoder(images)
        conv_feat = out[0] if isinstance(out, tuple) else out      # [B,256,H',W']
        B = conv_feat.size(0)


        elbo_cols = []
        for t in task_names:
            pool_t = pool_dict[t].to(args.device).eval()
            vae_t  = vae_dict[t].to(args.device).eval()
            f_t = pool_t(conv_feat)                    # [B, D]
            recon, mu, logvar = vae_t(f_t)
            elbo_t = elbo_loss(recon, f_t, mu, logvar, beta=beta)  # [B]
            elbo_cols.append(elbo_t.unsqueeze(1))
        elbo_scores = torch.cat(elbo_cols, dim=1)  # [B, T]


        min_idx, is_unknown, chosen_tau, min_vals = _route_scores_with_optional_task_taus(args, task_names, elbo_scores)


        all_min_elbos.append(min_vals.detach().cpu())
        elbo_np = elbo_scores.detach().cpu().float().numpy()
        for t_idx, t_name in enumerate(task_names):
            v = float(np.mean(elbo_np[:, t_idx]))
            per_task_elbo_sum[t_name] = per_task_elbo_sum.get(t_name, 0.0) + v
            per_task_elbo_cnt[t_name] = per_task_elbo_cnt.get(t_name, 0) + 1


        gtp = batched_input.get("gt_prompt", {}) or {}
        gtp = {k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in gtp.items()}
        boxes = gtp.get('bboxes', None)
        if boxes is not None and torch.is_tensor(boxes):
            boxes = boxes.to(args.device, non_blocking=True)

        sparse_embeddings_full, dense_embeddings_full = sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        K = sparse_embeddings_full.shape[0]
        assert K % B == 0, f"K={K} not divisible by B={B}"
        times = K // B


        groups: Dict[str, List[int]] = {}
        for b in range(B):
            if bool(is_unknown[b].item()):
                groups.setdefault("__unknown__", []).append(b)
            else:
                tname = task_names[min_idx[b].item()]
                groups.setdefault(tname, []).append(b)

        masks_collector = [None for _ in range(B * times)]
        iou_collector   = [None for _ in range(B * times)]
        loss_accum, nloss = 0.0, 0

        for key, idxs in groups.items():
            if len(idxs) == 0: continue
            idxs_tensor = torch.as_tensor(idxs, dtype=torch.long, device=args.device)
            ori_sub = conv_feat.index_select(dim=0, index=idxs_tensor)   # [b_sub, C, H', W']

            if key == "__unknown__":
                base_sub = zero_adapter(ori_sub)
            else:
                adapter = adapters_dict.get(key, None)
                base_sub = zero_adapter(ori_sub) if (adapter is None) else adapter(ori_sub)

            b_sub = base_sub.size(0)
            img_sub_rep = torch.cat([base_sub[i].unsqueeze(0).repeat(times, 1, 1, 1) for i in range(b_sub)], dim=0)

            segs = [torch.arange(i * times, (i + 1) * times, device=args.device) for i in idxs]
            kk = torch.cat(segs, dim=0)
            sp_sub = sparse_embeddings_full.index_select(dim=0, index=kk)

            if dense_embeddings_full is None or dense_embeddings_full.size(0) == img_sub_rep.size(0):
                de_sub = dense_embeddings_full
            else:
                de_sub = dense_embeddings_full.repeat_interleave(img_sub_rep.size(0) // dense_embeddings_full.size(0), dim=0)

            low_res_masks_sub, iou_pred_sub = sam.mask_decoder(
                image_embeddings=img_sub_rep,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp_sub,
                dense_prompt_embeddings=de_sub,
                multimask_output=args.multimask,
            )
            masks_sub = F.interpolate(low_res_masks_sub, (args.image_size, args.image_size),
                                      mode="bilinear", align_corners=False)

            for c, i in enumerate(idxs):
                start = i * times; end = start + times
                ss = c * times;    ee = ss + times
                masks_collector[start:end] = torch.unbind(masks_sub[ss:ee], dim=0)
                iou_collector[start:end]   = torch.unbind(iou_pred_sub[ss:ee], dim=0)

            labels_4d = labels if labels.dim() == masks_sub.dim() else labels.unsqueeze(1)
            labels_group = labels_4d.index_select(dim=0, index=kk)
            loss_sub = criterion_seg(masks_sub, labels_group, iou_pred_sub)

            loss_accum += float(loss_sub.item()) * (b_sub * times)
            nloss += (b_sub * times)

        masks   = torch.stack(masks_collector, dim=0)
        iou_pred= torch.stack(iou_collector,   dim=0)

        labels_4d = labels if labels.dim() == masks.dim() else labels.unsqueeze(1)
        loss = loss_accum / max(1, nloss)
        test_loss.append(loss)

        cal_m_preds  = (masks  > 0.0)
        cal_m_labels = (labels_4d > 0)
        if cal_m_preds.shape != cal_m_labels.shape:
            cal_m_labels = cal_m_labels.unsqueeze(1)

        batch_metrics = SegMetrics(cal_m_preds, cal_m_labels, args.metrics)
        for i in range(len(args.metrics)):
            test_metrics[i].append(batch_metrics[i])

        unk_ratio = float(torch.mean(is_unknown.float()).item())
        test_pbar.set_postfix({"unk%": f"{100*unk_ratio:.1f}"})
        if (step % debug_every) == 0:
            mv = min_vals.detach().cpu().float().numpy()
            p05, p50, p95 = np.percentile(mv, [5, 50, 95])
            test_pbar.write(f"[VAE-LOGITS][batch {step}] min-ELBO p05/50/95 = {p05:.4f}/{p50:.4f}/{p95:.4f}")

    mean_loss = float(np.mean(test_loss)) if test_loss else 0.0
    mean_metrics = [float(np.mean(m)) if m else 0.0 for m in test_metrics]

    if all_min_elbos:
        all_min = torch.cat(all_min_elbos, dim=0).numpy()
        p01, p05, p50, p95, p99 = np.percentile(all_min, [1, 5, 50, 95, 99])
        print("[VAE-LOGITS][global] min-ELBO percentiles -> "
              f"p01={p01:.4f}, p05={p05:.4f}, p50={p50:.4f}, p95={p95:.4f}, p99={p99:.4f}")
        means = []
        for t in task_names:
            s = per_task_elbo_sum.get(t, 0.0); c = per_task_elbo_cnt.get(t, 1)
            means.append(f"{t}:{(s/c):.4f}")
        print("[VAE-LOGITS][global] per-task mean ELBO -> " + ", ".join(means))

    return mean_loss, mean_metrics
