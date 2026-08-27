import torch
import numpy as np
import cv2
import torch.nn.functional as F

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def hausdorff_distance_95(pr, gt, threshold=0.5, percentile=0.95):
    pr_, gt_ = _list_tensor(pr, gt)

    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)

    pr_ = pr_.float()
    gt_ = gt_.float()

    def get_mask_coordinates(mask):
        return torch.nonzero(mask)

    def compute_channel_distance(pr_, gt_):
        pr_coords = get_mask_coordinates(pr_)
        gt_coords = get_mask_coordinates(gt_)

        if pr_coords.size(0) == 0 or gt_coords.size(0) == 0:
            return torch.tensor(float('inf'))

        def compute_min_dist(coords1, coords2):
            dist_matrix = torch.cdist(coords1.float(), coords2.float())
            return dist_matrix.min(dim=1)[0]

        pr_to_gt = compute_min_dist(pr_coords, gt_coords)
        gt_to_pr = compute_min_dist(gt_coords, pr_coords)

        hausdorff_pr_to_gt_95 = torch.quantile(pr_to_gt, percentile)
        hausdorff_gt_to_pr_95 = torch.quantile(gt_to_pr, percentile)

        hausdorff_dist_95 = torch.max(hausdorff_pr_to_gt_95, hausdorff_gt_to_pr_95)

        return hausdorff_dist_95

    B, C, H, W = pr.shape
    hausdorff_dists = torch.zeros(C)

    for c in range(C):
        hausdorff_dists[c] = compute_channel_distance(pr_[:, c, :, :], gt_[:, c, :, :])

    hausdorff_distance = torch.mean(hausdorff_dists)
    return hausdorff_distance.cpu().numpy()

def _dilate(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

def _erode(x: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=pad)

def _thin_boundary(mask01: torch.Tensor) -> torch.Tensor:

    er = _erode(mask01, 3)
    bd = (mask01 - er).clamp_min(0.0)
    return (bd > 0.5).float()

def boundary_iou(pr, gt, eps=1e-7, threshold=0.5, dilation_ratio=0.02):

    if isinstance(pr, list): pr = torch.tensor(np.array(pr))
    if isinstance(gt, list): gt = torch.tensor(np.array(gt))
    pr = pr.clone()
    gt = gt.clone()


    pr = (pr > threshold).float()
    gt = (gt > threshold).float()


    if pr.shape[-2:] != gt.shape[-2:]:
        pr = F.interpolate(pr, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pr = (pr > 0.5).float()


    pr_bd = _thin_boundary(pr)
    gt_bd = _thin_boundary(gt)


    H, W = gt.shape[-2:]
    diag = (H**2 + W**2) ** 0.5
    tol = max(1, int(round(dilation_ratio * diag)))
    if tol % 2 == 0:
        tol += 1
    if tol > 1:
        pr_bd = _dilate(pr_bd, tol)
        gt_bd = _dilate(gt_bd, tol)


    inter = (pr_bd * gt_bd).sum(dim=[1,2,3])
    union = (pr_bd + gt_bd - pr_bd * gt_bd).sum(dim=[1,2,3])

    biou = (inter + eps) / (union + eps)
    return biou.detach().cpu().numpy()

def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'hausdorff':
            metric_list.append(np.mean(hausdorff_distance_95(pred, label)))
        elif metric in ('biou', 'boundary_iou'): 
            metric_list.append(np.mean(boundary_iou(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric
