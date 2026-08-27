#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from typing import List, Optional


CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
SCRIPT_PATH         = os.path.join(os.path.dirname(__file__), "eval_vae_router_load_adapter.py")
WORK_ROOT           = os.environ.get("CASAM_EVAL_ROOT", "./outputs/eval")
SAM_CKPT            = os.environ.get("CASAM_SAM_CKPT", "./pretrain_model/sam_vit_b_01ec64.pth")
DATA_DIR            = os.environ.get("CASAM_DATA_DIR", "./Med_datasets")


DEVICE      = "cuda:0"
MODEL_TYPE  = "vit_b"
METHOD      = "cnn"
CNN_LAYERS_LIST = [3]


TRAIN_BS    = 6
EVAL_BS     = 1
IMAGE_SIZE  = 1024
MASK_NUM    = 5
MULTIMASK   = False


VAE_FEAT        = "attn_pool"
VAE_IN_DIM      = 256
VAE_LATENT_DIM  = 64
VAE_EPOCHS      = 10
VAE_LR          = 5e-4
VAE_BETA        = 16.5
ROUTER_TAU      = 1.0
ZERO_ADAPTER    = "identity"
CLS_TOKEN_LEN   = 1


ADAPTERS_CKPT_ROOT = os.environ.get(
    "CASAM_ADAPTERS_CKPT_ROOT",
    "./outputs/train/vae_router_cnn_cnn3/adapters_ckpt",
)
VAES_CKPT_ROOT = os.environ.get(
    "CASAM_VAES_CKPT_ROOT",
    "./outputs/train/vae_router_cnn_cnn3/vaes_ckpt",
)


DATASETS = ["ACDC","EBHI-SEG"]

IDX_LIST: Optional[List[int]] = None


EXTRA = "--skip_train_vae"
CONTINUE_ON_FAIL = False
DRY_RUN = False
START_AT = 0



def work_dir_for(method: str, num_cnn: int) -> str:
    return os.path.join(WORK_ROOT, f"vae_router_{method}_cnn{num_cnn}")

def main():
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    datasets = DATASETS if IDX_LIST is None else [DATASETS[i] for i in IDX_LIST]

    print("Evaluation will run in this order: load adapter -> load trained VAE -> route and evaluate seen tasks")
    for i, ds in enumerate(datasets):
        print(f"  {i}: {ds}")
    print()

    os.makedirs(WORK_ROOT, exist_ok=True)

    vae_in_dim_local = VAE_IN_DIM
    if VAE_FEAT == "flatten":
        vae_in_dim_local = 512
    elif VAE_FEAT == "cls":
        vae_in_dim_local = 768

    for num_cnn in CNN_LAYERS_LIST:
        work_dir = work_dir_for(METHOD, num_cnn)
        os.makedirs(work_dir, exist_ok=True)

        cl_csv_dir = os.path.join(work_dir, "cl_metrics")
        os.makedirs(cl_csv_dir, exist_ok=True)
        iou_csv  = os.path.join(cl_csv_dir, f"vae_{METHOD}_cnn{num_cnn}_iou.csv")
        biou_csv = os.path.join(cl_csv_dir, f"vae_{METHOD}_cnn{num_cnn}_biou.csv")

        datasets_run = datasets[START_AT:]
        for i, ds in enumerate(datasets_run, start=START_AT):
            print(f"=== Start task: T{i+1:02d}_{ds}_cnn{num_cnn} (dataset={ds}) ===")

            common = [
                sys.executable, SCRIPT_PATH,
                "--work_dir", work_dir,
                "--dataset_name", ds,
                "--all_datasets", ",".join(datasets),
                "--device", DEVICE,
                "--model_type", MODEL_TYPE,
                "--sam_checkpoint", SAM_CKPT,
                "--data_dir", DATA_DIR,
                "--method", METHOD,
                "--num_cnn", str(num_cnn),
                "--train_batch_size", str(TRAIN_BS),
                "--eval_batch_size", str(EVAL_BS),
                "--image_size", str(IMAGE_SIZE),
                "--mask_num", str(MASK_NUM),
                "--adapters_ckpt_dir", ADAPTERS_CKPT_ROOT,
                "--router_ckpt_dir",  VAES_CKPT_ROOT,
                "--vae_feat", VAE_FEAT,
                "--vae_in_dim", str(vae_in_dim_local),
                "--vae_latent_dim", str(VAE_LATENT_DIM),
                "--vae_beta", str(VAE_BETA),
                "--vae_epochs", str(VAE_EPOCHS),
                "--vae_lr", str(VAE_LR),
                "--router_threshold", str(ROUTER_TAU),
                "--zero_adapter_mode", ZERO_ADAPTER,
                "--cl_matrix_csv", iou_csv,
                "--cl_matrix_biou_csv", biou_csv,
            ]

            if VAE_FEAT == "cls":
                common += ["--use_moda_encoder"]
                common += ["--cls_token_len", str(CLS_TOKEN_LEN)]

            if EXTRA.strip():
                common.extend(EXTRA.strip().split())

            print(" ".join(common))
            rc = subprocess.call(common) if not DRY_RUN else 0

            if rc != 0:
                msg = f"Task failed: T{i+1:02d}_{ds}_cnn{num_cnn}"
                if CONTINUE_ON_FAIL:
                    print("!!! " + msg + ", skipping and continuing", file=sys.stderr)
                    continue
                else:
                    raise SystemExit(rc)

            print(f"+++ Finished task: T{i+1:02d}_{ds}_cnn{num_cnn}")

    print("\nAll done. CSV files were appended with stage-wise evaluation results.")

if __name__ == "__main__":
    main()
