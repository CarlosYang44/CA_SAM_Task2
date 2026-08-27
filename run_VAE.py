#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from typing import List, Optional


CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
SCRIPT_PATH         = os.path.join(os.path.dirname(__file__), "train_align_CL_VAE.py")
WORK_ROOT           = os.environ.get("CASAM_WORK_ROOT", "./outputs/train")
SAM_CKPT            = os.environ.get("CASAM_SAM_CKPT", "./pretrain_model/sam_vit_b_01ec64.pth")
DATA_DIR            = os.environ.get("CASAM_DATA_DIR", "./Med_datasets")

DEVICE      = "cuda:0"
MODEL_TYPE  = "vit_b"
METHOD      = "cnn"
EPOCHS      = 24
LR          = 1e-4
TRAIN_BS    = 6
EVAL_BS     = 1
IMAGE_SIZE  = 1024
MASK_NUM    = 5
MULTIMASK   = False


ROUTER_TYPE       = "vae"
ROUTER_THRESHOLD  = 1.0
ZERO_ADAPTER_MODE = "identity"
VAE_FEAT          = "attn_pool"
VAE_IN_DIM        = 256
VAE_LATENT        = 64
VAE_BETA          = 16.5
VAE_EPOCHS        = 10
VAE_LR            = 5e-4


CNN_LAYERS_LIST = [3]


DATASETS = ["ACDC","EBHI-SEG"]


IDX_LIST: Optional[List[int]] = None


EXTRA = ""

CONTINUE_ON_FAIL = False

DRY_RUN = False


START_AT = 0



def work_dir_for(method: str, num_cnn: int) -> str:

    return os.path.join(WORK_ROOT, f"vae_router_{method}_cnn{num_cnn}")

def run_name_for(index: int, ds: str, num_cnn: int) -> str:
    return f"T{index+1:02d}_{ds}_cnn{num_cnn}"

def adapters_root(work_dir: str) -> str:
    return os.path.join(work_dir, "adapters_ckpt")

def routers_root(work_dir: str) -> str:
    return os.path.join(work_dir, "vaes_ckpt")

def adapter_path_for_dataset(work_dir: str, ds: str, method: str, num_cnn: int) -> str:
    return os.path.join(adapters_root(work_dir), ds, f"align_{method}_{num_cnn}.pth")

def vae_path_for_dataset(work_dir: str, ds: str) -> str:
    return os.path.join(routers_root(work_dir), ds, "vae.pth")


def main():
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    datasets = DATASETS if IDX_LIST is None else [DATASETS[i] for i in IDX_LIST]

    print("Training will run in this order:")
    for i, ds in enumerate(datasets):
        print(f"  {i}: {ds}")
    print()

    os.makedirs(WORK_ROOT, exist_ok=True)

    for num_cnn in CNN_LAYERS_LIST:
        work_dir = work_dir_for(METHOD, num_cnn)
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(adapters_root(work_dir), exist_ok=True)
        os.makedirs(routers_root(work_dir), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "checkpoints"), exist_ok=True)


        if START_AT > 0:
            prev_ds = datasets[START_AT - 1]
            prev_adapter = adapter_path_for_dataset(work_dir, prev_ds, METHOD, num_cnn)
            if not os.path.isfile(prev_adapter):
                raise FileNotFoundError(f"Resuming from a later task requires the previous task adapter, but it was not found:\n  {prev_adapter}")


        datasets_run = datasets[START_AT:]
        for i, ds in enumerate(datasets_run, start=START_AT):
            run_name = run_name_for(i, ds, num_cnn)


            common = [
                sys.executable, SCRIPT_PATH,

                "--work_dir", work_dir,
                "--run_name", run_name,
                "--dataset_name", ds,
                "--device", DEVICE,
                "--model_type", MODEL_TYPE,
                "--sam_checkpoint", SAM_CKPT,
                "--data_dir", DATA_DIR,
                "--method", METHOD,
                "--num_cnn", str(num_cnn),
                "--epochs", str(EPOCHS),
                "--lr", str(LR),
                "--train_batch_size", str(TRAIN_BS),
                "--eval_batch_size", str(EVAL_BS),
                "--image_size", str(IMAGE_SIZE),
                "--mask_num", str(MASK_NUM),
                "--all_datasets", ",".join(datasets),
            ]

            if MULTIMASK:
                common.append("--multimask")


            common += [
                "--adapters_ckpt_dir", adapters_root(work_dir),
                "--router_ckpt_dir",   routers_root(work_dir),
            ]


            if ROUTER_TYPE == "vae":
                common += [
                    "--router_type", "vae",
                    "--router_threshold", str(ROUTER_THRESHOLD),
                    "--zero_adapter_mode", ZERO_ADAPTER_MODE,
                    "--vae_feat", VAE_FEAT,
                    "--vae_in_dim", str(VAE_IN_DIM),
                    "--vae_latent_dim", str(VAE_LATENT),
                    "--vae_beta", str(VAE_BETA),
                    "--vae_epochs", str(VAE_EPOCHS),
                    "--vae_lr", str(VAE_LR),
                ]

            if EXTRA.strip():
                common.extend(EXTRA.strip().split())

            print(f"=== Start task: {run_name} (dataset={ds}, cnn_layers={num_cnn}) ===")
            print(" ".join(common))

            rc = subprocess.call(common) if not DRY_RUN else 0
            if rc != 0:
                msg = f"Task failed: {run_name}"
                if CONTINUE_ON_FAIL:
                    print("!!! " + msg + ", skipping and continuing", file=sys.stderr)
                    continue
                else:
                    raise SystemExit(rc)


            cur_adapter = adapter_path_for_dataset(work_dir, ds, METHOD, num_cnn)
            cur_vae     = vae_path_for_dataset(work_dir, ds)

            missing = []
            if not os.path.isfile(cur_adapter):
                missing.append(cur_adapter)
            if ROUTER_TYPE == "vae" and not os.path.isfile(cur_vae):
                missing.append(cur_vae)

            if missing:
                msg = "Training finished but the following artifacts were not found:\n  " + "\n  ".join(missing)
                if CONTINUE_ON_FAIL:
                    print("!!! " + msg + ", skipping and continuing", file=sys.stderr)
                else:
                    raise FileNotFoundError(msg)

            print(f"+++ Finished task: {run_name}")
            print(f"    Adapter: {cur_adapter}")
            if ROUTER_TYPE == "vae":
                print(f"    VAE    : {cur_vae}")


if __name__ == "__main__":
    main()
