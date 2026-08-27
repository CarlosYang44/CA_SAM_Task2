# CA-SAM

Official implementation for **Continual Alignment for SAM: Rethinking Foundation Models for Medical Image Segmentation in Continual Learning**, accepted to **CVPR 2026 Findings**.

Paper page: https://cvpr.thecvf.com/virtual/2026/poster/41240

This repository contains the source code for the CA-SAM training and VAE-router evaluation pipeline. It does not include medical datasets, SAM checkpoints, trained adapters, trained VAEs, or patient data. Users should obtain datasets and checkpoints from their official sources and follow their licenses.

## Method Overview

CA-SAM freezes the SAM encoder, prompt encoder, and mask decoder, then inserts a lightweight task-specific Alignment Layer between the encoder and decoder. For continual learning, each task owns:

- one Alignment Layer;
- one TaskVAE router model;
- one task-specific confidence threshold `tau`.

At inference time, the frozen SAM encoder feature is pooled into a vector, all task VAEs compute ELBO scores, and the task with the lowest ELBO is selected. If the selected score is larger than the selected task threshold, the sample is treated as out-of-distribution and uses identity alignment, equivalent to falling back to the original frozen SAM feature.

## Repository Layout

```text
.
├── train_align_CL_VAE.py              # Train one task's Alignment Layer and VAE
├── eval_vae_router_load_adapter.py    # Evaluate saved adapters/VAEs with VAE routing
├── run_VAE.py                         # Sequential training launcher
├── run_eval_vae_router.py             # Sequential evaluation launcher
├── data_loader.py                     # Dataset reader
├── dataloaders/                       # Resize, normalization, prompt utilities
├── metrics.py                         # IoU, Dice, BIoU, Hausdorff metrics
├── utils.py                           # Losses, logging, prompt helpers
├── CL/                                # VAE router and adapter utilities
├── segment_anything/                  # SAM model code plus Alignment Layer modules
├── DATA_PREPARATION.md                # Expected dataset structure
├── requirements.txt                   # Minimal dependency list
└── requirements-full.txt              # Full environment snapshot from experiments
```

## Environment

Python 3.10 is recommended. Install PyTorch according to your CUDA version first, then install the remaining packages.

Example:

```bash
conda create -n casam python=3.10
conda activate casam

# Install a PyTorch build matching your CUDA driver from https://pytorch.org/get-started/locally/
# Example only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

`requirements-full.txt` is kept as an experiment environment snapshot. It may contain extra packages and CUDA runtime wheels from the original server, so it is not the recommended default installation file.

## Checkpoints

The reported experiments used the SAM ViT-B checkpoint:

```text
sam_vit_b_01ec64.pth
```

Download it from the official Segment Anything Model release and pass its path with `--sam_checkpoint`, or edit the launcher variables in `run_VAE.py` and `run_eval_vae_router.py`.

This repository does not distribute:

- SAM pretrained checkpoints;
- trained Alignment Layer checkpoints;
- trained TaskVAE checkpoints;
- task `tau.json` files.

When training is finished, the code writes task assets under the configured output directory:

```text
<work_dir>/checkpoints/Txx_<dataset>_cnn<num_cnn>/align_<method>_<num_cnn>.pth
<work_dir>/adapters_ckpt/<dataset>/align_<method>_<num_cnn>.pth
<work_dir>/vaes_ckpt/<dataset>/vae.pth
<work_dir>/vaes_ckpt/<dataset>/tau.json
```

## Data

Prepare datasets in the format described in `DATA_PREPARATION.md`, then pass the dataset root through `--data_dir`.

Each task directory must contain a `dataset.json` file with `training`, `test`, and `labels` fields. Training samples additionally need pseudo-mask files referenced by `imask`.

## Training

Single-task training example:

```bash
python train_align_CL_VAE.py \
  --dataset_name ACDC \
  --all_datasets ACDC,EBHI-SEG,56Nx,DN,Polyp,MSD_Prostate,MSD_Spleen,promise12,STS-2D \
  --data_dir /path/to/Med_datasets \
  --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --work_dir /path/to/output \
  --save_root /path/to/output/vae_router_cnn_cnn3 \
  --router_ckpt_dir /path/to/output/vae_router_cnn_cnn3/vaes_ckpt \
  --adapters_ckpt_dir /path/to/output/vae_router_cnn_cnn3/adapters_ckpt \
  --router_type vae \
  --method cnn \
  --num_cnn 3
```

Sequential training can be launched with:

```bash
python run_VAE.py
```

Before running, edit `WORK_ROOT`, `DATA_DIR`, `SAM_CKPT`, and `DATASETS` in the launcher.

## Evaluation

After training adapters, VAEs, and tau files:

```bash
python run_eval_vae_router.py
```

The evaluation script uses accumulated seen tasks. At stage `t`, only datasets from the task sequence up to `t` are visible to the router.

## Tau Calibration

For each task, CA-SAM estimates `tau` with K-fold held-out ELBO:

1. extract frozen SAM encoder features from the current task training set;
2. split features into `k_folds` folds, default `5`;
3. train a temporary VAE on `K-1` folds;
4. score the held-out fold with ELBO;
5. merge all held-out ELBO scores;
6. save `tau_p95`, `tau_p97`, `tau_p99`, mean, std, and max;
7. use `tau_p97` as `tau_suggested`.

Inference rule:

```text
s_t <= tau_t  -> use task Alignment Layer
s_t >  tau_t  -> use identity alignment fallback
```

## Citation

```bibtex
@InProceedings{Wang_2026_CVPR,
  author    = {Wang, Jiayi and Dai, Wei and Wang, Haoyu and Yang, Sihan and Bi, Haixia and Sun, Jian},
  title     = {Continual Alignment for SAM: Rethinking Foundation Models for Medical Image Segmentation in Continual Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026}
}
```

## License

This project is released under the Apache License 2.0. The bundled SAM-derived files retain their original copyright notices.
