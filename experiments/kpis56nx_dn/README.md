# CA-SAM 56Nx → DN reproduction

This directory runs the released CA-SAM implementation on the two-task
KPIs2024 sequence `56Nx → DN`. It is an experiment wrapper, not a new method.
It does not modify the CA-SAM model implementation.

## Scope

- T1: train the task-specific CNN-3 Alignment Layer and VAE router for 56Nx.
- T2: train the task-specific CNN-3 Alignment Layer and VAE router for DN.
- Evaluate T1 with the 56Nx router/adaptor set.
- Evaluate T2 with both 56Nx and DN visible to the router.
- Save stage-wise IoU and BIoU matrices, logs, checkpoints, tau statistics,
  environment details, and the exact Git revision.

This reproduces the CA-SAM task-specific-adapter baseline. It does not measure
shared-parameter catastrophic forgetting and does not implement SR²-LoRA.

## UCloud setup

Create the Python environment and install the CA-SAM requirements. Download the
official SAM ViT-B checkpoint separately. If `/mnt/ufs/Med_datasets` has not
been generated yet, first follow `docs/KPIS2024_PREPARATION.md` and run
`scripts/prepare_kpis2024_for_casam.py`; the same preparation command is safe to
rerun on UCloud. Then set paths:

```bash
cd /path/to/CA_SAM_Task2
cp experiments/kpis56nx_dn/ucloud.env.example /tmp/casam-ucloud.env
nano /tmp/casam-ucloud.env
source /tmp/casam-ucloud.env
```

The default formal configuration follows the repository launcher:

```text
method=cnn, num_cnn=3, epochs=24, lr=1e-4
train_batch_size=6, eval_batch_size=1
image_size=1024, mask_num=5
router=VAE, feature=attn_pool, in_dim=256, latent_dim=64
vae_beta=16.5, vae_epochs=10, vae_lr=5e-4
tau=5-fold held-out p97
```

## Execution order

Run each gate separately so a failed smoke test cannot silently start a long
training run:

```bash
python experiments/kpis56nx_dn/run_experiment.py preflight
python experiments/kpis56nx_dn/run_experiment.py smoke
python experiments/kpis56nx_dn/run_experiment.py train
python experiments/kpis56nx_dn/run_experiment.py eval
```

Or, after the preflight has already been inspected:

```bash
python experiments/kpis56nx_dn/run_experiment.py all
```

The runner repeats preflight validation before each operation unless
`--skip-preflight` is explicitly supplied.

## GPU-memory overrides

Start with the official batch size. If CUDA reports out-of-memory, reduce only
the training batch size first and record the override:

```bash
python experiments/kpis56nx_dn/run_experiment.py train \
  --train-batch-size 2
```

Do not silently change `image-size`, `mask-num`, the task order, or router
configuration for a claimed paper reproduction.

## Resume behavior

Training skips a task only when its adapter, VAE, and `tau.json` all exist. A
partially completed task is rerun. If an existing run was created with different
settings, the runner stops instead of relabelling old artifacts; choose a new
run root or use `--force` intentionally. Every evaluation uses a new timestamped
directory by default, avoiding duplicate CSV rows.

## Results

Outputs are written below `$CASAM_RUN_ROOT`:

```text
preflight.json
smoke/
train/vae_router_cnn_cnn3/
├── adapters_ckpt/{56Nx,DN}/
├── vaes_ckpt/{56Nx,DN}/
├── checkpoints/
├── logs/
└── driver_logs/
eval/<timestamp>/
├── cl_metrics/
└── driver_logs/
```

Render the two metric matrices as one Markdown table:

```bash
python experiments/kpis56nx_dn/summarize_results.py \
  "$CASAM_RUN_ROOT/eval/<timestamp>"
```

Keep the generated outputs outside Git. The repository `.gitignore` already
excludes the default `outputs/`, checkpoints, `.pth`, `.npy`, and `.npz` files.
