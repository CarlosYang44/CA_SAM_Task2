# Shared Alignment Layer pilot: 56Nx → DN

This experiment measures forgetting and plasticity when **one Alignment Layer** is
updated sequentially on `56Nx` and then `DN`. It deliberately does not use the
official CA-SAM task-specific adapter bank or VAE router.

The frozen SAM backbone is reconstructed from the same base checkpoint for every
training/evaluation process. Only the CNN-3 Alignment Layer is trainable.

## Experiment design

| Run | Initialization | Training | Saved model | Evaluation |
| --- | --- | --- | --- | --- |
| A | fresh SAM + fresh AL | 56Nx, 24 epochs | `M_56Nx.pth` | 56Nx (`56Nx_before`) |
| B | fresh frozen SAM + **load `M_56Nx` into the AL** | DN, 24 epochs | `M_56Nx_DN.pth` | DN (`DN_sequential`) and 56Nx (`56Nx_after_DN`) |
| C | fresh SAM + fresh AL | DN, 24 epochs | `M_DN_only.pth` | DN (`DN_only`) |

Run B starts from the exact Alignment Layer state produced by Run A. The runner
records the source checkpoint SHA-256 beside `M_56Nx_DN.pth` so that this link is
auditable. The optimizer is newly constructed for each training stage; no replay,
router, or task-specific adapter selection is used.

For each metric (`iou`, `dice`, and `biou`) the summary reports:

```text
forgetting     = 56Nx_before - 56Nx_after_DN
plasticity_gap = DN_only - DN_sequential
```

IoU is also exposed as the primary top-level result in `summary.json`.

## UCloud setup

From the repository root:

```bash
cp experiments/kpis56nx_dn/ucloud.env.example /tmp/casam-shared-al.env
vi /tmp/casam-shared-al.env
source /tmp/casam-shared-al.env
```

`CASAM_DATA_DIR` defaults to `/mnt/ufs/Med_datasets`. The SAM checkpoint can be
set with `CASAM_SAM_CKPT` or passed explicitly with `--sam-checkpoint`.

## Validate, smoke-test, and run

Run the strict data/checkpoint/environment validation first:

```bash
python experiments/kpis56nx_dn/run_experiment.py preflight
```

The smoke test executes all three training stages, verifies the Run A → Run B
checkpoint hand-off, evaluates four small test subsets, and writes a summary:

```bash
python experiments/kpis56nx_dn/run_experiment.py smoke
```

Run the complete 24-epoch pilot and all evaluations:

```bash
python experiments/kpis56nx_dn/run_experiment.py all
```

The stages can also be resumed separately. Existing completed artifacts are
skipped unless `--force` is supplied:

```bash
python experiments/kpis56nx_dn/run_experiment.py train
python experiments/kpis56nx_dn/run_experiment.py eval
python experiments/kpis56nx_dn/run_experiment.py summarize
```

Common overrides:

```bash
python experiments/kpis56nx_dn/run_experiment.py all \
  --data-dir /mnt/ufs/Med_datasets \
  --sam-checkpoint /mnt/ufs/checkpoints/sam_vit_b_01ec64.pth \
  --run-root /mnt/ufs/CA_SAM_runs/56nx_dn_shared \
  --cuda-visible-devices 0 \
  --device cuda:0
```

## Outputs

With the example environment, the complete run is stored under:

```text
/mnt/ufs/CA_SAM_runs/56nx_dn_shared/shared_al_cnn3/
├── experiment_config.json
├── checkpoints/
│   ├── M_56Nx.pth
│   ├── M_56Nx.json
│   ├── M_56Nx_DN.pth
│   ├── M_56Nx_DN.json
│   ├── M_DN_only.pth
│   └── M_DN_only.json
├── stages/
│   ├── run_a_56Nx/
│   ├── run_b_56Nx_to_DN/
│   └── run_c_DN_only/
├── driver_logs/
└── results/
    ├── raw/
    │   ├── 56Nx_before.json
    │   ├── 56Nx_after_DN.json
    │   ├── DN_sequential.json
    │   └── DN_only.json
    ├── summary.json
    ├── summary.csv
    └── summary.md
```

Smoke-test artifacts are isolated under `smoke_shared_al/` and cannot be mistaken
for full-run results.
