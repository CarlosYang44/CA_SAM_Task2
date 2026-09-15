# KPIs2024 preparation for CA-SAM

CA-SAM expects a parent data directory containing one folder per task:

```text
Med_datasets/
├── 56Nx/
│   ├── dataset.json
│   ├── image/
│   ├── label/
│   └── imask/
└── DN/
    ├── dataset.json
    ├── image/
    ├── label/
    └── imask/
```

The KPIs2024 Task 1 patch archives already contain the exact split reported by
CA-SAM after macOS `._*` ZIP metadata is ignored:

| Dataset | Training | Test |
| --- | ---: | ---: |
| 56Nx | 558 | 463 |
| DN | 724 | 391 |

The validation archive and Task 2 WSI TIFF files are retained as raw source
data but are not used by this CA-SAM reproduction split.

## Run on macOS

From the CA-SAM repository root, after installing `requirements.txt`:

```bash
python scripts/prepare_kpis2024_for_casam.py \
  --raw-root /Volumes/KINGSTON/KPIs2024 \
  --output-root /Volumes/KINGSTON/Med_datasets \
  --dry-run

python scripts/prepare_kpis2024_for_casam.py \
  --raw-root /Volumes/KINGSTON/KPIs2024 \
  --output-root /Volumes/KINGSTON/Med_datasets
```

The command is resumable: valid existing generated files are retained and
missing files are recreated. Use `--overwrite` only when intentionally
regenerating all selected output files.

## Run again on UCloud

If the raw KPIs2024 folder is mounted at `/mnt/ufs/KPIs2024`, use:

```bash
python scripts/prepare_kpis2024_for_casam.py \
  --raw-root /mnt/ufs/KPIs2024 \
  --output-root /mnt/ufs/Med_datasets \
  --dry-run

python scripts/prepare_kpis2024_for_casam.py \
  --raw-root /mnt/ufs/KPIs2024 \
  --output-root /mnt/ufs/Med_datasets
```

Then pass the generated root to CA-SAM:

```bash
python train_align_CL_VAE.py \
  --dataset_name 56Nx \
  --all_datasets 56Nx,DN \
  --data_dir /mnt/ufs/Med_datasets \
  --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --epochs 1 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --mask_num 1 \
  --router_type none
```

## Conversion details

- Source JPEG patches are extracted without recompression.
- Binary JPEG masks are thresholded at values greater than zero.
- Labels are stored as sparse CSR `.npz` files with logical shape
  `(1, 2048, 2048, 1)`, as required by `data_loader.py`.
- Training `imask` files are deterministic `int8` arrays with background `-1`
  and foreground `1`. The released loader requires these files even though its
  current training collate function uses ground-truth prompts.
- Raw ZIP and WSI files are never moved, renamed, or deleted.
