# Data Preparation

This repository does not redistribute medical datasets. Please download each dataset from its official source and make sure your use follows the corresponding license, privacy policy, and citation requirements.

## Expected Root

Pass the parent directory of all task folders as `--data_dir`.

Example:

```text
/path/to/Med_datasets/
├── ACDC/
│   ├── dataset.json
│   ├── image/
│   ├── label/
│   └── imask/
├── EBHI-SEG/
│   ├── dataset.json
│   ├── image/
│   ├── label/
│   └── imask/
└── ...
```

If your extracted archive creates an extra nested folder, use the inner folder as the dataset directory. For example, if the files are under `Med_datasets/ACDC/ACDC/dataset.json`, either move the inner `ACDC` directory up or set the root so that the loader can resolve:

```text
<data_dir>/ACDC/dataset.json
```

## Dataset Folder Format

Each task folder must contain:

```text
<dataset_name>/
├── dataset.json
├── image/
├── label/
└── imask/
```

Optional folders such as `GT_img`, `GT_per_class`, and `ori_label` may be kept for visualization or preprocessing, but the training and evaluation loader only requires the files referenced by `dataset.json`.

## dataset.json Format

Each `dataset.json` should contain:

```json
{
  "name": "ACDC",
  "description": "optional description",
  "dimension": "2D",
  "modality": {
    "0": "MR"
  },
  "labels": {
    "0": "background",
    "1": "class_1",
    "2": "class_2"
  },
  "numTraining": 1632,
  "training": [
    {
      "image": "image/x/sample_001.png",
      "label": "label/x___sample_001.(3, 256, 216, 1).npz",
      "imask": "imask/x___sample_001.npy"
    }
  ],
  "test": [
    {
      "image": "image/x/sample_101.png",
      "label": "label/x___sample_101.(3, 256, 216, 1).npz"
    }
  ]
}
```

The paths are relative to the dataset folder.

## Image Files

Images are read with PIL and converted to NumPy arrays. PNG files are recommended:

```text
image/<subfolder>/<sample>.png
```

The loader resizes or pads images according to `--image_size`, whose default is `1024`.

## Label Files

Labels are sparse `.npz` files loaded with `scipy.sparse.load_npz`.

The loader reconstructs the label array shape from the filename suffix:

```text
label/x___patient001_frame01_1.(3, 256, 216, 1).npz
```

This means the filename must include the original dense label shape before `.npz`:

```text
(num_classes_without_background, height, width, channels)
```

During loading, the sparse matrix is converted back with:

```python
label_array = sparse.load_npz(label_path).toarray().reshape(shape_from_filename)
```

For multi-class datasets, the label tensor should be organized per foreground class. The `background` entry exists in `labels`, but the loader removes it before assigning target names.

## Pseudo-Mask Files

Training samples require the `imask` field:

```text
imask/<subfolder>/<sample>.npy
```

The pseudo mask is loaded with `np.load` and used to sample prompts during Alignment Layer training. Test samples do not require `imask`.

## Prompt Generation

The loader generates:

- point prompts from foreground masks;
- box prompts from foreground masks;
- pseudo prompts from pseudo masks during training.

The training script randomly uses point or box prompts. Evaluation uses box prompts.

## Datasets Reported in the Paper

The paper reports nine continual medical segmentation tasks:

| Dataset | Train | Test | Modality | Target |
| --- | ---: | ---: | --- | --- |
| ACDC | 1632 | 177 | MR | left ventricle, myocardium, right ventricle |
| EBHI-SEG | 1701 | 487 | pathology | colon cancer affected area |
| 56Nx | 558 | 463 | pathology | glomerulus |
| DN | 724 | 391 | pathology | glomerulus |
| Polyp | 804 | 196 | RGB endoscopy | polyp |
| MSD_Prostate | 419 | 53 | MR T2 | peripheral zone, transition zone |
| MSD_Spleen | 876 | 146 | CT | spleen |
| promise12 | 712 | 66 | MR | prostate |
| STS-2D | 1700 | 70 | X-ray | teeth |

Use dataset folder names that match the values passed in `--all_datasets` and the `DATASETS` list in the launcher.

## Minimal Sanity Check

Before long training, verify that each dataset can be read:

```bash
python train_align_CL_VAE.py \
  --dataset_name ACDC \
  --all_datasets ACDC \
  --data_dir /path/to/Med_datasets \
  --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --epochs 1 \
  --train_batch_size 1 \
  --eval_batch_size 1 \
  --mask_num 1 \
  --router_type none
```

This check only verifies loader/model connectivity. It is not expected to reproduce paper-level accuracy.
