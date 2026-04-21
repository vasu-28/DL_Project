# Missing-Modality Brain Tumor Segmentation

This project trains a 3D PyTorch model for BraTS brain tumor segmentation when one or more MRI modalities are missing. The repository includes:

- `brats_data_pipeline.py` for dataset inspection, preprocessing, and verification
- `train.py` for model sanity checks, training, checkpoint loading, and evaluation
- sample output files such as `eval_results.json` and `patient_visualization.png`

The raw BraTS dataset is not included in this repository. You will need access to the dataset before running preprocessing or training.

## Requirements

- Python 3.10 or newer
- A working PyTorch installation
- Enough disk space for the BraTS dataset and processed `.npz` files
- A GPU is strongly recommended for training and evaluation

Install PyTorch using the build that matches your machine, then install the remaining Python packages:

```bash
pip install numpy nibabel matplotlib tqdm tensorboard
```

If PyTorch is not installed yet, install it first and then verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Repository Layout

```text
CS671_Project/
├── brats_data_pipeline.py
├── train.py
├── eval_results.json
├── patient_visualization.png
├── processed/              # generated after preprocessing
├── checkpoints/            # generated during training
├── runs/                   # TensorBoard logs
└── README.md
```

## 1. Clone And Set Up The Environment

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd CS671_Project

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install numpy nibabel matplotlib tqdm tensorboard
```

Install PyTorch separately if it is not already available in your environment.

## 2. Prepare The BraTS Dataset

Edit the configuration block at the top of `brats_data_pipeline.py` and update these paths:

- `TRAIN_DIR`
- `ADDITIONAL_TRAIN_DIR`
- `VAL_DIR`

If you do not have an additional training folder, set:

```python
ADDITIONAL_TRAIN_DIR = None
```

The script expects each patient folder to contain files named like:

```text
BraTS-GLI-xxxxx-xxx/
├── BraTS-GLI-xxxxx-xxx-t1n.nii.gz
├── BraTS-GLI-xxxxx-xxx-t1c.nii.gz
├── BraTS-GLI-xxxxx-xxx-t2w.nii.gz
├── BraTS-GLI-xxxxx-xxx-t2f.nii.gz
└── BraTS-GLI-xxxxx-xxx-seg.nii.gz
```

## 3. Explore And Preprocess The Data

Inspect the dataset layout first:

```bash
python brats_data_pipeline.py --step explore
```

Optionally generate a sample visualization:

```bash
python brats_data_pipeline.py --step visualize
```

Preprocess raw `.nii.gz` files into compressed `.npz` files:

```bash
python brats_data_pipeline.py --step preprocess
```

Verify that the processed dataset loads correctly:

```bash
python brats_data_pipeline.py --step verify
```

By default, preprocessing writes to:

- `processed/train`
- `processed/val`

## 4. Run A Sanity Check

Before training, confirm the model builds and the forward pass works:

```bash
python train.py --stage test
```

This is the safest first command to run on a new machine.

## 5. Train The Model

Start training with the processed training set:

```bash
python train.py --stage train --epochs 200 --batch_size 1
```

Useful arguments:

- `--epochs` sets the number of training epochs
- `--batch_size` controls batch size
- `--lr` sets the learning rate
- `--base_features` changes model width
- `--dropout` changes dropout
- `--grad_accum` controls gradient accumulation
- `--checkpoint` resumes training from a saved checkpoint

Example with a custom configuration:

```bash
python train.py --stage train \
  --epochs 100 \
  --batch_size 2 \
  --lr 2e-4 \
  --grad_accum 2
```

To resume from a checkpoint:

```bash
python train.py --stage train --checkpoint checkpoints/best_model.pth
```

Training outputs:

- `checkpoints/best_model.pth`
- `checkpoints/ep50.pth`, `ep100.pth`, and so on
- TensorBoard logs under `runs/train`

To monitor training:

```bash
tensorboard --logdir runs
```

## 6. Evaluate The Model

Evaluate the trained model across all 15 modality-presence combinations:

```bash
python train.py --stage eval --checkpoint checkpoints/best_model.pth
```

If `--checkpoint` is omitted, the script uses:

```text
checkpoints/best_model.pth
```

Evaluation writes results to:

```text
eval_results.json
```

## Full Workflow

If you want the shortest complete path from raw data to results:

```bash
python brats_data_pipeline.py --step explore
python brats_data_pipeline.py --step preprocess
python brats_data_pipeline.py --step verify
python train.py --stage test
python train.py --stage train --epochs 200 --batch_size 1
python train.py --stage eval --checkpoint checkpoints/best_model.pth
```

## Important Notes

- The official validation set does not contain segmentation labels, so `train.py` creates its own validation split from `processed/train`.
- Training and preprocessing are storage-heavy. Make sure you have enough disk space before running them.
- Full training is intended for GPU use. CPU execution is mainly practical for the `--stage test` sanity check.
- Large generated folders such as `processed/`, `checkpoints/`, and `runs/` are usually not committed to Git.

## Troubleshooting

- `FileNotFoundError: No .npz files in processed/train`
  Run preprocessing first with `python brats_data_pipeline.py --step preprocess`.

- `No checkpoint: checkpoints/best_model.pth`
  Train the model first or pass a valid checkpoint path with `--checkpoint`.

- `CUDA out of memory`
  Reduce `--batch_size`, lower `--base_features`, or increase `--grad_accum`.

- Missing BraTS files during preprocessing
  Re-check the folder names and the modality suffixes in `brats_data_pipeline.py`.

  
