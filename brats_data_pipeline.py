"""
================================================================================
BraTS DATA PIPELINE — FIXED WORKING VERSION
================================================================================
SETUP:
    pip install torch nibabel matplotlib numpy tqdm

USAGE:
    1. Update the 3 paths and MODALITY_SUFFIXES in the CONFIG section below
    2. Run:  python brats_data_pipeline.py --step explore
    3. Run:  python brats_data_pipeline.py --step preprocess
    4. Run:  python brats_data_pipeline.py --step verify
================================================================================
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works without display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

import torch
from torch.utils.data import Dataset, DataLoader


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG — UPDATE THESE 3 PATHS AND SUFFIXES TO MATCH YOUR DATA          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

# ──── Paths ────
TRAIN_DIR = "/path/to/BraTS2024/training"                       # ← CHANGE
ADDITIONAL_TRAIN_DIR = "/path/to/BraTS2024/additional_training"  # ← CHANGE (set None if absent)
VAL_DIR = "/path/to/BraTS2024/validation"                       # ← CHANGE

# ──── Output dirs for processed .npz files ────
PROCESSED_TRAIN_DIR = "processed/train"
PROCESSED_VAL_DIR = "processed/val"

# ──── Modality suffixes (run --step explore first to check yours) ────
MODALITY_SUFFIXES = {
    "t1":    "t1n",    # T1 native
    "t1ce":  "t1c",    # T1 contrast-enhanced
    "t2":    "t2w",    # T2 weighted
    "flair": "t2f",    # T2-FLAIR
}
SEG_SUFFIX = "seg"

# ──── Processing ────
TARGET_SHAPE = (128, 128, 128)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  EXPLORE                                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def explore_dataset(root_dir):
    print("=" * 70)
    print(f"EXPLORING: {root_dir}")
    print("=" * 70)

    if not os.path.isdir(root_dir):
        print(f"  ERROR: Directory does not exist: {root_dir}")
        return

    patient_dirs = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    print(f"  Patient folders found: {len(patient_dirs)}")

    if not patient_dirs:
        return

    first = os.path.join(root_dir, patient_dirs[0])
    files = sorted(os.listdir(first))
    print(f"\n  Files in '{patient_dirs[0]}':")
    for f in files:
        size_mb = os.path.getsize(os.path.join(first, f)) / (1024 * 1024)
        print(f"    {f}  ({size_mb:.1f} MB)")

    print(f"\n  Detected suffixes:")
    for f in files:
        if f.endswith('.nii.gz'):
            base = f.replace('.nii.gz', '')
            suffix = base.split('-')[-1]
            label = {
                't1n': 'T1 native', 't1c': 'T1 contrast', 't2w': 'T2 weighted',
                't2f': 'T2-FLAIR', 'seg': 'Segmentation',
                't1': 'T1 (old)', 't1ce': 'T1ce (old)', 't2': 'T2 (old)', 'flair': 'FLAIR (old)',
            }.get(suffix, 'Unknown')
            print(f"    '{suffix}' -> {label}")
    print()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PREPROCESSING                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def z_score_normalize(volume):
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std < 1e-8:
        return volume - mean
    normalized = (volume - mean) / std
    normalized[~mask] = 0
    return normalized


def pad_or_crop_to_shape(volume, target_shape):
    result = np.zeros(target_shape, dtype=volume.dtype)
    slices_src = []
    slices_dst = []
    for i in range(3):
        if volume.shape[i] > target_shape[i]:
            start = (volume.shape[i] - target_shape[i]) // 2
            slices_src.append(slice(start, start + target_shape[i]))
            slices_dst.append(slice(0, target_shape[i]))
        else:
            start = (target_shape[i] - volume.shape[i]) // 2
            slices_src.append(slice(0, volume.shape[i]))
            slices_dst.append(slice(start, start + volume.shape[i]))
    result[slices_dst[0], slices_dst[1], slices_dst[2]] = \
        volume[slices_src[0], slices_src[1], slices_src[2]]
    return result


def load_and_preprocess_patient(patient_dir, target_shape=TARGET_SHAPE):
    patient_name = os.path.basename(patient_dir)
    mod_names = ["t1", "t1ce", "t2", "flair"]

    raw = {}
    for mod_name in mod_names:
        suffix = MODALITY_SUFFIXES[mod_name]
        filepath = os.path.join(patient_dir, f"{patient_name}-{suffix}.nii.gz")
        if os.path.exists(filepath):
            raw[mod_name] = nib.load(filepath).get_fdata().astype(np.float32)
        else:
            raise FileNotFoundError(f"Missing: {filepath}")

    seg_path = os.path.join(patient_dir, f"{patient_name}-{SEG_SUFFIX}.nii.gz")
    seg = None
    if os.path.exists(seg_path):
        seg = nib.load(seg_path).get_fdata().astype(np.int64)

    # Brain bounding box
    combined = np.max(np.stack(list(raw.values())), axis=0)
    nonzero = np.where(combined > 0)
    if len(nonzero[0]) == 0:
        volume = np.zeros((4,) + target_shape, dtype=np.float32)
        if seg is not None:
            seg = np.zeros(target_shape, dtype=np.int64)
        return volume, seg

    margin = 1
    mins = [max(0, np.min(ax) - margin) for ax in nonzero]
    maxs = [min(combined.shape[i], np.max(nonzero[i]) + margin + 1) for i in range(3)]

    processed = []
    for mod_name in mod_names:
        vol = z_score_normalize(raw[mod_name])
        vol = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        vol = pad_or_crop_to_shape(vol, target_shape)
        processed.append(vol)

    volume = np.stack(processed, axis=0).astype(np.float32)

    if seg is not None:
        seg = seg[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        seg = pad_or_crop_to_shape(seg, target_shape).astype(np.int64)

    return volume, seg


def preprocess_and_save_all(root_dir, output_dir, target_shape=TARGET_SHAPE):
    if root_dir is None or not os.path.isdir(root_dir):
        print(f"  Skipping (not found): {root_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    patient_dirs = sorted([
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    print(f"\nPreprocessing {len(patient_dirs)} patients -> {output_dir}")
    failed = []
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        patient_name = os.path.basename(patient_dir)
        output_path = os.path.join(output_dir, f"{patient_name}.npz")
        if os.path.exists(output_path):
            continue
        try:
            volume, seg = load_and_preprocess_patient(patient_dir, target_shape)
            if seg is not None:
                np.savez_compressed(output_path, volume=volume, seg=seg)
            else:
                np.savez_compressed(output_path, volume=volume)
        except Exception as e:
            print(f"\n  FAILED: {patient_name} — {e}")
            failed.append(patient_name)

    print(f"Done: {len(patient_dirs) - len(failed)}/{len(patient_dirs)} succeeded")
    if failed:
        print(f"Failed: {failed[:10]}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PYTORCH DATASET                                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class BraTSDataset(Dataset):
    """
    Returns dict with:
        'volume':       (4, H, W, D) float32 — input (zeroed if missing)
        'original':     (4, H, W, D) float32 — full original (recon target)
        'mask':         (4,) float32          — 1=present, 0=missing
        'seg':          (3, H, W, D) float32  — WT, TC, ET binary masks
        'has_seg':      int (0 or 1)
        'patient_name': str
    """

    def __init__(self, data_dir, mode='train', missing_strategy='random',
                 fixed_missing=None, min_present=1, augment=False):
        self.data_dir = data_dir
        self.mode = mode
        self.missing_strategy = missing_strategy
        self.fixed_missing = fixed_missing or []
        self.min_present = min_present
        self.augment = augment and (mode == 'train')

        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        print(f"BraTSDataset: {len(self.file_list)} patients | mode={mode} | "
              f"missing={missing_strategy} | augment={augment}")

    def __len__(self):
        return len(self.file_list)

    def _simulate_missing(self, volume):
        num_mods = 4
        mask = np.ones(num_mods, dtype=np.float32)

        if self.missing_strategy == 'none':
            return volume, mask
        elif self.missing_strategy == 'fixed':
            for idx in self.fixed_missing:
                volume[idx] = 0.0
                mask[idx] = 0.0
            return volume, mask
        elif self.missing_strategy == 'random':
            max_drop = num_mods - self.min_present
            num_drop = random.randint(0, max_drop)
            if num_drop > 0:
                drop_indices = random.sample(range(num_mods), num_drop)
                for idx in drop_indices:
                    volume[idx] = 0.0
                    mask[idx] = 0.0
            return volume, mask
        return volume, mask

    def _compute_seg_targets(self, seg):
        wt = (seg > 0).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        return np.stack([wt, tc, et], axis=0)

    def _augment_volume(self, volume, seg):
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                volume = np.flip(volume, axis=axis).copy()
                if seg is not None:
                    seg = np.flip(seg, axis=axis - 1).copy()
        return volume, seg

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        volume = data['volume'].copy()           # (4, H, W, D)
        seg = data['seg'] if 'seg' in data else None
        patient_name = os.path.basename(self.file_list[idx]).replace('.npz', '')

        # Augment first
        if self.augment:
            volume, seg = self._augment_volume(volume, seg)

        # Save original BEFORE masking — reconstruction target
        original = volume.copy()

        # Simulate missing modalities
        volume, mod_mask = self._simulate_missing(volume)

        # Seg targets
        has_seg = 1 if seg is not None else 0
        if seg is not None:
            seg_targets = self._compute_seg_targets(seg)
        else:
            seg_targets = np.zeros((3,) + volume.shape[1:], dtype=np.float32)

        return {
            'volume': torch.from_numpy(volume),
            'original': torch.from_numpy(original),
            'mask': torch.from_numpy(mod_mask),
            'seg': torch.from_numpy(seg_targets),
            'has_seg': has_seg,
            'patient_name': patient_name,
        }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DATALOADER                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def create_dataloaders(train_npz_dir, val_npz_dir, batch_size=2, num_workers=2):
    train_dataset = BraTSDataset(
        data_dir=train_npz_dir, mode='train',
        missing_strategy='random', min_present=1, augment=True,
    )
    val_dataset = BraTSDataset(
        data_dir=val_npz_dir, mode='val',
        missing_strategy='none', augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\nDataLoaders ready:")
    print(f"  Train: {len(train_dataset)} patients, {len(train_loader)} batches (bs={batch_size})")
    print(f"  Val:   {len(val_dataset)} patients, {len(val_loader)} batches (bs=1)")
    return train_loader, val_loader


def get_all_missing_combinations():
    names = ["T1", "T1ce", "T2", "FLAIR"]
    combos = []
    for num in range(1, 5):
        for present in combinations(range(4), num):
            missing = [i for i in range(4) if i not in present]
            
            # Create a readable description of what is present (e.g., "T1 + FLAIR")
            desc = " + ".join([names[i] for i in present])
            
            combos.append({
                'present': list(present), 
                'missing': missing,
                'description': desc  # <-- Added the missing key here!
            })
    return combos

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS Data Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['explore', 'visualize', 'preprocess', 'verify'])
    args = parser.parse_args()

    if args.step == 'explore':
        explore_dataset(TRAIN_DIR)
        if ADDITIONAL_TRAIN_DIR:
            explore_dataset(ADDITIONAL_TRAIN_DIR)
        explore_dataset(VAL_DIR)

    elif args.step == 'visualize':
        patient_dirs = sorted(glob.glob(os.path.join(TRAIN_DIR, "*")))
        if patient_dirs:
            patient_name = os.path.basename(patient_dirs[0])
            mod_names = ["t1", "t1ce", "t2", "flair"]
            vols = {}
            for mod in mod_names:
                suffix = MODALITY_SUFFIXES[mod]
                fp = os.path.join(patient_dirs[0], f"{patient_name}-{suffix}.nii.gz")
                if os.path.exists(fp):
                    vols[mod] = nib.load(fp).get_fdata()
            seg_path = os.path.join(patient_dirs[0], f"{patient_name}-{SEG_SUFFIX}.nii.gz")
            seg = nib.load(seg_path).get_fdata() if os.path.exists(seg_path) else None
            first = list(vols.values())[0]
            slice_idx = int(np.argmax((seg > 0).sum(axis=(0, 1)))) if seg is not None else first.shape[2] // 2

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            for i, mod in enumerate(mod_names):
                if mod in vols:
                    axes[i].imshow(vols[mod][:, :, slice_idx].T, cmap='gray', origin='lower')
                axes[i].set_title(mod.upper())
                axes[i].axis('off')
            if seg is not None and "flair" in vols:
                axes[4].imshow(vols["flair"][:, :, slice_idx].T, cmap='gray', origin='lower')
                s = seg[:, :, slice_idx].T
                overlay = np.zeros((*s.shape, 4))
                overlay[s == 1] = [1, 0, 0, 0.5]
                overlay[s == 2] = [0, 1, 0, 0.4]
                overlay[s == 3] = [1, 1, 0, 0.6]
                axes[4].imshow(overlay, origin='lower')
            axes[4].set_title("SEG")
            axes[4].axis('off')
            plt.tight_layout()
            plt.savefig("patient_visualization.png", dpi=150)
            plt.close()
            print("Saved: patient_visualization.png")

    elif args.step == 'preprocess':
        preprocess_and_save_all(TRAIN_DIR, PROCESSED_TRAIN_DIR, TARGET_SHAPE)
        if ADDITIONAL_TRAIN_DIR:
            preprocess_and_save_all(ADDITIONAL_TRAIN_DIR, PROCESSED_TRAIN_DIR, TARGET_SHAPE)
        preprocess_and_save_all(VAL_DIR, PROCESSED_VAL_DIR, TARGET_SHAPE)

    elif args.step == 'verify':
        print("Verifying dataset...\n")
        train_loader, val_loader = create_dataloaders(
            PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR,
            batch_size=2, num_workers=0,
        )
        batch = next(iter(train_loader))
        print(f"\nBatch contents:")
        print(f"  volume:   {batch['volume'].shape} {batch['volume'].dtype}")
        print(f"  original: {batch['original'].shape} {batch['original'].dtype}")
        print(f"  mask:     {batch['mask']}")
        print(f"  seg:      {batch['seg'].shape}")
        print(f"  has_seg:  {batch['has_seg']}")
        print(f"  names:    {batch['patient_name']}")

        mask = batch['mask'][0]
        vol = batch['volume'][0]
        orig = batch['original'][0]
        for i, name in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
            v = vol[i].abs().sum().item()
            o = orig[i].abs().sum().item()
            status = "PRESENT" if mask[i] > 0 else "MISSING"
            print(f"    {name}: masked={v:.0f}, original={o:.0f} -> {status}")

        print("\nVerification PASSED!")
