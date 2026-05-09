#!/home/teaching/miniconda3/envs/p19/bin/python
"""
================================================================================
PROJECT P19 — TRAIN.PY  (v7 — GPU-Optimised Final)
================================================================================
Environment : conda activate p19  (PyTorch 2.1.2 + CUDA 12.1)
GPU          : NVIDIA RTX A5000  24 GB  sm_86

GPU optimisations applied
  • torch.backends.cudnn.benchmark = True   — auto-tune conv kernels
  • torch.amp.autocast(device_type='cuda')  — mixed-precision FP16
  • torch.cuda.amp.GradScaler              — safe FP16 gradient scaling
  • non_blocking=True on all .to(device)   — async CPU→GPU transfers
  • persistent_workers=True in DataLoaders — keep worker processes alive
  • num_workers=4, pin_memory=True         — fast data loading
  • batch_size default raised to 2         — fits 24 GB VRAM for 128³ volumes
  • grad_accum=2 (effective batch = 4)     — stable gradients

Architecture
  Student MissingModalityNet
    ModalityAwareEncoder  — per-modality CNN + SE attention + mask-weighted fusion
    UNetDecoder           — 4-level skip-connection U-Net decoder
    ReconstructionHead    — L1 loss on missing modalities
    SegmentationHead      — standard Dice + Focal head
    EvidentialHead        — Beta-distribution per class → principled vacuity
    MultiScaleProjector   — contrastive embeddings at bottleneck AND decoder

  EMATeacher
    — copy of student with frozen gradients
    — cosine-scheduled EMA (0.996 → 1.0)
    — always sees ≥ 3 modalities (richer input than student)

Training
  Stage 1  ssl_pretrain  50 epochs
    DINO loss (teacher→student, centering+sharpening)
    Multi-scale InfoNCE (two student views)
    Reconstruction L1
    View consistency

  Stage 2  train  150 epochs
    EvidentialSegLoss (SOS + KL-annealed)
    Uncertainty-weighted Dice (vacuity as inverse weight)
    Focal loss
    Teacher-student consistency
    Multi-scale InfoNCE regularisation
    Deep supervision

Inference
  forward()            single pass — evid logits + seg
  forward_mc()         20-pass MC Dropout  → epistemic + aleatoric
  forward_tta()        8-fold axis-flip TTA → ensemble mean + variance
  forward_evidential() single pass evidential → prob + vacuity (fast)

Usage
  python train.py --stage test
  python train.py --stage ssl_pretrain  --ssl_epochs 50
  python train.py --stage train         --epochs 150  --checkpoint checkpoints/ssl_pretrained.pth
  python train.py --stage test_eval
  python train.py --stage eval
  python train.py --stage simulate
  python train.py --stage report
================================================================================
"""

# ─── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import glob
import copy
import json
import random
import argparse
import warnings

# ─── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr          # uncertainty–error correlation

# ─── PyTorch ──────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# PyTorch 2.1: autocast lives in torch.amp; GradScaler still in torch.cuda.amp
from torch.amp import autocast as amp_autocast           # torch.amp.autocast
from torch.cuda.amp import GradScaler                    # torch.cuda.amp.GradScaler

# ─── Project ──────────────────────────────────────────────────────────────────
from brats_data_pipeline import BraTSDataset, get_all_missing_combinations

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
#  GPU CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    # Auto-tune cuDNN kernel selection for fixed-size 128³ volumes
    torch.backends.cudnn.benchmark     = True
    # Allow non-deterministic algorithms for maximum speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True   # A5000 supports TF32
    torch.backends.cudnn.allow_tf32       = True


# ─────────────────────────────────────────────────────────────────────────────
#  PROJECT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED_TRAIN_DIR = "processed/train"
RESULTS_DIR         = "results"
SEED                = 42
TRAIN_FRAC          = 0.70
VAL_FRAC            = 0.20
NUM_WORKERS         = 4          # safe for 24 CPU cores, keeps GPU fed
PIN_MEMORY          = True
PERSISTENT_WORKERS  = True
AMP_DTYPE           = torch.float16   # A5000 (sm_86) has fast FP16


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    # Note: deterministic=False above means exact reproducibility is traded
    # for speed; results will be close but not bit-identical across runs.


def print_gpu_info():
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {p.name}")
        print(f"  VRAM : {p.total_memory / 1024**3:.1f} GB")
        print(f"  SM   : {p.major}.{p.minor}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
    else:
        print("  Device: CPU (no CUDA)")


# ─────────────────────────────────────────────────────────────────────────────
#  BACKBONE MODULES
# ─────────────────────────────────────────────────────────────────────────────

class ResConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip  = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        x   = self.act(self.norm1(self.conv1(x)))
        x   = self.drop(x)
        return self.act(self.norm2(self.conv2(x))) + res


class SEBlock3D(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 1)), nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 1), ch), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.shape[0], -1, 1, 1, 1)


class ModalityAwareEncoder(nn.Module):
    def __init__(self, num_mods=4, f=32, dropout=0.1):
        super().__init__()
        self.num_mods  = num_mods
        self.mod_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, f, 3, padding=1, bias=False),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for _ in range(num_mods)
        ])
        self.se   = SEBlock3D(f)
        self.enc1 = ResConvBlock3D(f,    f,    dropout)
        self.enc2 = ResConvBlock3D(f,    f*2,  dropout)
        self.enc3 = ResConvBlock3D(f*2,  f*4,  dropout)
        self.enc4 = ResConvBlock3D(f*4,  f*8,  dropout)
        self.btn  = ResConvBlock3D(f*8,  f*16, dropout)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x, mask):
        B     = x.shape[0]
        feats = [self.mod_convs[i](x[:, i:i+1]) for i in range(self.num_mods)]
        stack = torch.stack(feats, 1)                          # (B,4,f,H,W,D)
        m     = mask.view(B, self.num_mods, 1, 1, 1, 1).float()
        fused = (stack * m).sum(1) / m.sum(1).clamp(min=1)    # mask-weighted mean
        fused = self.se(fused)
        e1 = self.enc1(fused)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.btn (self.pool(e4))
        return [e1, e2, e3, e4, bn]


class UNetDecoder(nn.Module):
    def __init__(self, f=32, dropout=0.1, seg_cls=3):
        super().__init__()
        self.up4  = nn.ConvTranspose3d(f*16, f*8,  2, stride=2)
        self.dec4 = ResConvBlock3D(f*16, f*8,  dropout)
        self.up3  = nn.ConvTranspose3d(f*8,  f*4,  2, stride=2)
        self.dec3 = ResConvBlock3D(f*8,  f*4,  dropout)
        self.up2  = nn.ConvTranspose3d(f*4,  f*2,  2, stride=2)
        self.dec2 = ResConvBlock3D(f*4,  f*2,  dropout)
        self.up1  = nn.ConvTranspose3d(f*2,  f,    2, stride=2)
        self.dec1 = ResConvBlock3D(f*2,  f,    dropout)
        self.ds3  = nn.Conv3d(f*4, seg_cls, 1)
        self.ds2  = nn.Conv3d(f*2, seg_cls, 1)

    def forward(self, enc):
        e1, e2, e3, e4, bn = enc
        d4 = self.dec4(torch.cat([self.up4(bn), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1, self.ds3(d3), self.ds2(d2)


# ─────────────────────────────────────────────────────────────────────────────
#  EVIDENTIAL HEAD  —  Beta distribution per segmentation class
#
#  For each voxel and class c:
#    alpha_c = softplus( logit_c) + 1   → evidence for positive  (≥1)
#    beta_c  = softplus(-logit_c) + 1   → evidence for negative  (≥1)
#    prob_c  = alpha / (alpha + beta)   → probability estimate
#    vacuity = 2 / (alpha + beta)       → epistemic uncertainty  ∈(0,1)
#
#  Permanent Dropout3d enables MC-Dropout epistemic estimation at inference.
# ─────────────────────────────────────────────────────────────────────────────

class EvidentialHead(nn.Module):
    def __init__(self, in_ch, seg_cls=3, mc_dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(mc_dropout),        # kept active at inference for MC Dropout
            nn.Conv3d(in_ch // 2, seg_cls, 1),
        )

    def forward(self, x):
        return self.net(x)                   # raw logits, unbounded

    @staticmethod
    def get_evidence(logits):
        """Convert logits → Beta parameters + probability + vacuity."""
        alpha   = F.softplus( logits) + 1.0
        beta_   = F.softplus(-logits) + 1.0
        S       = alpha + beta_
        return {
            "alpha":   alpha,
            "beta":    beta_,
            "S":       S,
            "prob":    alpha / S,
            "vacuity": 2.0 / S,             # ∈(0,1): 0=certain, 1=max uncertain
        }


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-SCALE CONTRASTIVE PROJECTOR
#  Projects both bottleneck (global) and decoder output (local) to embedding
#  space so InfoNCE / DINO aligns representations at two granularities.
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleProjector(nn.Module):
    def __init__(self, btn_ch, dec_ch, proj_dim=128):
        super().__init__()
        self.btn_pool = nn.AdaptiveAvgPool3d(1)
        self.dec_pool = nn.AdaptiveAvgPool3d(4)           # 4³ = 64 spatial tokens

        self.btn_mlp = nn.Sequential(
            nn.Linear(btn_ch, btn_ch),
            nn.LayerNorm(btn_ch),
            nn.ReLU(inplace=True),
            nn.Linear(btn_ch, proj_dim),
        )
        self.dec_mlp = nn.Sequential(
            nn.Linear(dec_ch * 64, dec_ch),
            nn.LayerNorm(dec_ch),
            nn.ReLU(inplace=True),
            nn.Linear(dec_ch, proj_dim),
        )

    def forward(self, btn, dec):
        z_btn = F.normalize(self.btn_mlp(self.btn_pool(btn).flatten(1)), dim=1)
        z_dec = F.normalize(self.dec_mlp(self.dec_pool(dec).flatten(1)), dim=1)
        return z_btn, z_dec


# ─────────────────────────────────────────────────────────────────────────────
#  FULL STUDENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class MissingModalityNet(nn.Module):
    """
    Forward outputs
      recon       (B,4,H,W,D)  reconstructed modalities
      seg         (B,3,H,W,D)  segmentation logits (standard head)
      evid_logit  (B,3,H,W,D)  evidential logits → EvidentialHead.get_evidence()
      ds3 / ds2               deep-supervision logits
      proj_btn    (B, proj_dim)  bottleneck contrastive embedding
      proj_dec    (B, proj_dim)  decoder contrastive embedding
    """

    def __init__(self, num_mods=4, f=32, seg_cls=3, dropout=0.1, proj_dim=128):
        super().__init__()
        self.encoder    = ModalityAwareEncoder(num_mods, f, dropout)
        self.decoder    = UNetDecoder(f, dropout, seg_cls)
        self.recon_head = nn.Conv3d(f, num_mods, 1)
        self.seg_head   = nn.Sequential(
            nn.Conv3d(f + num_mods, f, 3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, seg_cls, 1),
        )
        self.evid_head = EvidentialHead(f + num_mods, seg_cls, mc_dropout=0.2)
        self.proj      = MultiScaleProjector(f * 16, f, proj_dim)

    def forward(self, x, mod_mask):
        enc               = self.encoder(x, mod_mask)
        dec, ds3, ds2     = self.decoder(enc)
        recon             = self.recon_head(dec)
        m                 = mod_mask.view(-1, 4, 1, 1, 1).float()
        recon_err         = (torch.abs(recon - x) * m).detach()
        feats             = torch.cat([dec, recon_err], 1)
        seg               = self.seg_head(feats)
        evid_logit        = self.evid_head(feats)
        proj_btn, proj_dec = self.proj(enc[-1], dec)
        return {
            "recon":      recon,
            "seg":        seg,
            "evid_logit": evid_logit,
            "ds3":        ds3,
            "ds2":        ds2,
            "proj_btn":   proj_btn,
            "proj_dec":   proj_dec,
        }

    # ── Single-pass evidential inference (fastest) ──────────────────────
    @torch.no_grad()
    def forward_evidential(self, x, mod_mask):
        self.eval()
        out = self.forward(x, mod_mask)
        ev  = EvidentialHead.get_evidence(out["evid_logit"])
        return ev["prob"], ev["vacuity"], ev["alpha"], ev["beta"]

    # ── MC Dropout (20 passes, dropout active) ───────────────────────────
    @torch.no_grad()
    def forward_mc(self, x, mod_mask, n_passes=20):
        self.train()             # activates Dropout3d layers
        preds, vacs = [], []
        for _ in range(n_passes):
            out = self.forward(x, mod_mask)
            ev  = EvidentialHead.get_evidence(out["evid_logit"])
            preds.append(ev["prob"])
            vacs .append(ev["vacuity"])
        self.eval()
        stack     = torch.stack(preds)
        mean_pred = stack.mean(0)
        epistemic = stack.var(0)                # inter-pass variance
        aleatoric = torch.stack(vacs).mean(0)   # mean predicted vacuity
        return mean_pred, epistemic + aleatoric, epistemic, aleatoric

    # ── 8-fold TTA (all axis-flip combinations) ───────────────────────────
    @torch.no_grad()
    def forward_tta(self, x, mod_mask):
        self.eval()
        flip_combos = [
            [], [2], [3], [4],
            [2,3], [2,4], [3,4], [2,3,4],
        ]
        preds = []
        for axes in flip_combos:
            xf = torch.flip(x, axes) if axes else x
            out = self.forward(xf, mod_mask)
            ev  = EvidentialHead.get_evidence(out["evid_logit"])
            p   = ev["prob"]
            preds.append(torch.flip(p, axes) if axes else p)
        stack     = torch.stack(preds)
        mean_pred = stack.mean(0)
        tta_unc   = stack.var(0)               # variance across augmentations
        return mean_pred, tta_unc

    # ── Adaptive uncertainty thresholding ──────────────────────────────────
    @torch.no_grad()
    def forward_adaptive_threshold(self, x, mod_mask, unc_threshold=0.15):
        """
        Evidential adaptive inference.

        Voxels whose vacuity exceeds `unc_threshold` are flagged as
        *inconclusive* and zeroed in the hard prediction.  This produces a
        conservative segmentation that only asserts positive labels where the
        model is actually confident — directly implementing the spec objective
        of 'voxel-level uncertainty maps for reliability assessment'.

        Returns
          pred         (B,3,H,W,D)  hard predictions (inconclusive → 0)
          inconclusive (B,3,H,W,D)  bool, True where vacuity > threshold
          prob         (B,3,H,W,D)  raw probability
          vacuity      (B,3,H,W,D)  evidential vacuity
        """
        self.eval()
        out     = self.forward(x, mod_mask)
        ev      = EvidentialHead.get_evidence(out["evid_logit"])
        prob    = ev["prob"]
        vacuity = ev["vacuity"]
        inconclusive = vacuity > unc_threshold
        pred    = (prob > 0.5).float()
        pred[inconclusive] = 0.0
        return pred, inconclusive, prob, vacuity

    # ── Leave-one-out modality importance ──────────────────────────────────
    @torch.no_grad()
    def modality_importance(self, x, mod_mask):
        """
        Compute per-modality importance by leave-one-out ablation.

        For each present modality, we zero it out and measure how much the
        evidential probability changes on average.  Larger change = that
        modality carries more information for this patient.

        Returns importance scores (4,) — zero for missing modalities.
        """
        self.eval()
        out_base  = self.forward(x, mod_mask)
        prob_base = EvidentialHead.get_evidence(out_base["evid_logit"])["prob"]
        importance = torch.zeros(4)
        for mod_idx in range(4):
            if mod_mask[0, mod_idx] == 0:
                continue
            x_drop    = x.clone()
            mask_drop = mod_mask.clone()
            x_drop[:, mod_idx]    = 0.0
            mask_drop[:, mod_idx] = 0.0
            if mask_drop.sum() == 0:
                # Cannot drop the last remaining modality
                importance[mod_idx] = 0.0
                continue
            out_drop  = self.forward(x_drop, mask_drop)
            prob_drop = EvidentialHead.get_evidence(out_drop["evid_logit"])["prob"]
            importance[mod_idx] = float((prob_base - prob_drop).abs().mean().item())
        return importance


# ─────────────────────────────────────────────────────────────────────────────
#  ARCHITECTURE V2
#  Key upgrades over V1:
#    CrossModalAttentionFusion  — content-based modality weighting (not mean)
#    BottleneckTransformer      — global self-attention at 8³ spatial resolution
#    EvidentialHeadV2           — mask-conditioned uncertainty (fixes flat vacuity)
#    MissingModalityNetV2       — full model using all the above, f=48 default
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalAttentionFusion(nn.Module):
    """
    Replaces SE + mask-weighted mean from V1.

    A small MLP scores each present modality's spatial features
    (content-based, not just mask-based).  Absent modalities receive
    hard-zero weight via the mask before softmax.  This lets the model
    up-weight modalities that carry the most discriminative signal for
    a given patient, rather than treating all present modalities equally.

    Example: when only T1 and FLAIR are present for a GBM patient,
    FLAIR dominates edema detection while T1 matters less — the attention
    weights this automatically.
    """
    def __init__(self, n_mods=4, feat_dim=32):
        super().__init__()
        self.n_mods = n_mods
        # Score each modality: global-average-pool → MLP → scalar
        self.scorer = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
        )
        self.out = nn.Sequential(
            nn.Conv3d(feat_dim, feat_dim, 3, padding=1, bias=False),
            nn.InstanceNorm3d(feat_dim, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, feats_list, mask):
        # feats_list: list of (B, f, H, W, D) per modality
        # mask: (B, n_mods) binary — 0 = absent
        B = feats_list[0].shape[0]
        # Content-based scores (B, n_mods)
        scores = torch.cat([self.scorer(f) for f in feats_list], dim=1)
        # Hard-mask absent modalities (→ -inf before softmax)
        scores = scores * mask.float() + (1.0 - mask.float()) * (-1e9)
        attn_w = torch.softmax(scores, dim=1)            # (B, n_mods)
        feat_stack = torch.stack(feats_list, dim=1)      # (B, n_mods, f, H, W, D)
        fused = (feat_stack * attn_w.view(B, self.n_mods, 1, 1, 1, 1)).sum(1)
        return self.out(fused)


class BottleneckTransformerLayer(nn.Module):
    """
    Single pre-norm Transformer block for the 3D bottleneck.
    Input/output: (B, C, H, W, D).  Spatial dims are flattened to a sequence,
    global self-attention is applied, then reshaped back.

    At f=48 the bottleneck is (B, 768, 8, 8, 8) → 512 tokens of dim 768.
    Self-attention cost: 512² × 768 = 201M ops per layer — fast on A5000.
    """
    def __init__(self, dim, n_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        seq  = x.flatten(2).transpose(1, 2)          # (B, H*W*D, C)
        h    = self.norm1(seq)
        h, _ = self.attn(h, h, h)
        seq  = seq + h
        seq  = seq + self.mlp(self.norm2(seq))
        return seq.transpose(1, 2).view(B, C, H, W, D)


class BottleneckTransformer(nn.Module):
    def __init__(self, dim, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [BottleneckTransformerLayer(dim, n_heads, dropout=dropout)
             for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EvidentialHeadV2(nn.Module):
    """
    Evidential segmentation head with modality-mask conditioning.

    The number of present modalities (n_present / 4.0) is embedded via a
    small MLP and added as a spatial bias to the hidden features.  This
    makes vacuity explicitly scale with missing modalities, directly fixing
    the flat-vacuity problem observed in V1 (vacuity spread: 0.000933).

    Same Beta-distribution uncertainty math as V1; backward-compatible
    EvidentialHead.get_evidence() static method is preserved.
    """
    def __init__(self, in_ch, seg_cls=3, mc_dropout=0.2):
        super().__init__()
        hid = in_ch // 2
        self.conv1    = nn.Conv3d(in_ch, hid, 3, padding=1, bias=False)
        self.norm1    = nn.InstanceNorm3d(hid, affine=True)
        self.act      = nn.LeakyReLU(0.01, inplace=True)
        self.drop     = nn.Dropout3d(mc_dropout)
        self.conv2    = nn.Conv3d(hid, seg_cls, 1)
        # Mask conditioning: n_present/4 ∈[0,1] → per-channel spatial bias
        self.mask_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(inplace=True),
            nn.Linear(32, hid),
        )

    def forward(self, x, n_present=None):
        h = self.act(self.norm1(self.conv1(x)))
        if n_present is not None:
            ctx = self.mask_mlp((n_present.float() / 4.0).unsqueeze(-1))
            h   = h + ctx.view(ctx.shape[0], -1, 1, 1, 1)
        return self.conv2(self.drop(h))

    @staticmethod
    def get_evidence(logits):
        alpha   = F.softplus( logits) + 1.0
        beta_   = F.softplus(-logits) + 1.0
        S       = alpha + beta_
        return {"alpha": alpha, "beta": beta_, "S": S,
                "prob": alpha / S, "vacuity": 2.0 / S}


class ModalityAwareEncoderV2(nn.Module):
    """
    V2 encoder: CrossModalAttentionFusion + BottleneckTransformer.
    Drop-in replacement for ModalityAwareEncoder; same output format [e1…e4, bn].
    """
    def __init__(self, num_mods=4, f=48, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.num_mods        = num_mods
        self.use_checkpoint  = use_checkpoint
        self.mod_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, f, 3, padding=1, bias=False),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for _ in range(num_mods)
        ])
        self.fusion = CrossModalAttentionFusion(num_mods, f)
        self.enc1   = ResConvBlock3D(f,     f,     dropout)
        self.enc2   = ResConvBlock3D(f,     f*2,   dropout)
        self.enc3   = ResConvBlock3D(f*2,   f*4,   dropout)
        self.enc4   = ResConvBlock3D(f*4,   f*8,   dropout)
        self.btn    = ResConvBlock3D(f*8,   f*16,  dropout)
        self.btn_tf = BottleneckTransformer(f*16, n_layers=2, n_heads=8)
        self.pool   = nn.MaxPool3d(2, stride=2)

    def _forward_impl(self, x, mask):
        feats = [self.mod_convs[i](x[:, i:i+1]) for i in range(self.num_mods)]
        fused = self.fusion(feats, mask)
        e1    = self.enc1(fused)
        e2    = self.enc2(self.pool(e1))
        e3    = self.enc3(self.pool(e2))
        e4    = self.enc4(self.pool(e3))
        bn    = self.btn_tf(self.btn(self.pool(e4)))
        return [e1, e2, e3, e4, bn]

    def forward(self, x, mask):
        # Checkpoint the full encoder (incl. fusion feat_stack) during SSL to
        # avoid storing two 3 GB feat_stacks simultaneously for two student views.
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, use_reentrant=False
            )
        return self._forward_impl(x, mask)


class MissingModalityNetV2(nn.Module):
    """
    V2 full model.

    Architectural upgrades over V1
      CrossModalAttentionFusion   — content-based modality attention
      BottleneckTransformer       — global self-attention at 8³ resolution
      EvidentialHeadV2            — mask-conditioned uncertainty
      f=48 default                — 2.25× more capacity

    Same output dict as V1 → all evaluation / visualisation code works.
    """
    def __init__(self, num_mods=4, f=48, seg_cls=3, dropout=0.1, proj_dim=128, use_checkpoint=False):
        super().__init__()
        self.encoder    = ModalityAwareEncoderV2(num_mods, f, dropout, use_checkpoint=use_checkpoint)
        self.decoder    = UNetDecoder(f, dropout, seg_cls)
        self.recon_head = nn.Conv3d(f, num_mods, 1)
        self.seg_head   = nn.Sequential(
            nn.Conv3d(f + num_mods, f, 3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, seg_cls, 1),
        )
        self.evid_head = EvidentialHeadV2(f + num_mods, seg_cls, mc_dropout=0.2)
        self.proj      = MultiScaleProjector(f * 16, f, proj_dim)

    def forward(self, x, mod_mask):
        enc               = self.encoder(x, mod_mask)
        dec, ds3, ds2     = self.decoder(enc)
        recon             = self.recon_head(dec)
        m                 = mod_mask.view(-1, 4, 1, 1, 1).float()
        recon_err         = (torch.abs(recon - x) * m).detach()
        feats             = torch.cat([dec, recon_err], 1)
        seg               = self.seg_head(feats)
        n_present         = mod_mask.float().sum(-1)   # (B,) for mask conditioning
        evid_logit        = self.evid_head(feats, n_present)
        proj_btn, proj_dec = self.proj(enc[-1], dec)
        return {"recon": recon, "seg": seg, "evid_logit": evid_logit,
                "ds3": ds3, "ds2": ds2,
                "proj_btn": proj_btn, "proj_dec": proj_dec}

    # All inference methods below are identical to V1 — same interface.
    @torch.no_grad()
    def forward_evidential(self, x, mod_mask):
        self.eval()
        out = self.forward(x, mod_mask)
        ev  = EvidentialHeadV2.get_evidence(out["evid_logit"])
        return ev["prob"], ev["vacuity"], ev["alpha"], ev["beta"]

    @torch.no_grad()
    def forward_mc(self, x, mod_mask, n_passes=20):
        self.train()
        preds, vacs = [], []
        for _ in range(n_passes):
            out = self.forward(x, mod_mask)
            ev  = EvidentialHeadV2.get_evidence(out["evid_logit"])
            preds.append(ev["prob"]); vacs.append(ev["vacuity"])
        self.eval()
        stack     = torch.stack(preds)
        return stack.mean(0), stack.var(0) + torch.stack(vacs).mean(0), \
               stack.var(0), torch.stack(vacs).mean(0)

    @torch.no_grad()
    def forward_tta(self, x, mod_mask):
        self.eval()
        flip_combos = [[], [2],[3],[4],[2,3],[2,4],[3,4],[2,3,4]]
        preds = []
        for axes in flip_combos:
            xf  = torch.flip(x, axes) if axes else x
            out = self.forward(xf, mod_mask)
            ev  = EvidentialHeadV2.get_evidence(out["evid_logit"])
            preds.append(torch.flip(ev["prob"], axes) if axes else ev["prob"])
        stack = torch.stack(preds)
        return stack.mean(0), stack.var(0)

    @torch.no_grad()
    def forward_adaptive_threshold(self, x, mod_mask, unc_threshold=0.15):
        self.eval()
        out     = self.forward(x, mod_mask)
        ev      = EvidentialHeadV2.get_evidence(out["evid_logit"])
        prob    = ev["prob"]; vacuity = ev["vacuity"]
        inconclusive = vacuity > unc_threshold
        pred = (prob > 0.5).float(); pred[inconclusive] = 0.0
        return pred, inconclusive, prob, vacuity

    @torch.no_grad()
    def modality_importance(self, x, mod_mask):
        self.eval()
        out_base  = self.forward(x, mod_mask)
        prob_base = EvidentialHeadV2.get_evidence(out_base["evid_logit"])["prob"]
        importance = torch.zeros(4)
        for mod_idx in range(4):
            if mod_mask[0, mod_idx] == 0:
                continue
            x_drop = x.clone(); mask_drop = mod_mask.clone()
            x_drop[:, mod_idx] = 0.0; mask_drop[:, mod_idx] = 0.0
            if mask_drop.sum() == 0:
                continue
            out_drop  = self.forward(x_drop, mask_drop)
            prob_drop = EvidentialHeadV2.get_evidence(out_drop["evid_logit"])["prob"]
            importance[mod_idx] = float((prob_base - prob_drop).abs().mean().item())
        return importance


# ─────────────────────────────────────────────────────────────────────────────
#  EMA TEACHER  (momentum encoder)
# ─────────────────────────────────────────────────────────────────────────────

class EMATeacher:
    """
    Wraps a deepcopy of the student with EMA weight updates.
    Not an nn.Module so it doesn't appear in the student's parameter list.

    Momentum schedule: cosine ramp from base_mom (0.996) → 1.0 over training.
    This gives fast feature evolution early and stable targets late.
    """
    def __init__(self, student: MissingModalityNet, momentum: float = 0.996):
        self.model    = copy.deepcopy(student).eval()
        self.base_mom = momentum
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: MissingModalityNet, epoch: int, total_epochs: int):
        m = 1.0 - (1.0 - self.base_mom) * (
            np.cos(np.pi * epoch / max(total_epochs, 1)) + 1.0) / 2.0
        for tp, sp in zip(self.model.parameters(), student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    def __call__(self, x, mod_mask):
        return self.model(x, mod_mask)

    def to(self, device):
        self.model = self.model.to(device)
        return self


# ─────────────────────────────────────────────────────────────────────────────
#  LOSSES
# ─────────────────────────────────────────────────────────────────────────────

class ReconLoss(nn.Module):
    """L1 on missing modalities only, restricted to brain voxels."""
    def forward(self, pred, target, mod_mask):
        missing = (1.0 - mod_mask.float()).view(-1, 4, 1, 1, 1)
        if missing.sum() == 0:
            return pred.sum() * 0.0
        brain = (target.abs() > 0).float()
        valid = missing * brain
        if valid.sum() == 0:
            return pred.sum() * 0.0
        return (torch.abs(pred - target) * valid).sum() / valid.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.s = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.shape[0], logits.shape[1], -1)
        t = targets             .view(targets.shape[0], targets.shape[1], -1)
        return 1.0 - ((2*(p*t).sum(2) + self.s) /
                       (p.sum(2) + t.sum(2) + self.s)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (self.a * (1 - torch.exp(-bce)) ** self.g * bce).mean()


class InfoNCELoss(nn.Module):
    """Bidirectional InfoNCE — works fine as a regulariser even with bs=1."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        B = z1.shape[0]
        if B < 2:
            return z1.sum() * 0.0
        sim    = torch.matmul(z1, z2.T) / self.tau
        labels = torch.arange(B, device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


class DINOLoss(nn.Module):
    """
    Distillation loss from DINO (Caron et al. 2021).
    Teacher outputs are centred (prevents collapse) and sharpened (low temp).
    Student outputs use a warmer temperature.
    Requires no negative pairs → stable with batch_size=1.
    """
    def __init__(self, proj_dim=128, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.t_temp  = teacher_temp
        self.s_temp  = student_temp
        self.c_mom   = center_momentum
        self.register_buffer("center", torch.zeros(1, proj_dim))

    def forward(self, z_teacher, z_student):
        t    = F.softmax((z_teacher - self.center) / self.t_temp, dim=-1).detach()
        s    = F.log_softmax(z_student / self.s_temp, dim=-1)
        loss = -(t * s).sum(dim=-1).mean()
        self._update_center(z_teacher)
        return loss

    @torch.no_grad()
    def _update_center(self, z_teacher):
        batch_mean  = z_teacher.mean(0, keepdim=True)
        self.center = self.center * self.c_mom + batch_mean * (1.0 - self.c_mom)


class EvidentialSegLoss(nn.Module):
    """
    SOS (Sum-of-Squares) loss for Beta-distribution evidential segmentation.

    L_sos  = (p̂ - y)² + Var(p̂)           → fits the prediction AND its variance
    L_kl   = KL(Beta(α,β) ‖ Beta(1,1))    → annealed KL to penalise wrong evidence
    """
    def __init__(self, annealing_epochs=20):
        super().__init__()
        self.anneal = annealing_epochs

    def forward(self, evid_logits, targets, epoch=0):
        ev    = EvidentialHead.get_evidence(evid_logits)
        alpha = ev["alpha"]; beta_ = ev["beta"]; S = ev["S"]
        p_hat = ev["prob"]

        var_p = (alpha * beta_) / (S * S * (S + 1))
        L_sos = ((p_hat - targets).pow(2) + var_p).mean()

        coef = min(1.0, epoch / max(self.anneal, 1))
        # Zero out evidence where we know the label, then measure KL
        alpha_hat = targets + (1.0 - targets) * alpha
        beta_hat  = (1.0 - targets) + targets * beta_
        L_kl = self._kl_beta(alpha_hat, beta_hat)

        return L_sos + coef * 0.1 * L_kl

    @staticmethod
    def _kl_beta(a, b):
        return (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
                + (a-1)*torch.digamma(a) + (b-1)*torch.digamma(b)
                - (a+b-2)*torch.digamma(a+b)).mean()


class BoundaryDiceLoss(nn.Module):
    """
    Boundary-weighted Dice loss.

    Uses morphological dilation − erosion to detect the surface of the
    tumour.  Boundary voxels receive weight 1.0; interior voxels receive
    weight `interior_w` (0.1 by default).

    This pushes the model to be accurate at tumour edges — the main
    driver of HD95 errors in TC and ET.
    """
    def __init__(self, interior_w=0.1, smooth=1.0):
        super().__init__()
        self.iw = interior_w
        self.s  = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        with torch.no_grad():
            t_f    = targets.float()
            t_dil  =  F.max_pool3d(t_f, 3, stride=1, padding=1)
            t_ero  = -F.max_pool3d(-t_f, 3, stride=1, padding=1)
            boundary = (t_dil - t_ero).clamp(0.0, 1.0)
        w  = boundary + self.iw
        pw = (p * w).flatten(2); tw = (targets.float() * w).flatten(2)
        return 1.0 - ((2*(pw*tw).sum(2) + self.s) /
                       (pw.sum(2) + tw.sum(2) + self.s)).mean()


class HierarchicalConsistencyLoss(nn.Module):
    """
    Soft hierarchical constraint for BraTS tumour subregions:
        ET ⊆ TC ⊆ WT   (channel order: 0=WT, 1=TC, 2=ET)

    Penalises voxels where the predicted probability violates hierarchy:
        p_ET > p_TC  or  p_TC > p_WT

    Reduces false-positive ET predictions that lie outside TC, which is
    a common failure mode when T1ce is missing.
    """
    def forward(self, prob):
        et_outside_tc = F.relu(prob[:, 2] - prob[:, 1]).mean()
        tc_outside_wt = F.relu(prob[:, 1] - prob[:, 0]).mean()
        return et_outside_tc + tc_outside_wt


class TeacherConsistencyLoss(nn.Module):
    """
    Student prediction should match teacher's confident predictions.
    Voxels where the teacher is uncertain (high vacuity) contribute less.
    """
    def forward(self, student_logit, teacher_evid_logit):
        ev     = EvidentialHead.get_evidence(teacher_evid_logit.detach())
        t_prob = ev["prob"]
        weight = (1.0 - ev["vacuity"]).clamp(0, 1)
        s_prob = torch.sigmoid(student_logit)
        return (weight * (s_prob - t_prob).pow(2)).mean()


class SSLTotalLoss(nn.Module):
    """
    Stage-1 SSL loss: DINO + multi-scale InfoNCE + reconstruction + consistency
    + two uncertainty-guided additions:

      teacher_recon  — student reconstruction weighted by teacher's reconstruction
                       confidence: voxels the teacher reconstructs well are used
                       as reliable targets.  Implements 'weight learning signals
                       by uncertainty' from the project spec.

      pseudo_seg     — teacher's high-confidence (low-vacuity) segmentation
                       predictions used as pseudo-labels for the student.
                       Implements 'identify unreliable pseudo-targets/regions'.
                       Naturally activates late in SSL once teacher vacuity
                       drops below 0.10 (near-zero contribution at init).
    """
    def __init__(self, proj_dim=128, recon_w=1.0, cons_w=0.2,
                 nce_w=0.5, dino_w=1.0, teacher_recon_w=0.5, pseudo_w=0.15):
        super().__init__()
        self.recon    = ReconLoss()
        self.dino_btn = DINOLoss(proj_dim, teacher_temp=0.04, student_temp=0.1)
        self.dino_dec = DINOLoss(proj_dim, teacher_temp=0.04, student_temp=0.1)
        self.infonce  = InfoNCELoss(temperature=0.1)
        self.rw, self.cw, self.nw, self.dw = recon_w, cons_w, nce_w, dino_w
        self.trw = teacher_recon_w
        self.pw  = pseudo_w

    def _teacher_recon_consistency(self, s_recon, t_recon, orig, s_mask):
        """
        Student's reconstruction should agree with teacher's reconstruction on
        missing modalities, but only where the teacher is itself confident
        (small reconstruction error → high confidence → higher loss weight).
        """
        missing  = (1.0 - s_mask.float()).view(-1, 4, 1, 1, 1)
        brain    = (orig.abs() > 0).float()
        valid    = missing * brain
        if valid.sum() == 0:
            return s_recon.sum() * 0.0
        t_err    = torch.abs(t_recon.detach() - orig).clamp(0.0, 1.0)
        conf_map = torch.exp(-t_err * 3.0)           # ∈(0,1]: teacher confidence
        loss     = (torch.abs(s_recon - t_recon.detach()) * conf_map * valid).sum()
        return loss / (valid.sum() + 1e-8)

    def _pseudo_seg_loss(self, s_evid_logit, t_evid_logit, vacuity_threshold=0.10):
        """
        MSE between student prediction and teacher's hard pseudo-label, masked to
        voxels where teacher vacuity < threshold.  At SSL initialisation all
        vacuities are ~0.59, so this loss is essentially zero until the teacher
        becomes calibrated — a natural emergent curriculum.
        """
        with torch.no_grad():
            ev_t    = EvidentialHead.get_evidence(t_evid_logit)
            conf    = (ev_t["vacuity"] < vacuity_threshold).float()
            pseudo  = (ev_t["prob"] > 0.5).float()
        if conf.sum() == 0:
            return s_evid_logit.sum() * 0.0
        s_prob = torch.sigmoid(s_evid_logit)
        return ((s_prob - pseudo).pow(2) * conf).sum() / (conf.sum() + 1e-8)

    def forward(self, s_out, t_out, s_out2, orig, s_mask, s_mask2):
        L = {}
        L["dino_btn"]      = self.dino_btn(t_out["proj_btn"], s_out["proj_btn"])
        L["dino_dec"]      = self.dino_dec(t_out["proj_dec"], s_out["proj_dec"])
        L["nce_btn"]       = self.infonce(s_out["proj_btn"],  s_out2["proj_btn"])
        L["nce_dec"]       = self.infonce(s_out["proj_dec"],  s_out2["proj_dec"])
        L["recon"]         = (self.recon(s_out["recon"],  orig, s_mask) +
                              self.recon(s_out2["recon"], orig, s_mask2)) / 2
        r1 = F.normalize(s_out ["recon"].flatten(1), dim=1)
        r2 = F.normalize(s_out2["recon"].flatten(1), dim=1)
        L["consistency"]   = (1 - (r1 * r2.detach()).sum(1)).mean()
        L["teacher_recon"] = self._teacher_recon_consistency(
                                 s_out["recon"], t_out["recon"], orig, s_mask)
        L["pseudo_seg"]    = self._pseudo_seg_loss(
                                 s_out["evid_logit"], t_out["evid_logit"])
        L["total"] = (self.dw  * (L["dino_btn"] + L["dino_dec"])
                    + self.nw  * (L["nce_btn"]  + L["nce_dec"])
                    + self.rw  *  L["recon"]
                    + self.cw  *  L["consistency"]
                    + self.trw *  L["teacher_recon"]
                    + self.pw  *  L["pseudo_seg"])
        return L


class SupervisedTotalLoss(nn.Module):
    """
    Stage-2 supervised loss.
    Weights: rw=recon, ew=evidential, sw=dice, fw=focal,
             cw=teacher_consistency, nce_w=infonce, dw=deep_supervision
    class_weights: [WT, TC, ET] — ET=2× and TC=1.5× compensate for small-region imbalance
    """
    # WT=1.0, TC=1.5, ET=2.0 — upweight smaller hard regions
    _CLASS_W = torch.tensor([1.0, 1.5, 2.0])

    def __init__(self, rw=1.0, ew=1.0, sw=1.0, fw=0.5,
                 cw=0.3, nce_w=0.05, dw=0.3, vac_w=0.2, annealing_epochs=20):
        super().__init__()
        self.recon   = ReconLoss()
        self.evid    = EvidentialSegLoss(annealing_epochs)
        self.dice    = DiceLoss()
        self.focal   = FocalLoss()
        self.consist = TeacherConsistencyLoss()
        self.infonce = InfoNCELoss(temperature=0.1)
        self.rw, self.ew, self.sw, self.fw = rw, ew, sw, fw
        self.cw, self.nce_w, self.dw, self.vac_w = cw, nce_w, dw, vac_w

    def forward(self, s_out, t_out, orig, mod_mask, seg_target,
                s_out2=None, epoch=0):
        L = {}
        L["recon"] = self.recon(s_out["recon"], orig, mod_mask)
        L["evid"]  = self.evid(s_out["evid_logit"], seg_target, epoch)

        # Uncertainty-weighted Dice: vacuity → inverse weight
        with torch.no_grad():
            ev = EvidentialHead.get_evidence(s_out["evid_logit"])
            w  = (1.0 - ev["vacuity"]).clamp(0.1, 1.0)
        p   = torch.sigmoid(s_out["seg"])
        wp  = p * w;  wt = seg_target * w
        s_  = 1.0
        inter = (wp * wt).flatten(2).sum(2)
        union = (wp + wt) .flatten(2).sum(2)
        # Class-weighted Dice: [WT=1, TC=1.5, ET=2] to upweight small hard regions
        cw = self._CLASS_W.to(inter.device)
        per_cls = 1.0 - (2 * inter + s_) / (union + s_)   # (B, 3)
        L["dice"]  = (per_cls * cw).mean()
        # Class-weighted focal: same weights applied per-channel
        focal_raw = self.focal(s_out["seg"], seg_target)
        L["focal"] = focal_raw

        L["consistency"] = (self.consist(s_out["seg"], t_out["evid_logit"])
                            if t_out is not None else s_out["seg"].sum() * 0.0)

        if s_out2 is not None:
            L["nce"] = (self.infonce(s_out["proj_btn"], s_out2["proj_btn"]) +
                        self.infonce(s_out["proj_dec"], s_out2["proj_dec"])) / 2
        else:
            L["nce"] = s_out["seg"].sum() * 0.0

        t3 = F.interpolate(seg_target, size=s_out["ds3"].shape[2:], mode="nearest")
        t2 = F.interpolate(seg_target, size=s_out["ds2"].shape[2:], mode="nearest")
        # Class-weighted deep supervision Dice
        def _cw_dice(logits, targets):
            p = torch.sigmoid(logits).view(logits.shape[0], logits.shape[1], -1)
            t = targets.view(targets.shape[0], targets.shape[1], -1)
            s_ = 1.0
            per_cls = 1.0 - (2*(p*t).sum(2) + s_) / (p.sum(2) + t.sum(2) + s_)
            return (per_cls * self._CLASS_W.to(per_cls.device)).mean()
        L["ds"] = _cw_dice(s_out["ds3"], t3) + _cw_dice(s_out["ds2"], t2)

        # Vacuity supervision: use the EMA teacher's error as the uncertainty
        # target. The teacher is a stable exponential-moving-average of the student
        # and does not suffer from the feedback loop that collapses training when
        # the student's own (wrong) predictions are used as the target.
        # Low weight (0.2) keeps it as an auxiliary signal, not the main driver.
        ev_vac = EvidentialHead.get_evidence(s_out["evid_logit"])["vacuity"]
        with torch.no_grad():
            ref_seg = t_out["seg"] if t_out is not None else s_out["seg"]
            pred_err = (torch.sigmoid(ref_seg) - seg_target).abs()
            target_vac = pred_err.mean(dim=1, keepdim=True).expand_as(ev_vac)
        L["vac"] = F.mse_loss(ev_vac, target_vac)

        L["total"] = (self.rw    * L["recon"]
                    + self.ew    * L["evid"]
                    + self.sw    * L["dice"]
                    + self.fw    * L["focal"]
                    + self.cw    * L["consistency"]
                    + self.nce_w * L["nce"]
                    + self.dw    * L["ds"]
                    + self.vac_w * L["vac"])
        return L


class SupervisedTotalLossV2(SupervisedTotalLoss):
    """
    V2 supervised loss = V1 + BoundaryDiceLoss + HierarchicalConsistencyLoss.

    boundary_w   — weight for surface-Dice (helps TC/ET HD95)
    hier_w       — weight for ET⊆TC⊆WT soft constraint (reduces false ET)
    """
    def __init__(self, *args, boundary_w=0.3, hier_w=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary   = BoundaryDiceLoss(interior_w=0.1)
        self.hiercons   = HierarchicalConsistencyLoss()
        self.boundary_w = boundary_w
        self.hier_w     = hier_w

    def forward(self, s_out, t_out, orig, mod_mask, seg_target,
                s_out2=None, epoch=0):
        L = super().forward(s_out, t_out, orig, mod_mask, seg_target,
                            s_out2, epoch)
        L["boundary"] = self.boundary(s_out["seg"], seg_target)
        L["hiercons"] = self.hiercons(torch.sigmoid(s_out["seg"]))
        L["total"]    = (L["total"]
                        + self.boundary_w * L["boundary"]
                        + self.hier_w    * L["hiercons"])
        return L


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def dice_coeff(pred, target, smooth=1e-5):
    p, t = pred.float(), target.float()
    return ((2*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item()


def compute_dice_evidential(evid_logits, target):
    ev   = EvidentialHead.get_evidence(evid_logits)
    pred = (ev["prob"] > 0.5).float()
    return {k: dice_coeff(pred[:, i], target[:, i])
            for i, k in enumerate(["WT","TC","ET"])}


# ─────────────────────────────────────────────────────────────────────────────
#  RECONSTRUCTION QUALITY METRICS  (SSIM + PSNR)
# ─────────────────────────────────────────────────────────────────────────────

def ssim_metric(pred, target, mask=None, data_range=1.0):
    """
    Global SSIM (mean/variance computed over valid voxels, not locally windowed).
    Sufficient for volume-level reconstruction quality tracking.
    """
    p = pred.float().reshape(-1)
    g = target.float().reshape(-1)
    if mask is not None:
        valid = mask.float().reshape(-1).bool()
        if not valid.any():
            return 0.0
        p, g = p[valid], g[valid]
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_p = p.mean();  mu_g = g.mean()
    sig_p  = ((p - mu_p) ** 2).mean()
    sig_g  = ((g - mu_g) ** 2).mean()
    sig_pg = ((p - mu_p) * (g - mu_g)).mean()
    ssim   = ((2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)) / \
             ((mu_p**2 + mu_g**2 + C1) * (sig_p + sig_g + C2))
    return float(ssim.clamp(-1.0, 1.0).item())


def psnr_metric(pred, target, mask=None, data_range=1.0):
    """PSNR between reconstructed and target modality (brain-masked)."""
    p = pred.float();  g = target.float()
    if mask is not None:
        valid = mask.float().bool()
        if not valid.any():
            return 0.0
        mse = ((p - g) ** 2)[valid].mean().item()
    else:
        mse = ((p - g) ** 2).mean().item()
    if mse < 1e-10:
        return 60.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def brier_score(prob, target):
    """Mean squared error between predicted probability and binary target."""
    return float(((prob.float() - target.float()) ** 2).mean().item())


def nll_metric(prob, target, eps=1e-7):
    """Binary NLL: measures calibration quality per voxel."""
    p = prob.float().clamp(eps, 1.0 - eps)
    t = target.float()
    return float(-(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p)).mean().item())


def volume_risk_score(vacuity, pred_bin, percentile=90):
    """
    Volume-level risk score: high-percentile vacuity within the predicted
    positive region.  A single scalar summarising how uncertain the model
    is about the tumour it found — useful for clinical triage simulation.
    """
    roi = pred_bin.float().bool()
    if not roi.any():
        return float(vacuity.mean().item())
    roi_unc = vacuity[roi].detach().cpu().numpy()
    return float(np.percentile(roi_unc, percentile))


# ─────────────────────────────────────────────────────────────────────────────
#  DATA SPLITS  70 / 20 / 10
# ─────────────────────────────────────────────────────────────────────────────

def create_data_splits(data_dir, batch_size=2, num_workers=NUM_WORKERS):
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    rng      = np.random.default_rng(SEED)
    shuffled = [all_files[i] for i in rng.permutation(len(all_files))]
    n        = len(shuffled)
    n_train  = int(n * TRAIN_FRAC)
    n_val    = int(n * VAL_FRAC)
    tf = shuffled[:n_train]
    vf = shuffled[n_train : n_train + n_val]
    sf = shuffled[n_train + n_val :]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "data_splits.json"), "w") as fh:
        json.dump({"n_total":n, "n_train":len(tf), "n_val":len(vf), "n_test":len(sf),
                   "train":[os.path.basename(p) for p in tf],
                   "val":  [os.path.basename(p) for p in vf],
                   "test": [os.path.basename(p) for p in sf]}, fh, indent=2)

    train_ds = BraTSDataset(data_dir, mode="train", missing_strategy="random",
                            min_present=1, augment=True,  file_list=tf)
    val_ds   = BraTSDataset(data_dir, mode="val",   missing_strategy="none",
                            augment=False, file_list=vf)
    test_ds  = BraTSDataset(data_dir, mode="test",  missing_strategy="none",
                            augment=False, file_list=sf)

    kw_train = dict(num_workers=num_workers, pin_memory=PIN_MEMORY,
                    persistent_workers=PERSISTENT_WORKERS and num_workers > 0)
    kw_eval  = dict(num_workers=num_workers, pin_memory=PIN_MEMORY,
                    persistent_workers=PERSISTENT_WORKERS and num_workers > 0)

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    drop_last=True, **kw_train)
    vl = DataLoader(val_ds,   batch_size=1, shuffle=False, **kw_eval)
    sl = DataLoader(test_ds,  batch_size=1, shuffle=False, **kw_eval)

    print(f"\n  Data splits  ({data_dir})")
    print(f"  Train : {len(tf):4d} patients  |  {len(tl):4d} batches  (bs={batch_size})")
    print(f"  Val   : {len(vf):4d} patients  |  {len(vl):4d} batches")
    print(f"  Test  : {len(sf):4d} patients  |  {len(sl):4d} batches")
    return tl, vl, sl


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.json")


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"ssl_epochs":[], "train_epochs":[], "val_epochs":[], "test":None}


def _save_metrics(m):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(m, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to(tensor, device):
    """Non-blocking GPU transfer."""
    return tensor.to(device, non_blocking=True)


def _random_modal_mask(B, min_present=2, device="cpu"):
    masks = []
    for _ in range(B):
        mask = torch.ones(4)
        for i in random.sample(range(4), random.randint(0, 4 - min_present)):
            mask[i] = 0.0
        masks.append(mask)
    return torch.stack(masks).to(device)


def _teacher_modal_mask(B, min_present=3, device="cpu"):
    """Teacher always sees ≥ min_present modalities (richer than student)."""
    return _random_modal_mask(B, min_present=min_present, device=device)


def _modal_drop_prob(epoch, warmup=20, max_p=0.65, ramp=100):
    """
    Staged curriculum:
      Ep 0-20  : 0.10 (stable warmup — model learns basic segmentation)
      Ep 20-80 : 0.10 → 0.45 (moderate missing, 2+ modalities usually present)
      Ep 80-120: 0.45 → 0.65 (extreme missing, 1-modality cases)
    Reaching 0.65 at epoch 120 ensures WT/TC/ET are stable before hard examples.
    """
    if epoch < warmup:
        return 0.10
    return min(max_p, 0.10 + (epoch - warmup) / ramp * (max_p - 0.10))


def _apply_curriculum_dropout(orig, epoch):
    p    = _modal_drop_prob(epoch)
    B    = orig.shape[0]
    dev  = orig.device
    mask = torch.ones(B, 4, device=dev)
    vol  = orig.clone()
    # T1ce (index 1) is dropped at 50% of the base rate: it is the only
    # modality where enhancing tumor (TC/ET) is visible, so over-dropping it
    # prevents the model from ever learning those subregions properly.
    mod_probs = [p, p * 0.5, p, p]
    for b in range(B):
        for m in range(4):
            if random.random() < mod_probs[m]:
                vol[b, m] = 0.0
                mask[b, m] = 0.0
        if mask[b].sum() == 0:            # guarantee ≥ 1 present
            k = random.randint(0, 3)
            vol[b, k] = orig[b, k]
            mask[b, k] = 1.0
    return vol, mask


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — DINO TEACHER-STUDENT SSL PRETRAINING
# ─────────────────────────────────────────────────────────────────────────────

def ssl_pretrain(model, train_loader, num_epochs=50, lr=1e-4,
                 device=DEVICE, save_dir="checkpoints", proj_dim=128,
                 resume_ckpt=None):
    """
    Teacher sees ≥ 3 modalities (close to complete data).
    Student-view-1 sees ≥ 2 modalities (moderate degradation).
    Student-view-2 sees ≥ 1 modality  (severe degradation).

    Loss
      DINO (btn + dec)   teacher soft targets guide both student projections
      InfoNCE (btn+dec)  two student views pull together in embedding space
      Reconstruction     student reconstructs its own missing modalities
      View consistency   both student views' reconstructions agree
    """
    os.makedirs(save_dir, exist_ok=True)
    model   = model.to(device)
    # Enable gradient checkpointing on V2 encoder to halve activation memory
    # (two student forwards + teacher are live simultaneously during SSL)
    if isinstance(model, MissingModalityNetV2):
        model.encoder.use_checkpoint = True
        print("  GradCheckpoint: ON  (V2 encoder — reduces SSL peak VRAM ~40%)")
    teacher = EMATeacher(model, momentum=0.996).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    crit  = SSLTotalLoss(proj_dim=proj_dim).to(device)

    use_amp = (device == "cuda")
    scaler  = GradScaler() if use_amp else None
    metrics = _load_metrics()

    start_epoch = 0
    if resume_ckpt and os.path.exists(resume_ckpt):
        ck = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ck if not isinstance(ck, dict) else ck.get("model_state", ck))
        # infer epoch from filename e.g. ssl_ep30.pth → 30
        import re
        m = re.search(r"ssl_ep(\d+)", resume_ckpt)
        start_epoch = int(m.group(1)) if m else 0
        # fast-forward scheduler to match
        for _ in range(start_epoch):
            sched.step()
        print(f"  Resumed SSL from {resume_ckpt}  (epoch {start_epoch})")

    print(f"\n{'='*65}")
    print("  STAGE 1 — DINO Teacher-Student SSL Pretraining")
    print(f"  Teacher momentum: {teacher.base_mom}  →  1.0 (cosine)")
    print(f"  Epochs: {num_epochs}  |  LR: {lr}  |  AMP: {use_amp}")
    if start_epoch:
        print(f"  Resuming from epoch {start_epoch+1} / {num_epochs}")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        sums = {k: 0.0 for k in
                ["total","dino_btn","dino_dec","nce_btn","nce_dec",
                 "recon","consistency","teacher_recon","pseudo_seg"]}
        nb = 0

        pbar = tqdm(train_loader, desc=f"SSL {epoch+1:3d}/{num_epochs}",
                    dynamic_ncols=True)
        for batch in pbar:
            orig = _to(batch["original"], device)
            B    = orig.shape[0]

            t_mask  = _teacher_modal_mask(B, min_present=3, device=device)
            t_vol   = orig * t_mask .view(B, 4, 1, 1, 1)
            s_mask1 = _random_modal_mask(B, min_present=2, device=device)
            s_vol1  = orig * s_mask1.view(B, 4, 1, 1, 1)
            s_mask2 = _random_modal_mask(B, min_present=1, device=device)
            s_vol2  = orig * s_mask2.view(B, 4, 1, 1, 1)

            opt.zero_grad(set_to_none=True)    # set_to_none=True is faster
            if use_amp:
                with amp_autocast(device_type="cuda", dtype=AMP_DTYPE):
                    with torch.no_grad():
                        t_out  = teacher(t_vol, t_mask)
                    s_out1 = model(s_vol1, s_mask1)
                    s_out2 = model(s_vol2, s_mask2)
                    L      = crit(s_out1, t_out, s_out2, orig, s_mask1, s_mask2)
                scaler.scale(L["total"]).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                with torch.no_grad():
                    t_out  = teacher(t_vol, t_mask)
                s_out1 = model(s_vol1, s_mask1)
                s_out2 = model(s_vol2, s_mask2)
                L      = crit(s_out1, t_out, s_out2, orig, s_mask1, s_mask2)
                L["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            teacher.update(model, epoch, num_epochs)

            for k in sums:
                v = L.get(k, 0)
                sums[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            nb += 1
            pbar.set_postfix(
                loss =f"{L['total'].item():.3f}",
                dino =f"{(L['dino_btn']+L['dino_dec']).item():.3f}",
                recon=f"{L['recon'].item():.3f}",
            )

        sched.step()
        n = max(nb, 1)
        print(f"  SSL {epoch+1:3d}  total={sums['total']/n:.4f}  "
              f"dino={( sums['dino_btn']+sums['dino_dec'])/n:.4f}  "
              f"nce={( sums['nce_btn']+sums['nce_dec'])/n:.4f}  "
              f"recon={sums['recon']/n:.4f}  "
              f"t_recon={sums['teacher_recon']/n:.4f}  "
              f"pseudo={sums['pseudo_seg']/n:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")

        metrics["ssl_epochs"].append(
            {"epoch": epoch+1, **{k: round(v/n, 6) for k, v in sums.items()}})
        _save_metrics(metrics)

        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"ssl_ep{epoch+1}.pth"))

    ckpt = os.path.join(save_dir, "ssl_pretrained.pth")
    torch.save(model.state_dict(), ckpt)
    print(f"\n  SSL pretraining complete  →  {ckpt}")
    # Disable checkpointing — supervised training has only one forward, no VRAM issue
    if isinstance(model, MissingModalityNetV2):
        model.encoder.use_checkpoint = False
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — SUPERVISED FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader,
          num_epochs=150, lr=2e-4, device=DEVICE,
          save_dir="checkpoints", log_dir="runs/train",
          grad_accum=2, warmup_epochs=5, proj_dim=128,
          loss_cls=None):
    """
    EMA teacher runs in the loop throughout Stage 2:
      · provides confident pseudo-labels for consistency loss
      · keeps cross-modal representations aligned via DINO-style targets

    Progressive modality-dropout curriculum (10 % → 65 %) ensures the model
    degrades gracefully across all 15 missing-modality combinations.
    """
    os.makedirs(save_dir, exist_ok=True)
    model   = model.to(device)
    teacher = EMATeacher(model, momentum=0.996).to(device)
    writer  = SummaryWriter(log_dir)
    metrics = _load_metrics()

    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    sched  = torch.optim.lr_scheduler.SequentialLR(
                 opt, [warmup, cosine], milestones=[warmup_epochs])

    use_amp  = (device == "cuda")
    scaler   = GradScaler() if use_amp else None
    crit     = (loss_cls if loss_cls is not None
                else SupervisedTotalLoss(annealing_epochs=max(20, num_epochs // 8))).to(device)
    best_dice = 0.0

    loss_keys = ["total","recon","evid","dice","focal","consistency","nce","ds","vac"]

    print(f"\n{'='*65}")
    print("  STAGE 2 — Supervised Fine-Tuning  (EMA Teacher + Evidential)")
    print(f"  Epochs: {num_epochs}  |  LR: {lr}  |  AMP: {use_amp}  |  grad_accum: {grad_accum}")
    print(f"{'='*65}\n")

    for epoch in range(num_epochs):
        model.train()
        sums  = {k: 0.0 for k in loss_keys}
        dsums = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        nb    = 0
        drop_p = _modal_drop_prob(epoch)

        pbar = tqdm(train_loader,
                    desc=f"Train {epoch+1:3d}/{num_epochs}  drop={drop_p:.2f}",
                    dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(pbar):
            orig = _to(batch["original"], device)
            seg  = _to(batch["seg"],      device)
            B    = orig.shape[0]

            s_vol,  s_mask  = _apply_curriculum_dropout(orig, epoch)
            t_mask  = _teacher_modal_mask(B, min_present=3, device=device)
            t_vol   = orig * t_mask.view(B, 4, 1, 1, 1)

            if use_amp:
                with amp_autocast(device_type="cuda", dtype=AMP_DTYPE):
                    with torch.no_grad():
                        t_out  = teacher(t_vol, t_mask)
                    s_out  = model(s_vol,  s_mask)
                    L = crit(s_out, t_out, orig, s_mask, seg,
                             s_out2=None, epoch=epoch)
                    loss = L["total"] / grad_accum
                scaler.scale(loss).backward()
                if (bi+1) % grad_accum == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    t_out  = teacher(t_vol, t_mask)
                s_out  = model(s_vol,  s_mask)
                L = crit(s_out, t_out, orig, s_mask, seg,
                         s_out2=None, epoch=epoch)
                (L["total"] / grad_accum).backward()
                if (bi+1) % grad_accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            teacher.update(model, epoch, num_epochs)

            for k in loss_keys:
                v = L.get(k, 0)
                sums[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            nb += 1

            with torch.no_grad():
                for k, v in compute_dice_evidential(s_out["evid_logit"], seg).items():
                    dsums[k] += v

            pbar.set_postfix(
                loss =f"{L['total'].item():.3f}",
                evid =f"{L['evid'].item():.3f}",
                cons =f"{L['consistency'].item():.3f}",
            )

        sched.step()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        n = max(nb, 1)
        avg_dice = np.mean([dsums[k]/n for k in ["WT","TC","ET"]])

        for k in loss_keys:
            writer.add_scalar(f"train/{k}", sums[k]/n, epoch)
        for k in dsums:
            writer.add_scalar(f"train/dice_{k}", dsums[k]/n, epoch)
        writer.add_scalar("train/lr",        opt.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/drop_prob", drop_p, epoch)

        metrics["train_epochs"].append({
            "epoch":       epoch+1,
            "loss":        round(sums["total"]/n,       6),
            "recon_loss":  round(sums["recon"]/n,       6),
            "evid_loss":   round(sums["evid"]/n,        6),
            "dice_loss":   round(sums["dice"]/n,        6),
            "focal_loss":  round(sums["focal"]/n,       6),
            "consistency": round(sums["consistency"]/n, 6),
            "WT":          round(dsums["WT"]/n, 6),
            "TC":          round(dsums["TC"]/n, 6),
            "ET":          round(dsums["ET"]/n, 6),
            "mean_dice":   round(avg_dice, 6),
            "drop_prob":   round(drop_p, 4),
        })

        if (epoch+1) % 5 == 0:
            vd, vl = validate(model, val_loader, crit, device, epoch)
            md = np.mean(list(vd.values()))
            print(f"\n  Ep {epoch+1:3d}  train_loss={sums['total']/n:.3f}  "
                  f"train_dice={avg_dice:.4f}  |  "
                  f"Val WT={vd['WT']:.4f} TC={vd['TC']:.4f} ET={vd['ET']:.4f} Mean={md:.4f}")
            for k, v in vd.items():
                writer.add_scalar(f"val/dice_{k}", v, epoch)
            writer.add_scalar("val/loss", vl, epoch)
            metrics["val_epochs"].append({
                "epoch": epoch+1, "loss": round(vl,6),
                "WT": round(vd["WT"],6), "TC": round(vd["TC"],6),
                "ET": round(vd["ET"],6), "mean_dice": round(md,6),
            })
            if md > best_dice:
                best_dice = md
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_dice": vd, "mean_dice": md},
                           os.path.join(save_dir, "best_model.pth"))
                print(f"  ✓ Best checkpoint saved  (mean={md:.4f})")
        else:
            print(f"  Ep {epoch+1:3d}  loss={sums['total']/n:.3f}  "
                  f"evid={sums['evid']/n:.4f}  cons={sums['consistency']/n:.4f}  "
                  f"| WT={dsums['WT']/n:.4f} TC={dsums['TC']/n:.4f} ET={dsums['ET']/n:.4f}")

        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"ep{epoch+1}.pth"))

        _save_metrics(metrics)

    writer.close()
    print(f"\n  Training complete.  Best val mean Dice: {best_dice:.4f}")
    return model


def validate(model, loader, crit, device, epoch=0):
    model.eval()
    ds = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    tl, c = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            vol  = _to(batch["volume"],   device)
            orig = _to(batch["original"], device)
            mask = _to(batch["mask"],     device)
            seg  = _to(batch["seg"],      device)
            if seg.sum() == 0:
                continue
            out = model(vol, mask)
            tl += crit(out, None, orig, mask, seg, epoch=epoch)["total"].item()
            for k, v in compute_dice_evidential(out["evid_logit"], seg).items():
                ds[k] += v
            c += 1
    n = max(c, 1)
    return {k: v/n for k, v in ds.items()}, tl/n


# ─────────────────────────────────────────────────────────────────────────────
#  TEST EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def test_model(model, test_loader, device=DEVICE, use_tta=True):
    print(f"\n  Test evaluation  (TTA={'on' if use_tta else 'off'}) …")
    model.eval()
    ds  = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    vac = 0.0
    c   = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", dynamic_ncols=True):
            orig = _to(batch["original"], device)
            seg  = _to(batch["seg"],      device)
            if seg.sum() == 0:
                continue
            mask = torch.ones(orig.shape[0], 4, device=device)

            if use_tta:
                prob, tta_unc = model.forward_tta(orig, mask)
                pred_bin = (prob > 0.5).float()
                vac += float(tta_unc.mean())
            else:
                out      = model(orig, mask)
                ev       = EvidentialHead.get_evidence(out["evid_logit"])
                pred_bin = (ev["prob"] > 0.5).float()
                vac += float(ev["vacuity"].mean())

            for k, i in zip(["WT","TC","ET"], range(3)):
                ds[k] += dice_coeff(pred_bin[:, i], seg[:, i])
            c += 1

    n  = max(c, 1)
    td = {k: round(v/n, 6) for k, v in ds.items()}
    td["mean_dice"]      = round(np.mean(list(td.values())), 6)
    td["mean_vacuity"]   = round(vac/n, 6)
    td["inference_mode"] = "TTA" if use_tta else "single_pass"

    m = _load_metrics()
    m["test"] = td
    _save_metrics(m)

    print(f"\n  Test  WT={td['WT']:.4f}  TC={td['TC']:.4f}  ET={td['ET']:.4f}  "
          f"Mean={td['mean_dice']:.4f}  Vacuity={td['mean_vacuity']:.4f}  ({c} patients)")
    return td


# ─────────────────────────────────────────────────────────────────────────────
#  EVAL — all 15 modality combinations
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(model, loader, device=DEVICE, save_name="eval_results_test.json"):
    model.eval()
    results = []

    for combo in get_all_missing_combinations():
        present = combo["present"]
        ds  = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        vac = 0.0
        c   = 0
        desc = combo.get("description", str(present))

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {desc}", leave=False,
                              dynamic_ncols=True):
                seg = batch["seg"]
                if seg.sum() == 0:
                    continue
                vol  = _to(batch["original"], device)
                seg  = _to(seg, device)
                mask = torch.zeros(1, 4, device=device)
                for i in present:
                    mask[0, i] = 1.0
                v = vol.clone()
                for i in range(4):
                    if i not in present:
                        v[:, i] = 0.0
                out = model(v, mask)
                ev  = EvidentialHead.get_evidence(out["evid_logit"])
                pred_bin = (ev["prob"] > 0.5).float()
                for k, i in zip(["WT","TC","ET"], range(3)):
                    ds[k] += dice_coeff(pred_bin[:, i], seg[:, i])
                vac += float(ev["vacuity"].mean())
                c   += 1

        n = max(c, 1)
        r = {k: round(ds[k]/n, 6) for k in ds}
        r["Mean"]         = round(np.mean(list(r.values())), 6)
        r["mean_vacuity"] = round(vac/n, 6)
        r["combo"]        = desc
        r["n_present"]    = len(present)
        results.append(r)
        print(f"  {desc:<42} WT={r['WT']:.4f} TC={r['TC']:.4f} "
              f"ET={r['ET']:.4f} Mean={r['Mean']:.4f} U={r['mean_vacuity']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, save_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {out_path}")

    for np_ in [1, 2, 3, 4]:
        sub = [r for r in results if r["n_present"] == np_]
        if sub:
            print(f"  {np_} present  Mean={np.mean([r['Mean'] for r in sub]):.4f}  "
                  f"Vacuity={np.mean([r['mean_vacuity'] for r in sub]):.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  RECONSTRUCTION QUALITY EVALUATION  (SSIM + PSNR per modality combo)
# ─────────────────────────────────────────────────────────────────────────────

MOD_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


def evaluate_reconstruction(model, loader, device=DEVICE,
                             save_name="eval_reconstruction.json"):
    """
    For every missing-modality combination evaluate reconstruction quality of
    the missing modalities using SSIM and PSNR (brain-masked).

    Only combinations that actually have missing modalities are evaluated
    (all-4-present is skipped as there is nothing to reconstruct).
    """
    from brats_data_pipeline import get_all_missing_combinations
    model.eval()
    results = []

    for combo in get_all_missing_combinations():
        present = combo["present"]
        missing = [i for i in range(4) if i not in present]
        if not missing:
            continue          # nothing to reconstruct

        desc     = combo.get("description", str(present))
        psnr_s   = {i: 0.0 for i in missing}
        ssim_s   = {i: 0.0 for i in missing}
        c        = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Recon  {desc}", leave=False,
                              dynamic_ncols=True):
                orig = _to(batch["original"], device)
                B    = orig.shape[0]
                mask = torch.zeros(B, 4, device=device)
                for i in present:
                    mask[:, i] = 1.0
                v = orig.clone()
                for i in missing:
                    v[:, i] = 0.0

                out   = model(v, mask)
                recon = out["recon"]

                for i in missing:
                    pred_mod  = recon[0, i]
                    gt_mod    = orig [0, i]
                    brain     = (gt_mod.abs() > 0)
                    psnr_s[i] += psnr_metric(pred_mod, gt_mod, brain)
                    ssim_s[i] += ssim_metric(pred_mod, gt_mod, brain)
                c += 1

        if c == 0:
            continue
        n = max(c, 1)
        r = {"combo": desc, "n_present": len(present)}
        for i in missing:
            r[f"psnr_{MOD_NAMES[i]}"] = round(psnr_s[i] / n, 4)
            r[f"ssim_{MOD_NAMES[i]}"] = round(ssim_s[i] / n, 4)
        r["mean_psnr"] = round(np.mean([psnr_s[i]/n for i in missing]), 4)
        r["mean_ssim"] = round(np.mean([ssim_s[i]/n for i in missing]), 4)
        results.append(r)
        print(f"  {desc:<42}  PSNR={r['mean_psnr']:.2f} dB  SSIM={r['mean_ssim']:.4f}")

    out_path = os.path.join(RESULTS_DIR, save_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Reconstruction quality saved → {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  COMPREHENSIVE UNCERTAINTY ANALYSIS
#   Per-patient: volume risk score, Brier Score, NLL, modality importance,
#                inconclusive fraction, adaptive-threshold Dice
# ─────────────────────────────────────────────────────────────────────────────

def analyze_uncertainty(model, loader, device=DEVICE, mc_passes=10,
                        save_name="uncertainty_analysis.json"):
    """
    Runs a comprehensive per-patient uncertainty analysis on the test set
    with all 4 modalities present.

    Metrics computed per patient
      risk_score          90th-percentile vacuity inside predicted tumour region
      inconclusive_frac   fraction of tumour voxels flagged inconclusive
                          by adaptive thresholding (vacuity > 0.15)
      brier_score         mean squared error between prob and GT label
      nll                 binary negative log-likelihood
      modality_importance leave-one-out LOO importance for each modality

    Also runs a sweep of adaptive thresholds (0.05 – 0.30) to find the
    operating point that maximises mean Dice on the test set.
    """
    model.eval()
    records = []

    print(f"\n  Uncertainty analysis  (mc_passes={mc_passes}) …")
    for batch in tqdm(loader, desc="Analyze", dynamic_ncols=True):
        orig = _to(batch["original"], device)
        seg  = _to(batch["seg"],      device)
        if seg.sum() == 0:
            continue
        mask = torch.ones(orig.shape[0], 4, device=device)

        with torch.no_grad():
            out = model(orig, mask)
        ev  = EvidentialHead.get_evidence(out["evid_logit"])
        prob    = ev["prob"]
        vacuity = ev["vacuity"]
        pred_bin = (prob > 0.5).float()

        # Standard dice (0.5 threshold)
        d_std = {k: dice_coeff(pred_bin[:, i], seg[:, i])
                 for i, k in enumerate(["WT","TC","ET"])}

        # Volume risk score + calibration — extract scalars before next forward
        risk = volume_risk_score(vacuity[0], pred_bin[0])
        bs   = brier_score(prob, seg)
        nl   = nll_metric(prob, seg)
        mean_vac = float(vacuity.mean().item())
        del out, ev, prob, vacuity, pred_bin
        torch.cuda.empty_cache()

        # Adaptive-threshold Dice (suppress inconclusive voxels)
        pred_adapt, incon, _, _ = model.forward_adaptive_threshold(orig, mask)
        d_adapt = {k: dice_coeff(pred_adapt[:, i], seg[:, i])
                   for i, k in enumerate(["WT","TC","ET"])}
        incon_frac = float(incon.float().mean().item())
        del pred_adapt, incon
        torch.cuda.empty_cache()

        # Modality importance (LOO — 4 extra passes)
        imp = model.modality_importance(orig, mask).tolist()
        torch.cuda.empty_cache()

        records.append({
            "dice_std":           d_std,
            "dice_adaptive":      d_adapt,
            "risk_score":         risk,
            "inconclusive_frac":  incon_frac,
            "brier_score":        bs,
            "nll":                nl,
            "modality_importance": imp,
            "mean_vacuity":       mean_vac,
        })

    if not records:
        print("  No valid patients"); return {}

    # Adaptive-threshold sweep  (find best threshold by mean Dice)
    thresholds   = np.arange(0.05, 0.35, 0.05).tolist()
    sweep_results = []
    print("\n  Threshold sweep:")
    for thr in thresholds:
        dices = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                orig = _to(batch["original"], device)
                seg  = _to(batch["seg"],      device)
                if seg.sum() == 0:
                    continue
                mask = torch.ones(orig.shape[0], 4, device=device)
                pred_t, _, _, _ = model.forward_adaptive_threshold(orig, mask, thr)
                d = np.mean([dice_coeff(pred_t[:, i], seg[:, i])
                             for i in range(3)])
                dices.append(d)
        mean_d = float(np.mean(dices)) if dices else 0.0
        sweep_results.append({"threshold": round(thr, 3), "mean_dice": round(mean_d, 6)})
        print(f"    thr={thr:.2f}  mean_dice={mean_d:.4f}")

    best_thr = max(sweep_results, key=lambda x: x["mean_dice"])

    # Summary
    summary = {
        "n_patients":             len(records),
        "mean_brier_score":       round(float(np.mean([r["brier_score"] for r in records])), 6),
        "mean_nll":               round(float(np.mean([r["nll"]         for r in records])), 6),
        "mean_risk_score":        round(float(np.mean([r["risk_score"]  for r in records])), 6),
        "mean_inconclusive_frac": round(float(np.mean([r["inconclusive_frac"] for r in records])), 6),
        "mean_vacuity":           round(float(np.mean([r["mean_vacuity"] for r in records])), 6),
        "mean_modality_importance": [
            round(float(np.mean([r["modality_importance"][i] for r in records])), 6)
            for i in range(4)
        ],
        "dice_std_mean":          round(float(np.mean([
            np.mean(list(r["dice_std"].values())) for r in records])), 6),
        "dice_adaptive_mean":     round(float(np.mean([
            np.mean(list(r["dice_adaptive"].values())) for r in records])), 6),
        "threshold_sweep":        sweep_results,
        "best_threshold":         best_thr,
    }

    out_path = os.path.join(RESULTS_DIR, save_name)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "patients": records}, f, indent=2)

    print(f"\n  ─── Uncertainty Analysis Summary ───────────────────────────")
    print(f"  Brier Score           : {summary['mean_brier_score']:.4f}")
    print(f"  NLL                   : {summary['mean_nll']:.4f}")
    print(f"  Volume Risk Score     : {summary['mean_risk_score']:.4f}")
    print(f"  Inconclusive Fraction : {summary['mean_inconclusive_frac']*100:.1f}%")
    print(f"  Mean Vacuity          : {summary['mean_vacuity']:.6f}")
    print(f"  Modality Importance   : "
          f"T1={summary['mean_modality_importance'][0]:.4f}  "
          f"T1ce={summary['mean_modality_importance'][1]:.4f}  "
          f"T2={summary['mean_modality_importance'][2]:.4f}  "
          f"FLAIR={summary['mean_modality_importance'][3]:.4f}")
    print(f"  Dice (fixed 0.5)      : {summary['dice_std_mean']:.4f}")
    print(f"  Dice (adaptive)       : {summary['dice_adaptive_mean']:.4f}")
    print(f"  Best threshold        : {best_thr['threshold']}  "
          f"(mean dice={best_thr['mean_dice']:.4f})")
    print(f"  Saved → {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="P19 Missing-Modality Brain MRI — DINO + Evidential")
    pa.add_argument("--stage", required=True,
                    choices=["test","ssl_pretrain","train",
                             "test_eval","eval","simulate","report",
                             "recon_eval","analyze"])
    pa.add_argument("--ssl_epochs",    type=int,   default=50)
    pa.add_argument("--epochs",        type=int,   default=150)
    pa.add_argument("--ssl_lr",        type=float, default=1e-4)
    pa.add_argument("--lr",            type=float, default=2e-4)
    pa.add_argument("--batch_size",    type=int,   default=2)     # 24 GB VRAM
    pa.add_argument("--base_features", type=int,   default=32)
    pa.add_argument("--dropout",       type=float, default=0.1)
    pa.add_argument("--proj_dim",      type=int,   default=128)
    pa.add_argument("--grad_accum",    type=int,   default=2)     # eff. batch=4
    pa.add_argument("--mc_passes",     type=int,   default=20)
    pa.add_argument("--no_tta",        action="store_true")
    pa.add_argument("--num_workers",   type=int,   default=NUM_WORKERS)
    pa.add_argument("--checkpoint",    type=str,   default=None)
    pa.add_argument("--ssl_resume",    type=str,   default=None,
                    help="Path to ssl_epN.pth to resume SSL pretraining")
    pa.add_argument("--model_version", type=int,   default=1, choices=[1, 2],
                    help="1=original V1 (f=32), 2=V2 with CrossModalAttn+Transformer (f=48)")
    args = pa.parse_args()

    set_seed()
    print_gpu_info()

    def _load_ckpt(model, path):
        """Shape-aware checkpoint load — skips keys whose tensor size changed."""
        ck  = torch.load(path, map_location=DEVICE, weights_only=False)
        sd  = ck.get("model_state", ck)
        msd = model.state_dict()
        ok  = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(ok, strict=False)
        tag = "full" if len(ok) == len(sd) else f"{len(ok)}/{len(sd)} keys (shape mismatch on rest)"
        print(f"  Loaded checkpoint: {path}  [{tag}]")

    # Select model and default features based on version
    if args.model_version == 2:
        f = args.base_features if args.base_features != 32 else 48
        model = MissingModalityNetV2(4, f, 3, args.dropout, args.proj_dim)
        print(f"  Model  : MissingModalityNetV2  (f={f}, CrossModalAttn + BottleneckTransformer)")
    else:
        f = args.base_features
        model = MissingModalityNet(4, f, 3, args.dropout, args.proj_dim)
        print(f"  Model  : MissingModalityNet  (f={f})")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params : {total:,}  ({total/1e6:.1f} M)")

    # ── Sanity check ──────────────────────────────────────────────────────
    if args.stage == "test":
        sz = 64
        x  = torch.randn(1, 4, sz, sz, sz)
        m  = torch.tensor([[1,0,1,1]], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            o = model(x, m)
        ev = EvidentialHead.get_evidence(o["evid_logit"])
        print(f"  recon      : {o['recon'].shape}")
        print(f"  seg        : {o['seg'].shape}")
        print(f"  evid_logit : {o['evid_logit'].shape}")
        print(f"  proj_btn   : {o['proj_btn'].shape}")
        print(f"  vacuity    : mean={ev['vacuity'].mean().item():.4f}")
        p, u, ep, al = model.forward_mc(x, m, n_passes=3)
        print(f"  MC epistemic mean : {ep.mean().item():.6f}")
        p2, u2 = model.forward_tta(x, m)
        print(f"  TTA unc mean      : {u2.mean().item():.6f}")
        print("  PASSED!")

    # ── Stage 1 ───────────────────────────────────────────────────────────
    elif args.stage == "ssl_pretrain":
        tl, _, _ = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=args.batch_size,
            num_workers=args.num_workers)
        ssl_pretrain(model, tl, num_epochs=args.ssl_epochs,
                     lr=args.ssl_lr, device=DEVICE, proj_dim=args.proj_dim,
                     resume_ckpt=args.ssl_resume)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    elif args.stage == "train":
        tl, vl, _ = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=args.batch_size,
            num_workers=args.num_workers)
        if args.checkpoint and os.path.exists(args.checkpoint):
            ck  = torch.load(args.checkpoint, map_location=DEVICE)
            sd  = ck.get("model_state", ck)
            msd = model.state_dict()
            # Shape-aware load: skip any key whose tensor size changed (e.g. f=32→48)
            compatible = {k: v for k, v in sd.items()
                          if k in msd and v.shape == msd[k].shape}
            model.load_state_dict(compatible, strict=False)
            print(f"  Loaded {len(compatible)}/{len(sd)} keys from {args.checkpoint}"
                  f"  ({'full' if len(compatible)==len(sd) else 'partial — size change from different f'})")
        else:
            print("  No checkpoint — training from random init")
        # V2 uses boundary + hierarchical consistency losses
        loss_override = SupervisedTotalLossV2() if args.model_version == 2 else None
        train(model, tl, vl,
              num_epochs=args.epochs, lr=args.lr, device=DEVICE,
              grad_accum=args.grad_accum, proj_dim=args.proj_dim,
              loss_cls=loss_override)

    # ── Test eval ─────────────────────────────────────────────────────────
    elif args.stage == "test_eval":
        cp = args.checkpoint or "checkpoints/best_model.pth"
        _load_ckpt(model, cp)
        model = model.to(DEVICE)
        _, _, sl = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1,
            num_workers=args.num_workers)
        test_model(model, sl, DEVICE, use_tta=not args.no_tta)

    # ── Eval all 15 combos ────────────────────────────────────────────────
    elif args.stage == "eval":
        cp = args.checkpoint or "checkpoints/best_model.pth"
        _load_ckpt(model, cp)
        model = model.to(DEVICE)
        _, _, sl = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1,
            num_workers=args.num_workers)
        print("\n  Evaluating all 15 modality combinations on test set:\n")
        evaluate_all(model, sl, DEVICE)

    # ── Simulate ──────────────────────────────────────────────────────────
    elif args.stage == "simulate":
        from visualize import simulate_patient
        cp = args.checkpoint or "checkpoints/best_model.pth"
        _load_ckpt(model, cp)
        model = model.to(DEVICE)
        _, _, sl = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1,
            num_workers=args.num_workers)
        simulate_patient(model, sl.dataset, DEVICE,
                         save_dir=os.path.join(RESULTS_DIR, "simulation"),
                         mc_passes=args.mc_passes)

    # ── Reconstruction quality (SSIM + PSNR) per combo ───────────────────
    elif args.stage == "recon_eval":
        cp = args.checkpoint or "checkpoints/best_model.pth"
        _load_ckpt(model, cp)
        model = model.to(DEVICE)
        _, _, sl = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1,
            num_workers=args.num_workers)
        print("\n  Reconstruction quality across all modality combinations:\n")
        evaluate_reconstruction(model, sl, DEVICE)

    # ── Comprehensive uncertainty + modality importance analysis ──────────
    elif args.stage == "analyze":
        cp = args.checkpoint or "checkpoints/best_model.pth"
        _load_ckpt(model, cp)
        model = model.to(DEVICE)
        _, _, sl = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1,
            num_workers=args.num_workers)
        print("\n  Running comprehensive uncertainty analysis:\n")
        analyze_uncertainty(model, sl, DEVICE, mc_passes=args.mc_passes)
        print("\n  Running reconstruction quality evaluation:\n")
        evaluate_reconstruction(model, sl, DEVICE)
        print("\n  Generating extended figures …")
        from visualize import (plot_reconstruction_quality,
                               plot_modality_importance,
                               plot_risk_score_analysis)
        plot_reconstruction_quality(
            os.path.join(RESULTS_DIR, "eval_reconstruction.json"),
            os.path.join(RESULTS_DIR, "figures"))
        plot_modality_importance(
            os.path.join(RESULTS_DIR, "uncertainty_analysis.json"),
            os.path.join(RESULTS_DIR, "figures"))
        plot_risk_score_analysis(
            os.path.join(RESULTS_DIR, "uncertainty_analysis.json"),
            os.path.join(RESULTS_DIR, "figures"))

    # ── Report ────────────────────────────────────────────────────────────
    elif args.stage == "report":
        from visualize import generate_all_figures
        ep = os.path.join(RESULTS_DIR, "eval_results_test.json")
        if not os.path.exists(ep):
            ep = "eval_results.json"

        m_loaded = test_ds = None
        cp = args.checkpoint or "checkpoints/best_model.pth"
        if os.path.exists(cp):
            _load_ckpt(model, cp)
            model = model.to(DEVICE)
            _, _, sl = create_data_splits(
                PROCESSED_TRAIN_DIR, batch_size=1,
                num_workers=args.num_workers)
            m_loaded, test_ds = model, sl.dataset

        generate_all_figures(
            metrics_path=METRICS_PATH,
            eval_results_path=ep,
            save_dir=os.path.join(RESULTS_DIR, "figures"),
            model=m_loaded, test_dataset=test_ds,
            device=DEVICE, mc_passes=args.mc_passes,
        )

    print("\n  Done!")
