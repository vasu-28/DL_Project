"""
uncertainty_map.py
──────────────────
Generates a brain MRI visualization showing which regions have
high vs. low uncertainty (evidential vacuity) under three conditions:
  1. All 4 modalities present    → low uncertainty, accurate prediction
  2. T1ce missing                → higher uncertainty in TC/ET regions
  3. T1 only                     → highest uncertainty overall

Usage:
    python uncertainty_map.py [--patient_idx 0] [--out results/figures/fig_uncertainty_map.png]
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
from train import (
    MissingModalityNetV2, EvidentialHead,
    DEVICE, RESULTS_DIR, PROCESSED_TRAIN_DIR
)
CHECKPOINT_DIR = "checkpoints"
from brats_data_pipeline import BraTSDataset

FIG_BG   = "#0d1117"
FIG_SURF = "#161b22"


# ── Model loading ────────────────────────────────────────────────────────────

def _load_model():
    model = MissingModalityNetV2().to(DEVICE)
    path  = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    ck    = torch.load(path, map_location=DEVICE, weights_only=False)
    sd    = ck.get("model_state", ck)
    msd   = model.state_dict()
    ok    = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    model.load_state_dict(ok, strict=False)
    model.eval()
    print(f"  Loaded {path}  [{len(ok)}/{len(sd)} keys]")
    return model


# ── Inference ────────────────────────────────────────────────────────────────

def _infer(model, orig_tensor, present_indices):
    """Single forward pass. Returns (prob, pred_bin, vacuity) as numpy (3,D,H,W)."""
    mask = torch.zeros(1, 4, device=DEVICE)
    mask[0, present_indices] = 1.0
    with torch.no_grad():
        out = model(orig_tensor, mask)
    ev      = EvidentialHead.get_evidence(out["evid_logit"])
    prob    = ev["prob"]   .squeeze(0).cpu().numpy()   # (3,D,H,W)
    vacuity = ev["vacuity"].squeeze(0).cpu().numpy()   # (3,D,H,W)
    pred    = (prob > 0.5).astype(np.float32)
    del out, ev
    torch.cuda.empty_cache()
    return prob, pred, vacuity


# ── Dice helper ──────────────────────────────────────────────────────────────

def _dice(pred, gt, smooth=1e-5):
    p, g = pred.astype(float), gt.astype(float)
    return (2 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth)


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _seg_fill(ax, mri_sl, wt, tc, et, alpha=0.38):
    ax.imshow(mri_sl, cmap="gray", interpolation="bilinear")
    if wt.any():
        ax.contourf(wt, levels=[0.5, 1.5], colors=["#2ECC71"], alpha=alpha)
    if tc.any():
        ax.contourf(tc, levels=[0.5, 1.5], colors=["#F1C40F"], alpha=alpha + 0.07)
    if et.any():
        ax.contourf(et, levels=[0.5, 1.5], colors=["#E74C3C"], alpha=alpha + 0.15)
    ax.axis("off")


def _uncertainty_overlay(ax, mri_sl, vac_sl, vmax):
    ax.imshow(mri_sl, cmap="gray", interpolation="bilinear")
    norm_vac  = np.clip(vac_sl / vmax, 0, 1)
    rgba      = plt.cm.hot(norm_vac).copy()
    rgba[..., 3] = np.clip(norm_vac * 2.2, 0.0, 0.92)
    ax.imshow(rgba, interpolation="bilinear")
    ax.axis("off")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_idx", type=int, default=0,
                        help="Start searching from this test-set index")
    parser.add_argument("--out", default=os.path.join(
        RESULTS_DIR, "figures", "fig_uncertainty_map.png"))
    args = parser.parse_args()

    print("\n  ── Uncertainty Map Visualization ──")
    model = _load_model()

    # ── Pick a patient with visible tumour ───────────────────────────────────
    ds = BraTSDataset(PROCESSED_TRAIN_DIR, mode="test",
                      missing_strategy="none", augment=False)
    sample = None
    chosen_idx = args.patient_idx
    for i in range(args.patient_idx, min(args.patient_idx + 30, len(ds))):
        s = ds[i]
        if float(s["seg"].sum()) > 800:
            sample     = s
            chosen_idx = i
            break
    if sample is None:
        print("  No patient with sufficient tumour found. Try a different --patient_idx")
        return

    orig_np = sample["original"].numpy()   # (4, D, H, W)
    seg_np  = sample["seg"].numpy()        # (3, D, H, W)
    orig_t  = sample["original"].unsqueeze(0).to(DEVICE)

    # ── Find best axial slice (most tumour voxels) ───────────────────────────
    tumor_per_slice = seg_np.sum(axis=(0, 2, 3))
    best_sl         = int(tumor_per_slice.argmax())
    tumour_count    = int(tumor_per_slice.max())
    print(f"  Patient idx : {chosen_idx}")
    print(f"  Best slice  : {best_sl}  ({tumour_count} tumour voxels in slice)")

    # ── Three missing-modality conditions ────────────────────────────────────
    conditions = [
        ("All 4 Modalities",  [0, 1, 2, 3]),
        ("Missing T1ce",      [0, 2, 3]),
        ("T1 Only",           [0]),
    ]

    print("\n  Running inference …")
    results = []
    for name, mods in conditions:
        prob, pred, vacuity = _infer(model, orig_t, mods)
        d_wt = _dice(pred[0], seg_np[0])
        d_tc = _dice(pred[1], seg_np[1])
        d_et = _dice(pred[2], seg_np[2])
        mean_vac = float(vacuity.mean())
        print(f"  {name:25s}  Dice WT={d_wt:.3f} TC={d_tc:.3f} ET={d_et:.3f}  "
              f"Vacuity={mean_vac:.5f}")
        results.append({
            "name":   name,
            "mods":   mods,
            "pred":   pred,
            "vac":    vacuity,
            "d_wt":   d_wt, "d_tc": d_tc, "d_et": d_et,
            "mean_vac": mean_vac,
        })

    # ── Background MRI slice: use T1ce (channel 1) normalised ───────────────
    mri_ch   = orig_np[1, best_sl]
    lo, hi   = np.percentile(mri_ch, 1), np.percentile(mri_ch, 99)
    mri_sl   = np.clip((mri_ch - lo) / (hi - lo + 1e-8), 0, 1)
    seg_sl   = seg_np[:, best_sl]                  # (3, H, W)

    # Shared vmax across conditions for consistent colour scale
    all_vac_flat = np.concatenate([r["vac"].mean(0)[best_sl].flatten()
                                   for r in results])
    vmax = float(np.percentile(all_vac_flat, 99)) * 1.15 + 1e-6

    # ── Figure layout: rows = conditions, cols = [MRI, GT, Pred, Uncertainty] ─
    n_rows = len(conditions)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(15, 4.2 * n_rows),
                             facecolor=FIG_BG)
    fig.patch.set_facecolor(FIG_BG)

    col_titles = ["MRI (T1ce)", "Ground Truth", "Prediction", "Vacuity (Uncertainty ↑)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, color="white", fontsize=11,
                               fontweight="bold", pad=10)

    MOD_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

    for row, r in enumerate(results):
        pred_sl = r["pred"][:, best_sl]    # (3, H, W)
        vac_sl  = r["vac"].mean(0)[best_sl]   # mean across WT/TC/ET → (H, W)

        # Col 0 — plain MRI
        ax = axes[row, 0]
        ax.imshow(mri_sl, cmap="gray", interpolation="bilinear")
        ax.axis("off")

        # Row label (left side)
        mods_str   = " + ".join([MOD_NAMES[m] for m in r["mods"]])
        label_text = (f"{r['name']}\n({mods_str})\n\n"
                      f"WT={r['d_wt']:.3f}\n"
                      f"TC={r['d_tc']:.3f}\n"
                      f"ET={r['d_et']:.3f}\n"
                      f"Vacuity={r['mean_vac']:.4f}")
        ax.set_ylabel(label_text, color="white", fontsize=8.5,
                      rotation=0, ha="right", va="center", labelpad=10)

        # Col 1 — Ground truth
        _seg_fill(axes[row, 1], mri_sl, seg_sl[0], seg_sl[1], seg_sl[2])

        # Col 2 — Prediction
        _seg_fill(axes[row, 2], mri_sl, pred_sl[0], pred_sl[1], pred_sl[2])

        # Col 3 — Uncertainty heatmap
        _uncertainty_overlay(axes[row, 3], mri_sl, vac_sl, vmax)

    # ── Shared colorbar ──────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap="hot",
                               norm=mcolors.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.014, 0.76])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean Vacuity", color="white", fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.ax.set_facecolor(FIG_BG)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_els = [
        Patch(facecolor="#2ECC71", alpha=0.8, label="WT — Whole Tumour"),
        Patch(facecolor="#F1C40F", alpha=0.8, label="TC — Tumour Core"),
        Patch(facecolor="#E74C3C", alpha=0.9, label="ET — Enhancing Tumour"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=3,
               facecolor=FIG_SURF, edgecolor="#30363d",
               labelcolor="white", fontsize=9.5,
               bbox_to_anchor=(0.46, 0.005))

    # ── Title ────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Evidential Vacuity (Uncertainty) Increases as Modalities Are Removed\n"
        "Hot = high uncertainty  |  Dark = low uncertainty",
        color="white", fontsize=12, fontweight="bold", y=0.995
    )

    plt.subplots_adjust(left=0.17, right=0.92, top=0.96,
                        bottom=0.055, hspace=0.06, wspace=0.04)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"\n  Saved → {args.out}")


if __name__ == "__main__":
    main()
