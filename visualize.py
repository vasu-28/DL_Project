"""
================================================================================
VISUALIZE.PY  (v2 — with Uncertainty)
================================================================================
Provides
  simulate_patient()              — reconstruction + seg + MC-Dropout uncertainty
  plot_training_curves()          — SSL / train / val / test loss & dice
  plot_uncertainty_calibration()  — reliability diagram (ECE)
  plot_uncertainty_error_corr()   — scatter: uncertainty vs dice error
  plot_eval_heatmap()             — 15-combo dice heatmap
  plot_modality_bar()             — dice bar-chart grouped by # present mods
  plot_combo_table()              — full results table (report appendix)
  generate_all_figures()          — runs all of the above

Standalone usage
  python visualize.py --metrics results/training_metrics.json \
                      --eval    results/eval_results_test.json \
                      --outdir  results/figures
================================================================================
"""

import os, json, argparse, warnings
import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import torch

try:
    from train import EvidentialHead
    _HAS_EVIDENTIAL = True
except ImportError:
    EvidentialHead = None
    _HAS_EVIDENTIAL = False

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MOD_NAMES   = ["T1", "T1ce", "T2", "FLAIR"]
SEG_NAMES   = ["WT", "TC", "ET"]
SEG_FULL    = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
SEG_COLORS  = ["#E74C3C", "#2ECC71", "#3498DB"]
SEG_ALPHAS  = [0.55, 0.50, 0.65]

DPI   = 200
STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.facecolor":  "white",
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tumor_slice(seg):
    """Axial index with most cumulative tumour voxels."""
    per_z = seg.sum(axis=tuple(range(seg.ndim - 1)))
    return int(np.argmax(per_z)) if per_z.max() > 0 else seg.shape[-1] // 2


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path}")


def _overlay_seg(ax, mri, seg3ch):
    """Show MRI in grey and overlay WT/TC/ET as coloured masks."""
    vmin = np.percentile(mri[mri != 0], 1) if mri.any() else 0
    vmax = np.percentile(mri, 99)
    ax.imshow(mri.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    for c, (col, alp) in enumerate(zip(SEG_COLORS, SEG_ALPHAS)):
        m = seg3ch[c]
        if m.max() == 0:
            continue
        rgba = np.zeros((*m.shape, 4))
        r, g, b = int(col[1:3],16)/255, int(col[3:5],16)/255, int(col[5:],16)/255
        rgba[m > 0.5] = [r, g, b, alp]
        ax.imshow(rgba.transpose(1,0,2), origin="lower")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
#  1.  PATIENT SIMULATION  (reconstruction + segmentation + uncertainty)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_patient(model, test_dataset, device,
                     save_dir="results/simulation", mc_passes=20):
    """
    Picks the test patient with the most tumour and runs three
    missing-modality scenarios.  For each scenario produces:

      fig_reconstruction.png    — input | reconstructed | GT per modality
      fig_segmentation.png      — GT | Pred | TP/FP/FN overlay
      fig_uncertainty.png       — epistemic / aleatoric / total uncertainty maps
      fig_uncertainty_proxy.png — per-modality reconstruction-error heatmap
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # ── Pick patient with most tumour ────────────────────────────────────
    print("  Scanning test set for best patient …")
    best_idx, best_tv = 0, -1
    for idx in range(min(len(test_dataset), 100)):
        try:
            tv = float(test_dataset[idx]["seg"].sum())
            if tv > best_tv:
                best_tv, best_idx = tv, idx
        except Exception:
            continue

    sample  = test_dataset[best_idx]
    orig    = sample["original"].numpy()     # (4, H, W, D)
    seg_gt  = sample["seg"].numpy()          # (3, H, W, D)
    name    = sample["patient_name"]
    z       = _tumor_slice(seg_gt)
    print(f"  Patient: {name}   tumour slice z={z}")

    # ── Three scenarios ──────────────────────────────────────────────────
    scenarios = [
        {"missing": [1],       "label": "T1ce missing\n(T1, T2, FLAIR present)"},
        {"missing": [2, 3],    "label": "T2 + FLAIR missing\n(T1, T1ce present)"},
        {"missing": [0, 1, 2], "label": "Only FLAIR present\n(T1, T1ce, T2 missing)"},
    ]

    has_evidential = _HAS_EVIDENTIAL

    results = []
    for sc in scenarios:
        inp  = orig.copy()
        mask = np.ones(4, dtype=np.float32)
        for i in sc["missing"]:
            inp[i] = 0.0; mask[i] = 0.0

        vt = torch.from_numpy(inp ).unsqueeze(0).to(device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(device)

        # Standard forward
        model.eval()
        with torch.no_grad():
            out   = model(vt, mt)
            recon = out["recon"].squeeze(0).cpu().numpy()

        # Evidential uncertainty (vacuity = principled epistemic)
        if has_evidential and "evid_logit" in out:
            ev      = EvidentialHead.get_evidence(out["evid_logit"])
            vacuity = ev["vacuity"].squeeze(0).cpu().numpy()   # (3,H,W,D)
            ev_prob = ev["prob"]   .squeeze(0).cpu().numpy()
        else:
            vacuity = np.zeros((3, *inp.shape[1:]), dtype=np.float32)
            ev_prob = torch.sigmoid(out["seg"]).squeeze(0).cpu().numpy()

        # MC Dropout (epistemic + aleatoric decomposition)
        mp, total_unc, epist, aleat = model.forward_mc(vt, mt, n_passes=mc_passes)
        mp        = mp       .squeeze(0).cpu().numpy()
        total_unc = total_unc.squeeze(0).cpu().numpy()
        epist     = epist    .squeeze(0).cpu().numpy()
        aleat     = aleat    .squeeze(0).cpu().numpy()

        # TTA prediction (most accurate for report)
        tta_prob, tta_unc = model.forward_tta(vt, mt)
        tta_prob = tta_prob.squeeze(0).cpu().numpy()
        tta_unc  = tta_unc .squeeze(0).cpu().numpy()

        results.append({
            "label":    sc["label"],
            "missing":  sc["missing"],
            "input":    inp,
            "recon":    recon,
            "seg_pred": (tta_prob > 0.5).astype(np.float32),  # TTA is most accurate
            "seg_mean": tta_prob,
            "vacuity":  vacuity,      # evidential: principled epistemic
            "epistemic":epist,        # MC Dropout: inter-pass variance
            "aleatoric":aleat,        # MC Dropout: mean predicted vacuity
            "total_unc":total_unc,    # MC Dropout: combined
            "tta_unc":  tta_unc,      # TTA variance
        })

    flair_sl = orig[3, :, :, z]    # FLAIR used as background for seg overlays

    # ════════════════════════════════════════════════════════════════════
    # Figure A — Reconstruction  (input | recon | GT) × 4 modalities × 3 scenarios
    # ════════════════════════════════════════════════════════════════════
    n_sc  = len(scenarios)
    fig_a, axes_a = plt.subplots(n_sc * 3, 4, figsize=(4*3.8, n_sc*3*3.2))
    fig_a.suptitle(
        f"Missing Modality Reconstruction  |  {name}  |  axial z={z}",
        fontsize=13, fontweight="bold", y=1.005)

    for si, p in enumerate(results):
        for row_off, (data, row_lbl) in enumerate([
            (p["input"],  "Input\n(masked)"),
            (p["recon"],  "Reconstructed"),
            (orig,        "Ground Truth"),
        ]):
            r = si * 3 + row_off
            for ci in range(4):
                ax = axes_a[r, ci]
                sl = data[ci, :, :, z]
                vmin = np.percentile(sl[sl != 0], 1) if sl.any() else 0
                vmax = np.percentile(sl, 99) if sl.any() else 1
                ax.imshow(sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                ax.axis("off")
                if r == 0:
                    ax.set_title(MOD_NAMES[ci], fontsize=11, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(row_lbl, fontsize=8, labelpad=4, va="center")
                    ax.yaxis.set_visible(True)
                    ax.tick_params(left=False, labelleft=False)
                # Red border on missing modalities
                if ci in p["missing"] and row_off in [0, 1]:
                    for sp in ax.spines.values():
                        sp.set_visible(True)
                        sp.set_edgecolor("#E74C3C"); sp.set_linewidth(2.5)
        # Scenario annotation
        axes_a[si*3+1, 0].annotate(
            p["label"], xy=(-0.55, 0.5), xycoords="axes fraction",
            fontsize=9, color="#2C3E50", ha="right", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECF0F1", ec="#BDC3C7"))
    _save(fig_a, os.path.join(save_dir, "fig_reconstruction.png"))

    # ════════════════════════════════════════════════════════════════════
    # Figure B — Segmentation  (GT | Pred | TP/FP/FN)
    # ════════════════════════════════════════════════════════════════════
    fig_b, axes_b = plt.subplots(n_sc, 3, figsize=(13, n_sc*4.5))
    fig_b.suptitle(
        f"Segmentation Under Missing Modalities  |  {name}  |  axial z={z}",
        fontsize=13, fontweight="bold", y=1.005)

    for si, p in enumerate(results):
        for ci, (data, title) in enumerate([
            (seg_gt,        "Ground Truth"),
            (p["seg_pred"], "MC-Dropout Mean Pred"),
            (None,          "TP / FP / FN Overlay"),
        ]):
            ax = axes_b[si, ci]; ax.axis("off")
            if si == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if ci < 2:
                _overlay_seg(ax, flair_sl, data[:, :, :, z])
            else:
                vmin = np.percentile(flair_sl, 1)
                vmax = np.percentile(flair_sl, 99)
                ax.imshow(flair_sl.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                for c_idx in range(3):
                    gt_m   = seg_gt[c_idx, :, :, z]
                    pr_m   = p["seg_pred"][c_idx, :, :, z]
                    tp = (gt_m > 0.5) & (pr_m > 0.5)
                    fp = (pr_m > 0.5) & ~(gt_m > 0.5)
                    fn = (gt_m > 0.5) & ~(pr_m > 0.5)
                    for reg, col in [(tp, "#2ECC71"), (fp, "#E74C3C"), (fn, "#F39C12")]:
                        if reg.any():
                            rgba = np.zeros((*reg.shape, 4))
                            r_, g_, b_ = int(col[1:3],16)/255, int(col[3:5],16)/255, int(col[5:],16)/255
                            rgba[reg] = [r_, g_, b_, 0.6]
                            ax.imshow(rgba.transpose(1,0,2), origin="lower")
                ax.legend(handles=[
                    mpatches.Patch(color="#2ECC71", label="True Positive"),
                    mpatches.Patch(color="#E74C3C", label="False Positive"),
                    mpatches.Patch(color="#F39C12", label="False Negative"),
                ], loc="lower right", fontsize=7, framealpha=0.8)
        axes_b[si, 0].set_ylabel(p["label"], fontsize=9, fontweight="bold",
                                  rotation=0, labelpad=95, va="center")
        axes_b[si, 0].yaxis.set_visible(True)
        axes_b[si, 0].tick_params(left=False, labelleft=False)
    _save(fig_b, os.path.join(save_dir, "fig_segmentation.png"))

    # ════════════════════════════════════════════════════════════════════
    # Figure C — Uncertainty Maps  (epistemic / aleatoric / total × 3 classes × 3 scenarios)
    # ════════════════════════════════════════════════════════════════════
    fig_c = plt.figure(figsize=(14, n_sc * 4.5))
    fig_c.suptitle(
        f"MC-Dropout Uncertainty Maps  |  {name}  |  axial z={z}\n"
        "Columns: WT · TC · ET for each uncertainty type",
        fontsize=13, fontweight="bold")
    outer = gridspec.GridSpec(n_sc, 1, hspace=0.45)

    unc_types = [("vacuity",   "Evidential Vacuity\n(principled epistemic)", "Blues"),
                 ("epistemic", "MC-Dropout Epistemic\n(model uncertainty)",   "Oranges"),
                 ("aleatoric", "MC-Dropout Aleatoric\n(data uncertainty)",    "Reds"),
                 ("tta_unc",   "TTA Variance\n(augmentation sensitivity)",    "Purples")]

    for si, p in enumerate(results):
        inner = gridspec.GridSpecFromSubplotSpec(
            len(unc_types), 3, subplot_spec=outer[si], hspace=0.05, wspace=0.05)

        for ui, (key, ulbl, cmap) in enumerate(unc_types):
            unc_vol = p[key]   # (3, H, W, D)
            for ci, cls in enumerate(SEG_NAMES):
                ax  = fig_c.add_subplot(inner[ui, ci])
                sl  = unc_vol[ci, :, :, z]
                im  = ax.imshow(sl.T, cmap=cmap, origin="lower",
                                vmin=0, vmax=max(sl.max(), 1e-9))
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format="%.3f")
                if ui == 0:
                    ax.set_title(cls, fontsize=10, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(ulbl, fontsize=8.5, rotation=90, labelpad=4)
                    ax.yaxis.set_visible(True)
                    ax.tick_params(left=False, labelleft=False)

        # Scenario label above each group
        fig_c.text(0.01, outer[si].get_position(fig_c).y1,
                   p["label"].replace("\n", "  "),
                   fontsize=9, color="#2C3E50", fontweight="bold", va="bottom")
    _save(fig_c, os.path.join(save_dir, "fig_uncertainty.png"))

    # ════════════════════════════════════════════════════════════════════
    # Figure D — Reconstruction Error Heatmap  (uncertainty proxy)
    # ════════════════════════════════════════════════════════════════════
    fig_d, axes_d = plt.subplots(n_sc, 4, figsize=(14, n_sc*3.5))
    fig_d.suptitle(
        f"Per-Modality Reconstruction Error  |  {name}  |  axial z={z}",
        fontsize=13, fontweight="bold")
    for si, p in enumerate(results):
        err = np.abs(p["recon"] - orig)
        for ci in range(4):
            ax = axes_d[si, ci]
            sl = err[ci, :, :, z]
            im = ax.imshow(sl.T, cmap="hot", origin="lower", vmin=0, vmax=sl.max())
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if si == 0:
                ax.set_title(MOD_NAMES[ci], fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(p["label"], fontsize=8, rotation=90, labelpad=4)
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)
    _save(fig_d, os.path.join(save_dir, "fig_uncertainty_proxy.png"))

    print(f"\n  Simulation figures → {save_dir}")
    for fn in ["fig_reconstruction.png", "fig_segmentation.png",
               "fig_uncertainty.png", "fig_uncertainty_proxy.png"]:
        print(f"    {fn}")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  TRAINING CURVES  (SSL + supervised + test)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(metrics_path, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(metrics_path):
        print(f"  [WARN] {metrics_path} not found"); return

    with open(metrics_path) as f:
        m = json.load(f)

    ssl_ep  = m.get("ssl_epochs",   [])
    tr_ep   = m.get("train_epochs", [])
    val_ep  = m.get("val_epochs",   [])
    test_r  = m.get("test",         None)

    with plt.style.context(STYLE):

        # ── Figure 1: SSL pretraining loss ───────────────────────────────
        if ssl_ep:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            fig.suptitle("Stage 1 — SSL Pretraining Losses", fontsize=13, fontweight="bold")
            ep = [e["epoch"] for e in ssl_ep]
            axes[0].plot(ep, [e["total"]                       for e in ssl_ep], label="Total",   color="#2980B9", lw=2)
            axes[0].plot(ep, [e.get("dino_btn",0)+e.get("dino_dec",0) for e in ssl_ep], label="DINO",    color="#E67E22", lw=1.6, ls="--")
            axes[0].plot(ep, [e.get("nce_btn",0)+e.get("nce_dec",0)   for e in ssl_ep], label="InfoNCE", color="#E74C3C", lw=1.4, ls="-.")
            axes[0].plot(ep, [e["recon"]                       for e in ssl_ep], label="Recon",   color="#27AE60", lw=1.6, ls=":")
            axes[0].set_xlabel("SSL Epoch"); axes[0].set_ylabel("Loss")
            axes[0].set_title("SSL Loss Components"); axes[0].legend(fontsize=9)
            axes[1].plot(ep, [e.get("consistency", 0) for e in ssl_ep], color="#8E44AD", lw=1.8)
            axes[1].set_xlabel("SSL Epoch"); axes[1].set_ylabel("Consistency Loss")
            axes[1].set_title("View Consistency (cosine distance)")
            _save(fig, os.path.join(save_dir, "fig_ssl_curves.png"))

        if not tr_ep:
            print("  [WARN] No supervised train epochs"); return

        t_ep   = [e["epoch"] for e in tr_ep]
        t_loss = [e["loss"]  for e in tr_ep]
        t_recon= [e.get("recon_loss", 0)  for e in tr_ep]
        t_unc  = [e.get("evid_loss", e.get("aleatoric", 0)) for e in tr_ep]
        t_con  = [e.get("consistency", e.get("contrastive", 0)) for e in tr_ep]
        t_wt   = [e["WT"] for e in tr_ep]
        t_tc   = [e["TC"] for e in tr_ep]
        t_et   = [e["ET"] for e in tr_ep]
        t_mean = [e["mean_dice"] for e in tr_ep]
        t_dp   = [e.get("drop_prob", 0) for e in tr_ep]

        v_ep   = [e["epoch"]     for e in val_ep]
        v_loss = [e["loss"]      for e in val_ep]
        v_wt   = [e["WT"]        for e in val_ep]
        v_tc   = [e["TC"]        for e in val_ep]
        v_et   = [e["ET"]        for e in val_ep]
        v_mean = [e["mean_dice"] for e in val_ep]

        # ── Figure 2: Loss curves ─────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Stage 2 — Supervised Fine-Tuning Losses", fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(t_ep, t_loss,  label="Train Total",      color="#2980B9", lw=1.8)
        ax.plot(t_ep, t_recon, label="Train Recon",      color="#27AE60", lw=1.4, ls="--")
        ax.plot(t_ep, t_unc,   label="Train Aleatoric",  color="#9B59B6", lw=1.4, ls=":")
        ax.plot(t_ep, t_con,   label="Train Contrastive",color="#E67E22", lw=1.2, ls="-.")
        if v_loss:
            ax.plot(v_ep, v_loss, label="Val Total", color="#E74C3C", lw=2.2,
                    marker="o", markersize=4)
        if test_r and "loss" in test_r:
            ax.axhline(test_r["loss"], color="#2C3E50", lw=1.5, ls="--",
                       label=f"Test={test_r['loss']:.4f}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Loss Curves"); ax.legend(fontsize=8.5)

        ax2 = axes[0].twinx()
        ax2.plot(t_ep, t_dp, color="#BDC3C7", lw=1.2, ls="-.", alpha=0.7)
        ax2.set_ylabel("Modality Dropout Prob", color="#BDC3C7", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#BDC3C7")
        ax2.set_ylim(0, 1)
        ax2.spines["right"].set_visible(True)

        ax = axes[1]
        ax.plot(t_ep, t_mean, label="Train Mean Dice", color="#2980B9", lw=1.8)
        if v_mean:
            ax.plot(v_ep, v_mean, label="Val Mean Dice", color="#E74C3C", lw=2.2,
                    marker="o", markersize=4)
        if test_r:
            ax.axhline(test_r["mean_dice"], color="#2C3E50", lw=1.5, ls="--",
                       label=f"Test={test_r['mean_dice']:.4f}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Dice")
        ax.set_title("Mean Dice (WT+TC+ET)/3"); ax.set_ylim(0, 1); ax.legend(fontsize=9)
        _save(fig, os.path.join(save_dir, "fig_loss_curves.png"))

        # ── Figure 3: Per-class Dice ──────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle("Per-Class Dice Over Training", fontsize=13, fontweight="bold")
        for i, (cls, tr, va, col) in enumerate([
            ("WT", t_wt, v_wt, "#E74C3C"),
            ("TC", t_tc, v_tc, "#2ECC71"),
            ("ET", t_et, v_et, "#3498DB"),
        ]):
            ax = axes[i]
            ax.plot(t_ep, tr, label=f"Train {cls}", color=col, lw=1.8, alpha=0.85)
            if va:
                ax.plot(v_ep, va, label=f"Val {cls}", color=col, lw=2.2,
                        ls="--", marker="o", markersize=4)
            if test_r:
                ax.axhline(test_r[cls], color="#2C3E50", lw=1.5, ls="-.",
                           label=f"Test {cls}={test_r[cls]:.4f}")
            ax.set_xlabel("Epoch")
            if i == 0:
                ax.set_ylabel("Dice Score")
            ax.set_title(SEG_FULL[i]); ax.set_ylim(0, 1); ax.legend(fontsize=8.5)
        _save(fig, os.path.join(save_dir, "fig_dice_per_class.png"))

        # ── Figure 4: Aleatoric uncertainty over training ─────────────────
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(t_ep, t_unc, color="#9B59B6", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Aleatoric NLL Loss")
        ax.set_title("Aleatoric Uncertainty (NLL) During Fine-Tuning\n"
                     "Decreasing trend = model learns to be calibrated")
        fig.suptitle("Uncertainty Learning Curve", fontsize=13, fontweight="bold")
        _save(fig, os.path.join(save_dir, "fig_aleatoric_curve.png"))

    print(f"  Training curves → {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  3.  UNCERTAINTY CALIBRATION  (reliability diagram)
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_calibration(model, test_dataset, device,
                                  save_dir="results/figures",
                                  mc_passes=20, n_bins=10):
    """
    Reliability diagram: compare mean predicted confidence with observed accuracy.
    A well-calibrated model lies on the diagonal.
    Also computes Expected Calibration Error (ECE).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_conf, all_acc = [], []

    print(f"  Calibration: running TTA on up to 100 test patients …")
    n_check = min(len(test_dataset), 100)

    for idx in range(n_check):
        try:
            sample = test_dataset[idx]
        except Exception:
            continue
        if float(sample["seg"].sum()) == 0:
            continue

        orig = sample["original"].unsqueeze(0).to(device)
        mask = torch.ones(1, 4, device=device)
        seg  = sample["seg"].numpy()                          # (3, H, W, D)

        # Use TTA for most accurate confidence estimates
        mean_pred, _ = model.forward_tta(orig, mask)
        conf = mean_pred.squeeze(0).cpu().numpy()             # (3, H, W, D)

        # Flatten all classes and voxels
        all_conf.append(conf.ravel())
        all_acc .append(seg .ravel())

    if not all_conf:
        print("  [WARN] No valid patients for calibration"); return

    conf_flat = np.concatenate(all_conf)
    acc_flat  = np.concatenate(all_acc)

    # Bin into n_bins equal-width confidence bins
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_acc    = np.zeros(n_bins)
    bin_conf   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        idx_ = (conf_flat >= bins[b]) & (conf_flat < bins[b+1])
        if idx_.sum() > 0:
            bin_acc  [b] = acc_flat [idx_].mean()
            bin_conf [b] = conf_flat[idx_].mean()
            bin_counts[b] = idx_.sum()

    # ECE
    ece = (bin_counts / max(bin_counts.sum(), 1) *
           np.abs(bin_acc - bin_conf)).sum()

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Uncertainty Calibration  |  ECE = {ece:.4f}",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.bar(bins[:-1], bin_acc, width=1/n_bins, align="edge",
               alpha=0.65, color="#3498DB", label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.set_xlabel("Mean Predicted Confidence"); ax.set_ylabel("Observed Accuracy")
        ax.set_title("Reliability Diagram"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=9)
        ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes,
                fontsize=11, color="#E74C3C", fontweight="bold")

        # Confidence histogram
        ax = axes[1]
        ax.bar(bins[:-1], bin_counts / max(bin_counts.sum(), 1),
               width=1/n_bins, align="edge", alpha=0.7, color="#E67E22")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Fraction of Voxels")
        ax.set_title("Confidence Distribution"); ax.set_xlim(0, 1)

        _save(fig, os.path.join(save_dir, "fig_calibration.png"))

    print(f"  ECE = {ece:.4f}  → {os.path.join(save_dir, 'fig_calibration.png')}")
    return ece


# ─────────────────────────────────────────────────────────────────────────────
#  4.  UNCERTAINTY vs ERROR CORRELATION  (scatter + rank correlation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_error_corr(model, test_dataset, device,
                                 save_dir="results/figures",
                                 mc_passes=20):
    """
    For each test patient compute:
      uncertainty = mean total uncertainty across the predicted tumour region
      dice_error  = 1 - mean Dice across WT/TC/ET
    Plot as scatter and compute Spearman rank correlation.
    A good uncertainty model should show positive correlation.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    uncertainties = []
    errors        = []

    def _dice(pred, target, smooth=1e-5):
        p, t = pred.astype(float), target.astype(float)
        return (2*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth)

    print("  Computing uncertainty–error correlation …")
    n_check = min(len(test_dataset), 200)

    for idx in range(n_check):
        try:
            sample = test_dataset[idx]
        except Exception:
            continue
        if float(sample["seg"].sum()) == 0:
            continue

        orig = sample["original"].unsqueeze(0).to(device)
        mask = torch.ones(1, 4, device=device)
        seg  = sample["seg"].numpy()

        # Use evidential vacuity as uncertainty if available (most principled)
        model.eval()
        with torch.no_grad():
            out = model(orig, mask)
        if _HAS_EVIDENTIAL and hasattr(out, '__getitem__') and "evid_logit" in out:
            try:
                ev  = EvidentialHead.get_evidence(out["evid_logit"])
                mp  = ev["prob"]   .squeeze(0).cpu().numpy()
                unc = ev["vacuity"].squeeze(0).cpu().numpy()
            except Exception:
                mp  = torch.sigmoid(out["seg"]).squeeze(0).cpu().numpy()
                unc = torch.zeros_like(out["seg"]).squeeze(0).cpu().numpy()
        else:
            _, total_unc, _, _ = model.forward_mc(orig, mask, n_passes=mc_passes)
            mp  = torch.sigmoid(out["seg"]).squeeze(0).cpu().numpy()
            unc = total_unc.squeeze(0).cpu().numpy()

        # Mean uncertainty in predicted positive region
        pred_bin    = (mp > 0.5).astype(float)
        roi_mask    = pred_bin.sum(0) > 0
        mean_unc    = float(unc.mean(0)[roi_mask].mean()) if roi_mask.any() else float(unc.mean())

        # Dice error
        dice_scores = [_dice((mp[c] > 0.5), seg[c]) for c in range(3)]
        dice_error  = 1.0 - np.mean(dice_scores)

        uncertainties.append(mean_unc)
        errors.append(dice_error)

    if len(uncertainties) < 5:
        print("  [WARN] Too few valid patients for correlation analysis"); return

    unc_arr = np.array(uncertainties)
    err_arr = np.array(errors)

    rho, pval = spearmanr(unc_arr, err_arr)

    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Uncertainty–Error Correlation  |  Spearman ρ={rho:.3f}  p={pval:.4f}",
            fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.scatter(unc_arr, err_arr, alpha=0.6, s=30, c="#3498DB", edgecolors="none")
        # Trend line
        z = np.polyfit(unc_arr, err_arr, 1)
        xr = np.linspace(unc_arr.min(), unc_arr.max(), 100)
        ax.plot(xr, np.polyval(z, xr), "r--", lw=1.8,
                label=f"Linear fit  ρ={rho:.3f}")
        ax.set_xlabel("Mean Total Uncertainty (MC-Dropout)")
        ax.set_ylabel("Dice Error (1 − Mean Dice)")
        ax.set_title("Per-Patient Uncertainty vs. Segmentation Error")
        ax.legend(fontsize=9)

        # Sorted bar chart of uncertainties coloured by error
        ax = axes[1]
        order  = np.argsort(unc_arr)
        colors = plt.cm.RdYlGn_r(err_arr[order] / max(err_arr.max(), 1e-9))
        ax.bar(range(len(order)), unc_arr[order], color=colors, alpha=0.85)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                    norm=plt.Normalize(0, err_arr.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Dice Error")
        ax.set_xlabel("Patient (sorted by uncertainty)")
        ax.set_ylabel("Total Uncertainty")
        ax.set_title("Patients Sorted by Uncertainty\n(colour = Dice error)")

        _save(fig, os.path.join(save_dir, "fig_uncertainty_correlation.png"))

    print(f"  Spearman ρ={rho:.3f}  →  fig_uncertainty_correlation.png")
    return rho, pval


# ─────────────────────────────────────────────────────────────────────────────
#  5.  EVALUATION HEATMAP  (15 modality combos)
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval_heatmap(eval_results_path, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(eval_results_path):
        print(f"  [WARN] {eval_results_path} not found"); return

    with open(eval_results_path) as f:
        results = json.load(f)
    results = sorted(results, key=lambda r: (r["n_present"], r["combo"]))

    labels    = [r["combo"] for r in results]
    n_present = [r["n_present"] for r in results]
    matrix    = np.array([[r["WT"], r["TC"], r["ET"], r["Mean"]] for r in results])
    cols      = ["WT", "TC", "ET", "Mean"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, max(10, len(results)*0.52)))
        fig.suptitle("Dice Score Heatmap — All 15 Modality Combinations (Test Set)",
                     fontsize=13, fontweight="bold", y=1.01)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        for i in range(len(results)):
            for j in range(4):
                v = matrix[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                        color="black" if 0.3 < v < 0.85 else "white",
                        fontweight="bold" if j == 3 else "normal")

        ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(results))); ax.set_yticklabels(labels, fontsize=9)
        ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")

        prev = n_present[0]
        for i, np_ in enumerate(n_present):
            if np_ != prev:
                ax.axhline(i-0.5, color="#2C3E50", lw=1.8, ls="--", alpha=0.6)
                prev = np_
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Dice Score")
        _save(fig, os.path.join(save_dir, "fig_eval_heatmap.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  6.  MODALITY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_modality_bar(eval_results_path, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(eval_results_path):
        print(f"  [WARN] {eval_results_path} not found"); return

    with open(eval_results_path) as f:
        results = json.load(f)

    groups = {1:[], 2:[], 3:[], 4:[]}
    for r in results:
        groups[r["n_present"]].append(r)

    x = np.arange(4); w = 0.22
    keys   = ["WT", "TC", "ET", "Mean"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB", "#2C3E50"]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Segmentation Dice vs. Number of Present Modalities",
                     fontsize=13, fontweight="bold")
        for ki, (key, col) in enumerate(zip(keys, colors)):
            means = [np.mean([r[key] for r in groups[np_]]) for np_ in [1,2,3,4]]
            stds  = [np.std ([r[key] for r in groups[np_]]) for np_ in [1,2,3,4]]
            bars  = ax.bar(x + (ki-1.5)*w, means, w, label=key, color=col, alpha=0.85,
                           yerr=stds, capsize=4)
            for bar, v in zip(bars, means):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["1 modality","2 modalities","3 modalities","4 modalities"], fontsize=10)
        ax.set_xlabel("Present Modalities", fontsize=11)
        ax.set_ylabel("Dice Score", fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.legend(title="Region", fontsize=9.5, title_fontsize=10)
        _save(fig, os.path.join(save_dir, "fig_modality_bar.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  7.  FULL RESULTS TABLE  (appendix)
# ─────────────────────────────────────────────────────────────────────────────

def plot_combo_table(eval_results_path, save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(eval_results_path):
        return

    with open(eval_results_path) as f:
        results = json.load(f)
    results = sorted(results, key=lambda r: (r["n_present"], r["combo"]))

    cell_data = [[r["combo"], str(r["n_present"]),
                  f"{r['WT']:.4f}", f"{r['TC']:.4f}",
                  f"{r['ET']:.4f}", f"{r['Mean']:.4f}"] for r in results]
    col_labels = ["Present Modalities", "# Present", "WT", "TC", "ET", "Mean"]

    with plt.style.context({"figure.facecolor":"white"}):
        fig, ax = plt.subplots(figsize=(12, 0.45*len(results)+2))
        ax.axis("off")
        fig.suptitle("Full Evaluation: All 15 Modality Combinations (Test Set)",
                     fontsize=13, fontweight="bold", y=0.98)
        tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.2, 1.4)
        for j in range(len(col_labels)):
            tbl[0,j].set_facecolor("#2C3E50")
            tbl[0,j].set_text_props(color="white", fontweight="bold")
        row_colors = {1:"#FDEDEC", 2:"#EBF5FB", 3:"#EAFAF1", 4:"#FEF9E7"}
        for i, r in enumerate(results, start=1):
            for j in range(len(col_labels)):
                tbl[i,j].set_facecolor(row_colors[r["n_present"]])
            v = r["Mean"]
            tbl[i,5].set_facecolor(
                "#27AE60" if v >= 0.8 else "#F39C12" if v >= 0.6 else "#E74C3C")
            tbl[i,5].set_text_props(color="white", fontweight="bold")
        _save(fig, os.path.join(save_dir, "fig_combo_table.png"))


# ─────────────────────────────────────────────────────────────────────────────
#  MASTER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(metrics_path, eval_results_path,
                         save_dir="results/figures",
                         model=None, test_dataset=None, device="cpu",
                         mc_passes=20):
    """
    Generate every report figure.
    Pass model + test_dataset to also produce calibration / correlation plots.
    """
    print(f"\n  Generating all figures → {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n[1/5] Training & SSL curves …")
    plot_training_curves(metrics_path, save_dir)

    print("\n[2/5] Evaluation heatmap …")
    plot_eval_heatmap(eval_results_path, save_dir)

    print("\n[3/5] Modality bar chart …")
    plot_modality_bar(eval_results_path, save_dir)

    print("\n[4/5] Combo table …")
    plot_combo_table(eval_results_path, save_dir)

    if model is not None and test_dataset is not None:
        print("\n[5/5] Uncertainty calibration + correlation …")
        plot_uncertainty_calibration(model, test_dataset, device,
                                      save_dir, mc_passes=mc_passes)
        plot_uncertainty_error_corr(model, test_dataset, device,
                                     save_dir, mc_passes=mc_passes)
    else:
        print("\n[5/5] Skipping uncertainty plots (no model/dataset provided)")

    print(f"\n  All figures saved to: {save_dir}")
    for fn in sorted(os.listdir(save_dir)):
        if fn.endswith(".png"):
            print(f"    {fn}")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Generate report figures")
    pa.add_argument("--metrics",  default="results/training_metrics.json")
    pa.add_argument("--eval",     default="results/eval_results_test.json")
    pa.add_argument("--outdir",   default="results/figures")
    pa.add_argument("--mode", default="all",
                    choices=["all","curves","heatmap","bar","table","calibration","correlation"])
    args = pa.parse_args()

    fn_map = {
        "curves":      lambda: plot_training_curves(args.metrics, args.outdir),
        "heatmap":     lambda: plot_eval_heatmap(args.eval, args.outdir),
        "bar":         lambda: plot_modality_bar(args.eval, args.outdir),
        "table":       lambda: plot_combo_table(args.eval, args.outdir),
    }

    if args.mode == "all":
        generate_all_figures(args.metrics, args.eval, args.outdir)
    elif args.mode in fn_map:
        fn_map[args.mode]()
    else:
        print(f"Mode '{args.mode}' requires --checkpoint; use train.py --stage report instead.")
