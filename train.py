"""
================================================================================
PROJECT P19 — FINAL FIXED TRAIN.PY (v3)
================================================================================
Fixes from v2:
  - Uses 80/20 split of TRAINING data (validation set has no seg labels!)
  - Fixes zero-dice bug: validation now actually computes segmentation
  - Fixes mixed precision for PyTorch 2.1.x
  - Proper dice tracking during training
  
USAGE:
    python train.py --stage test
    python train.py --stage train --epochs 200 --batch_size 2
    python train.py --stage eval
================================================================================
"""

import os
import sys
import glob
import argparse
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from brats_data_pipeline import BraTSDataset, get_all_missing_combinations


# ============================================================================
#  CONFIG
# ============================================================================

PROCESSED_TRAIN_DIR = "processed/train"  # This has seg labels
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
#  MODEL COMPONENTS
# ============================================================================

class ResConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.act(self.norm2(self.conv2(x)))
        return x + res


class SEBlock3D(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class ModalityAwareEncoder(nn.Module):
    def __init__(self, num_mods=4, f=32, dropout=0.1):
        super().__init__()
        self.num_mods = num_mods
        self.mod_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, f, 3, padding=1, bias=False),
                nn.InstanceNorm3d(f, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            ) for _ in range(num_mods)
        ])
        self.se = SEBlock3D(f)
        self.enc1 = ResConvBlock3D(f, f, dropout)
        self.enc2 = ResConvBlock3D(f, f*2, dropout)
        self.enc3 = ResConvBlock3D(f*2, f*4, dropout)
        self.enc4 = ResConvBlock3D(f*4, f*8, dropout)
        self.bottleneck = ResConvBlock3D(f*8, f*16, dropout)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x, mask):
        B = x.shape[0]
        feats = [self.mod_convs[i](x[:, i:i+1]) for i in range(self.num_mods)]
        stacked = torch.stack(feats, dim=1)
        m = mask.view(B, self.num_mods, 1, 1, 1, 1).float()
        fused = (stacked * m).sum(1) / m.sum(1).clamp(min=1)
        fused = self.se(fused)
        e1 = self.enc1(fused)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        return [e1, e2, e3, e4, bn]


class UNetDecoder(nn.Module):
    def __init__(self, f=32, dropout=0.1, seg_cls=3):
        super().__init__()
        self.up4 = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.dec4 = ResConvBlock3D(f*16, f*8, dropout)
        self.up3 = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.dec3 = ResConvBlock3D(f*8, f*4, dropout)
        self.up2 = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.dec2 = ResConvBlock3D(f*4, f*2, dropout)
        self.up1 = nn.ConvTranspose3d(f*2, f, 2, stride=2)
        self.dec1 = ResConvBlock3D(f*2, f, dropout)
        self.ds3 = nn.Conv3d(f*4, seg_cls, 1)
        self.ds2 = nn.Conv3d(f*2, seg_cls, 1)

    def forward(self, enc):
        e1, e2, e3, e4, bn = enc
        d4 = self.dec4(torch.cat([self.up4(bn), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1, self.ds3(d3), self.ds2(d2)


class MissingModalityNet(nn.Module):
    def __init__(self, num_mods=4, f=32, seg_cls=3, dropout=0.1):
        super().__init__()
        self.encoder = ModalityAwareEncoder(num_mods, f, dropout)
        self.decoder = UNetDecoder(f, dropout, seg_cls)
        self.recon_head = nn.Conv3d(f, num_mods, 1)
        self.seg_head = nn.Sequential(
            nn.Conv3d(f + num_mods, f, 3, padding=1, bias=False),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(f, seg_cls, 1),
        )

    def forward(self, x, mod_mask):
        enc = self.encoder(x, mod_mask)
        decoded, ds3, ds2 = self.decoder(enc)
        recon = self.recon_head(decoded)
        # Recon error only on PRESENT modalities
        m = mod_mask.view(-1, 4, 1, 1, 1).float()
        recon_err = (torch.abs(recon - x) * m).detach()
        seg = self.seg_head(torch.cat([decoded, recon_err], 1))
        return {'recon': recon, 'seg': seg, 'ds3': ds3, 'ds2': ds2}


# ============================================================================
#  LOSSES
# ============================================================================

class ReconLoss(nn.Module):
    def forward(self, pred, target, mod_mask):
        missing = (1.0 - mod_mask.float()).view(-1, 4, 1, 1, 1)
        if missing.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        brain = (target.abs() > 0).float()
        valid = missing * brain
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return (torch.abs(pred - target) * valid).sum() / valid.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(logits.shape[0], logits.shape[1], -1)
        t = targets.view(targets.shape[0], targets.shape[1], -1)
        inter = (p * t).sum(2)
        return 1.0 - ((2*inter + self.smooth) / (p.sum(2) + t.sum(2) + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.a, self.g = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (self.a * (1 - torch.exp(-bce)) ** self.g * bce).mean()


class TotalLoss(nn.Module):
    def __init__(self, rw=1.0, sw=1.0, dw=0.3):
        super().__init__()
        self.recon = ReconLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.rw, self.sw, self.dw = rw, sw, dw

    def forward(self, out, original, mod_mask, seg_target):
        L = {}
        L['recon'] = self.recon(out['recon'], original, mod_mask)
        L['dice'] = self.dice(out['seg'], seg_target)
        L['focal'] = self.focal(out['seg'], seg_target)
        t3 = F.interpolate(seg_target, size=out['ds3'].shape[2:], mode='nearest')
        t2 = F.interpolate(seg_target, size=out['ds2'].shape[2:], mode='nearest')
        L['ds'] = self.dice(out['ds3'], t3) + self.dice(out['ds2'], t2)
        L['total'] = self.rw * L['recon'] + self.sw * (L['dice'] + 0.5*L['focal']) + self.dw * L['ds']
        return L


# ============================================================================
#  METRICS
# ============================================================================

def dice_coeff(pred, target, smooth=1e-5):
    p, t = pred.float(), target.float()
    return ((2*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth)).item()


def compute_dice(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return {k: dice_coeff(pred[:, i], target[:, i]) for i, k in enumerate(['WT', 'TC', 'ET'])}


# ============================================================================
#  DATA LOADING — Split training data 80/20 (val set has no seg labels!)
# ============================================================================

def create_data_splits(data_dir, batch_size=2, num_workers=2, val_frac=0.15):
    """Split the TRAINING data into train/val since BraTS val has no seg labels."""
    full_dataset = BraTSDataset(data_dir, mode='train', missing_strategy='random',
                                min_present=1, augment=True)

    n_val = int(len(full_dataset) * val_frac)
    n_train = len(full_dataset) - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    # Override val subset to not augment and not drop modalities
    # We'll handle this in the validation loop by using original volumes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"\nData split (from {data_dir}):")
    print(f"  Train: {n_train} patients, {len(train_loader)} batches (bs={batch_size})")
    print(f"  Val:   {n_val} patients, {len(val_loader)} batches (bs=1)")
    return train_loader, val_loader


# ============================================================================
#  TRAINING
# ============================================================================

def train(model, train_loader, val_loader, num_epochs=200, lr=2e-4,
          device='cuda', save_dir='checkpoints', log_dir='runs/train',
          grad_accum=2, warmup_epochs=5):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    use_amp = device == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    criterion = TotalLoss()
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        sums = {'total': 0, 'recon': 0, 'dice': 0, 'focal': 0}
        d_sums = {'WT': 0, 'TC': 0, 'ET': 0}
        nb = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for bi, batch in enumerate(pbar):
            vol = batch['volume'].to(device)
            orig = batch['original'].to(device)
            mask = batch['mask'].to(device)
            seg = batch['seg'].to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(vol, mask)
                    L = criterion(out, orig, mask, seg)
                    loss = L['total'] / grad_accum
                scaler.scale(loss).backward()
                if (bi + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                out = model(vol, mask)
                L = criterion(out, orig, mask, seg)
                (L['total'] / grad_accum).backward()
                if (bi + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            for k in sums:
                if k in L:
                    sums[k] += L[k].item()
            nb += 1

            with torch.no_grad():
                for k, v in compute_dice(out['seg'], seg).items():
                    d_sums[k] += v

            pbar.set_postfix(
                loss=f"{L['total'].item():.3f}",
                dice=f"{L['dice'].item():.3f}",
                recon=f"{L['recon'].item():.3f}",
            )

        scheduler.step()
        n = max(nb, 1)

        # Log training metrics
        for k, v in sums.items():
            writer.add_scalar(f'train/{k}', v/n, epoch)
        for k, v in d_sums.items():
            writer.add_scalar(f'train/dice_{k}', v/n, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        # Print training dice
        train_dice_str = " ".join([f"{k}={d_sums[k]/n:.4f}" for k in ['WT','TC','ET']])
        avg_train_dice = np.mean([d_sums[k]/n for k in ['WT','TC','ET']])

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            vd, vl = validate(model, val_loader, criterion, device)
            md = np.mean(list(vd.values()))
            print(f"\n  Ep {epoch+1}: train_loss={sums['total']/n:.3f} train_dice={avg_train_dice:.4f} | "
                  f"Val WT={vd['WT']:.4f} TC={vd['TC']:.4f} ET={vd['ET']:.4f} Mean={md:.4f}")
            for k, v in vd.items():
                writer.add_scalar(f'val/dice_{k}', v, epoch)
            writer.add_scalar('val/loss', vl, epoch)

            if md > best_dice:
                best_dice = md
                torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                            'val_dice': vd, 'mean_dice': md},
                           os.path.join(save_dir, 'best_model.pth'))
                print(f"  -> Saved best (mean={md:.4f})")
        else:
            if (epoch + 1) % 1 == 0:
                print(f"  Ep {epoch+1}: loss={sums['total']/n:.3f} dice_loss={sums['dice']/n:.3f} "
                      f"recon={sums['recon']/n:.3f} | Train {train_dice_str}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ep{epoch+1}.pth'))

    writer.close()
    print(f"\nDone. Best val mean dice: {best_dice:.4f}")
    return model


def validate(model, loader, criterion, device):
    model.eval()
    ds = {'WT': 0, 'TC': 0, 'ET': 0}
    tl, c = 0, 0
    with torch.no_grad():
        for batch in loader:
            vol = batch['volume'].to(device)
            orig = batch['original'].to(device)
            mask = batch['mask'].to(device)
            seg = batch['seg'].to(device)

            # Check if this sample has any tumor labels
            if seg.sum() == 0:
                continue

            out = model(vol, mask)
            tl += criterion(out, orig, mask, seg)['total'].item()
            for k, v in compute_dice(out['seg'], seg).items():
                ds[k] += v
            c += 1

    n = max(c, 1)
    return {k: v/n for k, v in ds.items()}, tl/n


# ============================================================================
#  EVALUATION — All 15 missing combos
# ============================================================================

def evaluate_all(model, loader, device='cuda'):
    model.eval()
    combos = get_all_missing_combinations()
    results = []

    for combo in combos:
        present = combo['present']
        ds = {'WT': 0, 'TC': 0, 'ET': 0}
        c = 0
        
        # Safely get the description (or default to raw numbers)
        desc = combo.get('description', f"Mods: {present}")
        
        with torch.no_grad():
            # Add tqdm progress bar here
            for batch in tqdm(loader, desc=f"Evaluating {desc}", leave=False):
                seg = batch['seg']
                if seg.sum() == 0:
                    continue
                
                vol = batch['original'].to(device)
                seg = seg.to(device)
                
                mask = torch.zeros(1, 4, device=device)
                for i in present:
                    mask[0, i] = 1.0
                    
                v = vol.clone()
                for i in range(4):
                    if i not in present:
                        v[:, i] = 0.0
                        
                out = model(v, mask)
                for k, val in compute_dice(out['seg'], seg).items():
                    ds[k] += val
                c += 1
                
        n = max(c, 1)
        r = {k: ds[k]/n for k in ds}
        r['Mean'] = np.mean(list(r.values()))
        r['combo'] = desc
        r['n_present'] = len(present)
        results.append(r)
        
        # Print results as soon as the combo finishes
        print(f"  {desc:<45} WT={r['WT']:.4f} TC={r['TC']:.4f} ET={r['ET']:.4f} Mean={r['Mean']:.4f}")

    # Save to JSON
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\nBy # present modalities:")
    for np_ in [1, 2, 3, 4]:
        sub = [r for r in results if r['n_present'] == np_]
        if sub:
            print(f"  {np_} present: Mean Dice = {np.mean([r['Mean'] for r in sub]):.4f}")
            
    return results

# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['test', 'train', 'eval'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_features', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Device: {DEVICE}")

    model = MissingModalityNet(4, args.base_features, 3, args.dropout)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.stage == 'test':
        sz = 32 if DEVICE == 'cpu' else 64
        x = torch.randn(1, 4, sz, sz, sz)
        m = torch.tensor([[1, 0, 1, 1]], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            o = model(x, m)
        print(f"Input: {x.shape}, Recon: {o['recon'].shape}, Seg: {o['seg'].shape}")
        print("PASSED!")

    elif args.stage == 'train':
        # Use training data split into train/val (val set has no seg labels!)
        train_loader, val_loader = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=args.batch_size, num_workers=2
        )

        if args.checkpoint and os.path.exists(args.checkpoint):
            ck = torch.load(args.checkpoint, map_location=DEVICE)
            model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
            print(f"Resumed from {args.checkpoint}")

        train(model, train_loader, val_loader, args.epochs, args.lr, DEVICE,
              grad_accum=args.grad_accum)

    elif args.stage == 'eval':
        cp = args.checkpoint or 'checkpoints/best_model.pth'
        if not os.path.exists(cp):
            print(f"No checkpoint: {cp}"); sys.exit(1)
        ck = torch.load(cp, map_location=DEVICE)
        model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
        model = model.to(DEVICE)
        print(f"Loaded: {cp}")
        
        print("\nRecreating data splits to fetch the unseen validation set...")
        # Get the validation loader (ignoring the train_loader with '_')
        _, val_loader = create_data_splits(
            PROCESSED_TRAIN_DIR, batch_size=1, num_workers=2
        )
        
        print("\nEval all 15 combos on Validation Set:\n")
        # Pass the val_loader into evaluate_all
        evaluate_all(model, val_loader, DEVICE)

    print("\nDone!")
