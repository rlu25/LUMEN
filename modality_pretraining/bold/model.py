#!/usr/bin/env python3 -u
"""
BOLD temporal MAE + longitudinal contrastive pretraining (BrainLM-style).

Architecture  : ViT-MAE — each token = (1 ROI × PATCH_SIZE consecutive TRs).
                366 ROIs × 10 patches = 3660 tokens; 75% masked.
MAE loss      : MSE on masked patches only (per-ROI z-scored BOLD).
Contrastive   : NT-Xent, same subject different visit (or window) = positive pair.
Input         : $LUMEN_BOLD_DIR/{sub-XXX}_{visit}.npy  (366, T)
Output        : checkpoint_RL/bold_mae_encoder.pt, checkpoint_RL/bold_mae.log
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import random, time

# ── Paths ─────────────────────────────────────────────────────────────────────
TSERIES_DIR = Path(os.environ.get('LUMEN_BOLD_DIR', 'data/bold'))
LABELS      = Path(os.environ.get('LUMEN_LABELS', 'labels.csv'))
CKPT        = Path(os.environ.get('LUMEN_MODALITY_CHECKPOINT_DIR', 'modality_pretraining'))


class _Tee:
    """Duplicate writes to multiple streams (e.g. console + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_ROIS      = 366
T_WIN       = 200       # TRs per window (files with T<200 will be zero-padded)
PATCH_SIZE  = 20        # TRs per temporal patch
N_PATCHES   = T_WIN // PATCH_SIZE   # = 10
N_TOKENS    = N_ROIS * N_PATCHES    # = 3660
MASK_RATIO  = 0.75                  # fraction of tokens masked
D_MODEL     = 256
N_HEADS     = 8
N_ENC_LAYERS = 4
N_DEC_LAYERS = 2
D_DECODER   = 128
LAMBDA_CL   = 0.5   # weight of contrastive loss
TEMP_CL     = 0.1   # NT-Xent temperature
SEED        = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_window(fp, rng=None):
    """Load a random T_WIN-TR window from fp, apply per-ROI z-score."""
    ts = np.load(fp).astype(np.float32)   # (366, T)
    T  = ts.shape[1]

    # Per-ROI temporal z-score on the REAL frames only, before any padding
    # (padding first would dilute mean/std with fake zeros and turn the pad
    # region into a nonzero, data-dependent value instead of "no signal").
    mu  = ts.mean(axis=1, keepdims=True)
    std = ts.std(axis=1, keepdims=True) + 1e-6
    ts  = (ts - mu) / std
    np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if T >= T_WIN:
        # random.randint(a, b) is inclusive on both ends; max valid start = T - T_WIN
        max_start = T - T_WIN
        start = rng.randint(0, max_start) if (rng is not None and max_start > 0) else 0
        ts    = ts[:, start : start + T_WIN]
    else:
        # zero-pad on the right (0.0 = per-ROI mean in normalized space)
        pad       = np.zeros((N_ROIS, T_WIN), dtype=np.float32)
        pad[:, :T] = ts
        ts        = pad

    return ts   # (366, 200)


class BOLDPairDataset(Dataset):
    """
    Returns two windows per sample for joint MAE + contrastive learning.
      win1: from the primary scan (subid, fp1)
      win2: from another visit of the same subject (if available)
             or a different random window of the same scan (single-visit subjects)
    """
    def __init__(self, all_files, seed=SEED):
        self.rng = random.Random(seed)

        # Group by subject
        sub_files = {}
        for fp in all_files:
            stem = fp.stem                  # "sub-003RTV85_Y0"
            sid  = stem.split('_')[0]       # "sub-003RTV85"
            sub_files.setdefault(sid, []).append(fp)

        self.sub_files = sub_files
        self.index     = [(sid, fp) for sid, fps in sub_files.items() for fp in fps]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, fp1 = self.index[idx]

        # self.rng is reseeded per DataLoader worker per epoch (see
        # _worker_init_fn) and advances across calls, so crops and paired-visit
        # choice actually vary across epochs instead of being frozen per idx.
        win1 = load_window(fp1, self.rng)

        # Positive window: another visit or another time window
        others = [f for f in self.sub_files[sid] if f != fp1]
        if others:
            fp2  = self.rng.choice(others)
            win2 = load_window(fp2, self.rng)
        else:
            win2 = load_window(fp1, self.rng)   # different window, same scan

        return torch.tensor(win1), torch.tensor(win2)


def _worker_init_fn(worker_id):
    """Reseed each DataLoader worker's RNG so crop/pairing choices don't
    repeat the same sequence every epoch (workers are re-forked from the
    same parent Dataset state each epoch since persistent_workers=False)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(info.seed)


def split_subjects(all_files, seed=SEED, val_frac=0.1, test_frac=0.1):
    """Subject-level train/val/test split (no subject's scans cross splits).

    test: never touched during pretraining (not trained on, not probed on) —
          reserved so stage 2 can report a leakage-free number.
    val:  held out from training; used for the label-free validation loss.
    train: everything else, used for the actual MAE + contrastive training.
    """
    sub_files = {}
    for fp in all_files:
        sid = fp.stem.split('_')[0]
        sub_files.setdefault(sid, []).append(fp)

    sids = sorted(sub_files.keys())
    rng = random.Random(seed)
    rng.shuffle(sids)

    n = len(sids)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    splits_sids = {
        'test':  set(sids[:n_test]),
        'val':   set(sids[n_test:n_test + n_val]),
        'train': set(sids[n_test + n_val:]),
    }
    splits_files = {
        name: [fp for sid in sid_set for fp in sub_files[sid]]
        for name, sid_set in splits_sids.items()
    }
    return splits_files, splits_sids


# ── Model ─────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    (B, 366, 200) → (B, 3660, d_model) patch tokens with positional embeddings.
    Each token represents one ROI's temporal patch (20 consecutive TRs).
    """
    def __init__(self, n_rois=N_ROIS, n_patches=N_PATCHES,
                 patch_size=PATCH_SIZE, d_model=D_MODEL):
        super().__init__()
        self.n_rois    = n_rois
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.proj      = nn.Linear(patch_size, d_model)
        self.roi_emb   = nn.Embedding(n_rois, d_model)
        self.time_emb  = nn.Embedding(n_patches, d_model)

    def forward(self, x):
        B = x.shape[0]
        # Reshape into patches: (B, 366, 10, 20)
        x = x.view(B, self.n_rois, self.n_patches, self.patch_size)
        x = self.proj(x)                        # (B, 366, 10, d)

        roi_idx  = torch.arange(self.n_rois,    device=x.device)
        time_idx = torch.arange(self.n_patches, device=x.device)
        x = x + self.roi_emb(roi_idx).unsqueeze(1)    # broadcast over patches
        x = x + self.time_emb(time_idx).unsqueeze(0)  # broadcast over ROIs
        x = x.view(B, self.n_rois * self.n_patches, -1)  # (B, 3660, d)
        return x


class BOLDMAEEncoder(nn.Module):
    def __init__(self, n_rois=N_ROIS, n_patches=N_PATCHES, patch_size=PATCH_SIZE,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_ENC_LAYERS, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(n_rois, n_patches, patch_size, d_model)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.n_tokens    = n_rois * n_patches

    def forward(self, x, mask_ratio=MASK_RATIO):
        """
        Returns
        -------
        emb        : (B, d_model)  CLS embedding
        vis_tokens : (B, n_vis, d_model)  encoded visible tokens (no CLS)
        ids_keep   : (B, n_vis)  indices of visible tokens
        ids_restore: (B, N)  argsort(ids_shuffle) for unshuffling in decoder
        """
        B     = x.shape[0]
        tokens = self.patch_embed(x)    # (B, 3660, d)
        N      = tokens.shape[1]

        # Random masking
        n_vis   = int(N * (1 - mask_ratio))
        noise   = torch.rand(B, N, device=x.device)
        ids_shuffle  = torch.argsort(noise, dim=1)          # (B, N)
        ids_keep     = ids_shuffle[:, :n_vis]               # visible indices
        ids_restore  = torch.argsort(ids_shuffle, dim=1)    # unshuffle order

        # Gather visible tokens
        vis = torch.gather(tokens, 1,
                           ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        # Prepend CLS token
        cls  = self.cls_token.expand(B, -1, -1)
        vis  = torch.cat([cls, vis], dim=1)                 # (B, n_vis+1, d)

        # Encode
        out  = self.transformer(vis)
        out  = self.norm(out)
        emb        = out[:, 0]                              # CLS (B, d)
        vis_tokens = out[:, 1:]                             # (B, n_vis, d)

        return emb, vis_tokens, ids_keep, ids_restore

    @torch.no_grad()
    def encode(self, x):
        """Inference: encode WITHOUT masking, return CLS embedding."""
        B     = x.shape[0]
        tokens = self.patch_embed(x)
        N      = tokens.shape[1]
        ids_keep    = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        ids_restore = ids_keep.clone()
        cls  = self.cls_token.expand(B, -1, -1)
        inp  = torch.cat([cls, tokens], dim=1)
        out  = self.transformer(inp)
        out  = self.norm(out)
        return out[:, 0]   # (B, d)


class BOLDMAEDecoder(nn.Module):
    def __init__(self, n_tokens=N_TOKENS, d_encoder=D_MODEL,
                 d_decoder=D_DECODER, patch_size=PATCH_SIZE,
                 n_heads=4, n_layers=N_DEC_LAYERS, dropout=0.0):
        super().__init__()
        self.n_tokens   = n_tokens
        self.proj       = nn.Linear(d_encoder, d_decoder)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_decoder) * 0.02)
        self.pos_emb    = nn.Embedding(n_tokens, d_decoder)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder, nhead=n_heads, dim_feedforward=d_decoder * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.head        = nn.Linear(d_decoder, patch_size)

    def forward(self, vis_tokens, ids_keep, ids_restore):
        """
        vis_tokens : (B, n_vis, d_encoder)
        ids_keep   : (B, n_vis)
        ids_restore: (B, N)
        Returns    : (B, N, patch_size)  reconstruction for ALL patches
        """
        B, n_vis, _ = vis_tokens.shape
        N = ids_restore.shape[1]

        x = self.proj(vis_tokens)   # (B, n_vis, d_dec)

        # Append mask tokens for the missing positions
        n_mask     = N - n_vis
        mask_tok   = self.mask_token.expand(B, n_mask, -1)
        x_full     = torch.cat([x, mask_tok], dim=1)   # (B, N, d_dec)

        # Unshuffle to original spatial order
        x_full = torch.gather(x_full, 1,
                               ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))

        # Add positional embedding
        pos_idx = torch.arange(N, device=x.device)
        x_full  = x_full + self.pos_emb(pos_idx)

        x_full = self.transformer(x_full)
        recon  = self.head(x_full)    # (B, N, patch_size)
        return recon


# ── Losses ────────────────────────────────────────────────────────────────────

def mae_loss(recon, x_orig, ids_keep):
    """MSE on masked patches only (visible patches excluded from loss)."""
    B, N, P = recon.shape

    # Target: reshape original BOLD into patches
    target = x_orig.view(B, N_ROIS, N_PATCHES, PATCH_SIZE).view(B, N, P)

    # Binary mask: 1 = masked, 0 = visible
    mask = torch.ones(B, N, device=recon.device)
    mask.scatter_(1, ids_keep, 0.0)

    loss = ((recon - target) ** 2 * mask.unsqueeze(-1)).sum() \
           / (mask.sum() * P + 1e-8)
    return loss


def nt_xent_loss(z1, z2, temperature=TEMP_CL):
    """NT-Xent loss. (z1[i], z2[i]) are positive pairs."""
    B  = z1.shape[0]
    z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, d)
    sim = torch.mm(z, z.T) / temperature                   # (2B, 2B)
    # Mask self-similarity
    sim.fill_diagonal_(float('-inf'))
    # Positive pair indices: i → i+B, i+B → i
    labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(B,    device=z.device)])
    return F.cross_entropy(sim, labels)


@torch.no_grad()
def eval_val_loss(encoder, decoder, val_loader, device, mask_ratio):
    """Label-free generalization check: same MAE+contrastive loss as
    training, computed on held-out (never-trained-on) subjects."""
    encoder.eval(); decoder.eval()
    tot_mae = tot_cl = 0.0
    n_bat = 0
    for win1, win2 in val_loader:
        win1, win2 = win1.to(device), win2.to(device)
        emb1, vis1, ids_keep1, ids_restore1 = encoder(win1, mask_ratio)
        recon1 = decoder(vis1, ids_keep1, ids_restore1)
        loss_mae = mae_loss(recon1, win1, ids_keep1)
        emb2, _, _, _ = encoder(win2, mask_ratio)
        loss_cl = nt_xent_loss(emb1, emb2)
        tot_mae += loss_mae.item(); tot_cl += loss_cl.item(); n_bat += 1
    encoder.train(); decoder.train()
    return tot_mae / max(n_bat, 1), tot_cl / max(n_bat, 1)


# ── Probe (linear evaluation during training) ─────────────────────────────────

@torch.no_grad()
def encode_all(encoder, fps, device, batch_size=64):
    """Encode all BOLD files (Y0 only) without masking."""
    encoder.eval()
    embs, sids = [], []
    buf_x, buf_s = [], []

    def flush():
        if not buf_x:
            return
        x   = torch.stack(buf_x).to(device)
        emb = encoder.encode(x)
        embs.append(emb.cpu().numpy())
        sids.extend(buf_s)
        buf_x.clear(); buf_s.clear()

    n_skip = 0
    for fp in fps:
        ts  = load_window(fp, rng=None)            # first T_WIN TRs, no randomness
        if not np.isfinite(ts).all():
            n_skip += 1
            continue
        buf_x.append(torch.from_numpy(ts))
        stem = fp.stem
        sid  = 'NDAR_INV' + stem.split('_')[0].replace('sub-', '')
        buf_s.append(sid)
        if len(buf_x) >= batch_size:
            flush()
    flush()
    if n_skip:
        print(f'  [encode_all] Skipped {n_skip} NaN files')

    return np.concatenate(embs), np.array(sids)


def probe(encoder, y0_files, labels_df, device):
    embs, sids = encode_all(encoder, y0_files, device)
    # Drop NaN embeddings
    valid  = np.isfinite(embs).all(axis=1)
    embs, sids = embs[valid], sids[valid]
    d      = embs.shape[1]
    feat   = [f'e{i}' for i in range(d)]
    df_e   = pd.DataFrame(embs, columns=feat)
    df_e['subid'] = sids

    results = {}
    for task, col in [('g_factor', 'g_factor'), ('internalizing', 'internalizing')]:
        lbl = (labels_df[labels_df.visit == 'Y0'][['subid', col]]
               .dropna().drop_duplicates('subid'))
        mg  = df_e.merge(lbl, on='subid', how='inner')
        if len(mg) < 50:
            results[task] = 0.0
            continue
        X = mg[feat].values; y = mg[col].values
        kf = KFold(5, shuffle=True, random_state=SEED)
        yp = np.zeros(len(y))
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            yp[te] = (Ridge(alpha=1000)
                      .fit(sc.transform(X[tr]), y[tr])
                      .predict(sc.transform(X[te])))
        results[task] = round(pearsonr(y, yp)[0], 4)
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs',      type=int,   default=1000)
    ap.add_argument('--bs',          type=int,   default=32)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--mask_ratio',  type=float, default=MASK_RATIO)
    ap.add_argument('--lambda_cl',   type=float, default=LAMBDA_CL)
    ap.add_argument('--d_model',     type=int,   default=D_MODEL)
    ap.add_argument('--n_enc',       type=int,   default=N_ENC_LAYERS)
    ap.add_argument('--n_dec',       type=int,   default=N_DEC_LAYERS)
    ap.add_argument('--workers',     type=int,   default=4)
    ap.add_argument('--patience',    type=int,   default=10,
                    help='stop after this many probe checks with no improvement in the g_factor probe score')
    args = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    log_file = open(CKPT / 'bold_mae.log', 'a')
    sys.stdout = _Tee(sys.stdout, log_file)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── Data ──────────────────────────────────────────────────────────────────
    all_files = sorted(TSERIES_DIR.glob('*.npy'))

    # Subject-level train/val/test split. `test` is never touched here (not
    # trained on, not probed on) so stage 2 can report a leakage-free number;
    # `val` is held out from training and used for the label-free val loss.
    splits_files, splits_sids = split_subjects(all_files, seed=SEED)
    train_files, val_files, test_files = splits_files['train'], splits_files['val'], splits_files['test']
    print(f'Total files: {len(all_files)}  |  subjects: train={len(splits_sids["train"])}'
          f'  val={len(splits_sids["val"])}  test={len(splits_sids["test"])}')

    pd.Series(sorted(splits_sids['test']), name='subid').to_csv(
        CKPT / 'held_out_test_subjects.csv', index=False)
    print(f'Held-out test subjects (never used in pretraining) saved to '
          f'{CKPT / "held_out_test_subjects.csv"} — exclude these from stage 2 eval leakage checks.')

    y0_files = [f for f in train_files + val_files if '_Y0' in f.stem]
    print(f'Y0 files available for the label probe (train+val only): {len(y0_files)}')

    dataset = BOLDPairDataset(train_files, seed=SEED)
    loader  = DataLoader(dataset, batch_size=args.bs, shuffle=True,
                         num_workers=args.workers, pin_memory=True,
                         drop_last=True, persistent_workers=False,
                         worker_init_fn=_worker_init_fn)
    print(f'Train dataset: {len(dataset)} samples  |  {len(loader)} batches/epoch')

    val_dataset = BOLDPairDataset(val_files, seed=SEED + 1)
    val_loader  = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                             num_workers=0, drop_last=False)
    print(f'Val dataset: {len(val_dataset)} samples  |  {len(val_loader)} batches/epoch\n')

    labels_df = pd.read_csv(LABELS)

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = BOLDMAEEncoder(
        d_model=args.d_model, n_layers=args.n_enc).to(device)
    decoder = BOLDMAEDecoder(
        d_encoder=args.d_model, n_layers=args.n_dec).to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f'BOLDMAEEncoder  params={n_enc:,}  d={args.d_model}  L_enc={args.n_enc}')
    print(f'BOLDMAEDecoder  params={n_dec:,}  d_dec={D_DECODER}  L_dec={args.n_dec}')
    print(f'N_ROIS={N_ROIS}  T_WIN={T_WIN}  PATCH_SIZE={PATCH_SIZE}'
          f'  N_TOKENS={N_TOKENS}  mask={args.mask_ratio}')
    print(f'bs={args.bs}  lr={args.lr}  lambda_cl={args.lambda_cl}'
          f'  epochs={args.epochs}\n')

    opt   = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score, best_ep, patience_cnt = -1.0, 0, 0
    # Capped at 20 epochs: epochs//20 alone means larger --epochs runs check
    # checkpoints less and less often (e.g. every 100 epochs at epochs=2000),
    # leaving a long blind spot early in training when checkpoint selection
    # matters most. Cap keeps the max gap fixed regardless of total epochs.
    probe_every = min(max(1, args.epochs // 20), 20)

    for ep in range(1, args.epochs + 1):
        encoder.train(); decoder.train()
        tot_mae = tot_cl = 0.0
        n_bat = 0
        t0 = time.time()

        for win1, win2 in loader:
            win1 = win1.to(device)
            win2 = win2.to(device)

            # ── Encode win1 with masking ──────────────────────────────────────
            emb1, vis1, ids_keep1, ids_restore1 = encoder(win1, args.mask_ratio)
            recon1 = decoder(vis1, ids_keep1, ids_restore1)
            loss_mae = mae_loss(recon1, win1, ids_keep1)

            # ── Encode win2 with masking ──────────────────────────────────────
            emb2, _, _, _ = encoder(win2, args.mask_ratio)

            # ── NT-Xent contrastive loss ──────────────────────────────────────
            loss_cl = nt_xent_loss(emb1, emb2)

            loss = loss_mae + args.lambda_cl * loss_cl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            tot_mae += loss_mae.item()
            tot_cl  += loss_cl.item()
            n_bat   += 1

        sched.step()
        elapsed = time.time() - t0

        if ep % probe_every == 0 or ep == 1:
            val_mae, val_cl = eval_val_loss(encoder, decoder, val_loader, device, args.mask_ratio)
            res = probe(encoder, y0_files, labels_df, device)
            # Checkpoint selection uses g_factor only -- internalizing is still
            # computed/printed for visibility, but its probe correlation has
            # been sitting near zero (noise, not signal), so averaging it in
            # was diluting/adding noise to the save criterion.
            score = res.get('g_factor', 0)
            print(f'ep {ep:4d}  mae={tot_mae/n_bat:.4f}  cl={tot_cl/n_bat:.4f}'
                  f'  val_mae={val_mae:.4f}  val_cl={val_cl:.4f}'
                  f'  g={res.get("g_factor",0):.4f}  int={res.get("internalizing",0):.4f}'
                  f'  score={score:.4f}  ({elapsed:.0f}s)', flush=True)
            if score > best_score:
                best_score = score; best_ep = ep; patience_cnt = 0
                torch.save(encoder.state_dict(), CKPT / 'bold_mae_encoder.pt')
                print(f'  ✓ Best saved (score={best_score:.4f}, val_mae={val_mae:.4f})', flush=True)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  Early stop at ep {ep} (best ep {best_ep})', flush=True)
                    break
        elif ep % 10 == 0:
            print(f'ep {ep:4d}  mae={tot_mae/n_bat:.4f}  cl={tot_cl/n_bat:.4f}'
                  f'  ({elapsed:.0f}s)', flush=True)

    print(f'\nDone. Best ep {best_ep}  probe g_factor r={best_score:.4f}')
    print(f'Checkpoint: {CKPT}/bold_mae_encoder.pt')


if __name__ == '__main__':
    main()
