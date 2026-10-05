#!/usr/bin/env python3 -u
"""
Retrain rsFC Transformer encoder on all 4 visits (Y0+Y2+Y4+Y6, N~33915).
Architecture identical to pretrain_rsfc_transformer.py (352 ROIs).
Output: checkpoint_RL/rsfc_encoder_allvisits.pt, checkpoint_RL/rsfc_allvisits.log
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

RSFC   = Path(os.environ.get('LUMEN_RSFC_DIR', 'data/rsfc'))
LABELS = Path(os.environ.get('LUMEN_LABELS', 'labels.csv'))
CKPT   = Path(os.environ.get('LUMEN_MODALITY_CHECKPOINT_DIR', 'modality_pretraining'))
SEED   = 42
N_ROIS = 352


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

# ── Data ─────────────────────────────────────────────────────────────────────

def load_all_visits():
    roi_pairs = np.load(RSFC / 'roi_pairs.npy')   # (61776, 2)
    all_mats, all_pids, all_visits = [], [], []

    for visit in ['y0', 'y2', 'y4', 'y6']:
        X_flat = np.load(RSFC / f'{visit}_X.npy')          # (N, 61776)
        pids   = np.load(RSFC / f'{visit}_pids.npy', allow_pickle=True)

        # Drop all-NaN subjects
        valid  = ~np.isnan(X_flat).all(axis=1)
        X_flat = X_flat[valid].copy(); pids = pids[valid]

        # Impute remaining NaN with column mean
        col_means = np.nanmean(X_flat, axis=0)
        nan_mask  = np.isnan(X_flat)
        X_flat[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Clip & Fisher-z
        X_flat = np.clip(X_flat, -0.9999, 0.9999)
        X_z    = np.arctanh(X_flat).astype(np.float32)

        # Flat → matrix
        N   = len(X_z)
        mat = np.zeros((N, N_ROIS, N_ROIS), dtype=np.float32)
        ii  = roi_pairs[:, 0]; jj = roi_pairs[:, 1]
        mat[:, ii, jj] = X_z; mat[:, jj, ii] = X_z

        all_mats.append(mat)
        all_pids.append(np.array(['NDAR_INV' + p.replace('sub-', '') for p in pids]))
        all_visits.append(np.full(N, visit.upper()))
        print(f'  {visit.upper()}: N={N} (after NaN filter)')

    X_all     = np.concatenate(all_mats, axis=0)
    pids_all  = np.concatenate(all_pids, axis=0)
    vis_all   = np.concatenate(all_visits, axis=0)

    # Per-subject normalization: each subject's FC matrix is z-scored independently.
    # Preserves within-subject connectivity structure for masked reconstruction.
    flat_all = X_all.reshape(len(X_all), -1)
    mu  = flat_all.mean(axis=1, keepdims=True)
    std = flat_all.std(axis=1, keepdims=True).clip(min=1e-6)
    X_all = ((flat_all - mu) / std).reshape(X_all.shape)

    print(f'\nTotal: N={len(X_all)}  shape={X_all.shape}')
    return X_all, pids_all, vis_all


# ── Model (same as pretrain_rsfc_transformer.py) ──────────────────────────────

class FCTransformerEncoder(nn.Module):
    def __init__(self, n_rois=N_ROIS, d_model=256, n_heads=8, n_layers=4,
                 emb_dim=256, dropout=0.1):
        super().__init__()
        self.n_rois     = n_rois
        self.d_model    = d_model
        self.input_proj = nn.Linear(n_rois, d_model)
        self.roi_embed  = nn.Embedding(n_rois, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, emb_dim))

    def forward(self, x_mat, mask_roi=None):
        B = x_mat.shape[0]
        tokens  = self.input_proj(x_mat)
        roi_idx = torch.arange(self.n_rois, device=x_mat.device)
        tokens  = tokens + self.roi_embed(roi_idx).unsqueeze(0)
        if mask_roi is not None:
            mask_exp = mask_roi.unsqueeze(-1).float()
            tokens   = tokens * (1 - mask_exp) + self.mask_token * mask_exp
        out = self.transformer(tokens)
        emb = self.head(out.mean(dim=1))
        return emb, out


class FCMAE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.d_model
        n = encoder.n_rois
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n))

    def forward(self, x, mask):
        emb, tokens = self.encoder(x, mask)
        recon       = self.decoder(tokens)
        mask_exp    = mask.unsqueeze(-1).float()
        loss        = ((recon - x) ** 2 * mask_exp).sum() / (mask_exp.sum() * x.shape[-1]).clamp(1)
        return loss, emb


def make_mask(B, n_rois, ratio, device):
    n_mask = max(1, int(ratio * n_rois))
    idx    = torch.topk(torch.rand(B, n_rois, device=device), n_mask, dim=1).indices
    mask   = torch.zeros(B, n_rois, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


# ── Probe ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(encoder, X_t, device, batch=64):
    encoder.eval()
    embs = []
    for i in range(0, len(X_t), batch):
        emb, _ = encoder(X_t[i:i+batch].to(device), mask_roi=None)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs)


def probe(encoder, X_t, pids_all, vis_all, labels_df, device):
    embs = encode_all(encoder, X_t, device)
    results = {}
    for task, visit, col in [('g_factor', 'Y0', 'g_factor'),
                               ('internalizing', 'Y0', 'internalizing')]:
        mask_v = vis_all == visit
        emb_v  = embs[mask_v]; pid_v = pids_all[mask_v]
        lbl    = labels_df[labels_df.visit == visit][['subid', col]].dropna()
        df_e   = pd.DataFrame(emb_v, columns=[f'e{i}' for i in range(emb_v.shape[1])])
        df_e['subid'] = pid_v
        mg = df_e.merge(lbl, on='subid', how='inner')
        if len(mg) < 50:
            results[task] = 0.0; continue
        feat = [f'e{i}' for i in range(emb_v.shape[1])]
        X = mg[feat].values; y = mg[col].values
        kf = KFold(5, shuffle=True, random_state=SEED)
        yp = np.zeros(len(y))
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            yp[te] = Ridge(alpha=1000).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        results[task] = round(pearsonr(y, yp)[0], 4)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import math, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--bs',         type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.30)
    parser.add_argument('--d_model',    type=int,   default=256)
    parser.add_argument('--n_layers',   type=int,   default=4)
    parser.add_argument('--patience',   type=int,   default=10,
                        help='stop after this many probe checks with no improvement in avg')
    args = parser.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    log_file = open(CKPT / 'rsfc_allvisits.log', 'a')
    sys.stdout = _Tee(sys.stdout, log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading all visits FC data...')
    X_all, pids_all, vis_all = load_all_visits()
    labels_df = pd.read_csv(LABELS)

    X_t    = torch.tensor(X_all, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t), batch_size=args.bs,
                        shuffle=True, drop_last=True, num_workers=0)

    encoder = FCTransformerEncoder(N_ROIS, args.d_model, 8, args.n_layers, 256).to(device)
    model   = FCMAE(encoder).to(device)
    n_p     = sum(p.numel() for p in model.parameters())
    print(f'\nFCMAE all-visits  N={len(X_all)}  n_rois={N_ROIS}  '
          f'd={args.d_model}  L={args.n_layers}  params={n_p:,}')
    print(f'mask={args.mask_ratio}  bs={args.bs}  epochs={args.epochs}\n')

    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    probe_every = max(1, args.epochs // 25)
    best_avg, best_ep, patience_cnt = -1, 0, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0; n = 0
        for (x,) in loader:
            x = x.to(device)
            mask = make_mask(len(x), N_ROIS, args.mask_ratio, device)
            loss, _ = model(x, mask)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
        sched.step()

        if ep % probe_every == 0 or ep == 1:
            res = probe(encoder, X_t, pids_all, vis_all, labels_df, device)
            avg = (res.get('g_factor', 0) + res.get('internalizing', 0)) / 2
            print(f'ep {ep:4d}  loss={total/n:.4f}  '
                  f'g_factor={res.get("g_factor",0):.4f}  '
                  f'internalizing={res.get("internalizing",0):.4f}  avg={avg:.4f}',
                  flush=True)
            if avg > best_avg:
                best_avg = avg; best_ep = ep; patience_cnt = 0
                torch.save(encoder.state_dict(), CKPT / 'rsfc_encoder_allvisits.pt')
                print(f'  ✓ Best saved (avg={best_avg:.4f})', flush=True)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  Early stop at ep {ep} (best ep {best_ep})', flush=True)
                    break
        elif ep % 10 == 0:
            print(f'ep {ep:4d}  loss={total/n:.4f}', flush=True)

    print(f'\nDone. Best ep {best_ep}  probe avg r={best_avg:.4f}')
    print(f'Checkpoint: {CKPT}/rsfc_encoder_allvisits.pt')


if __name__ == '__main__':
    main()
