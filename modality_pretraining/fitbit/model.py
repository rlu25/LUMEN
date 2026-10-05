#!/usr/bin/env python3 -u
"""
WBM-style Fitbit encoder pretraining on hourly data.

Design (following WBM, Apple ICML 2025):
  - Input per hour: 9 features (steps, calories, intensity, met, hr,
                  wake/light/deep/rem_min) + 9 missingness indicators → 18-dim token
  - Tokenization: Linear(18, d_model) + sinusoidal positional embedding
  - Encoder: Transformer (4 layers, 4 heads, d_model=128) + CLS token
  - Pretraining: subject-level InfoNCE contrastive
  - Augmentation: random token dropping (two independent views per subject)
  - Window: 168 hours (1 week), randomly cropped from full sequence

Checkpoint + log: checkpoint_RL/fitbit_encoder_wbm.pt, checkpoint_RL/fitbit_wbm.log
"""

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

CACHE = Path(os.environ.get('LUMEN_FITBIT_CACHE', 'data/fitbit_hourly_cache'))
DEMO  = Path(os.environ.get('LUMEN_DEMOGRAPHICS_CSV', 'data/abcd_y_lt.csv'))
COGN  = Path(os.environ.get('LUMEN_COGNITION_CSV', 'data/nc_y_nihtb.csv'))
CKPT  = Path(os.environ.get('LUMEN_MODALITY_CHECKPOINT_DIR', 'modality_pretraining'))
SEED  = 42


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

N_FEAT   = 9     # steps, calories, intensity, met, hr, wake/light/deep/rem_min
D_IN     = 18    # 9 features + 9 missingness indicators
WIN_HOURS = 168  # 1-week window


# ── Dataset ────────────────────────────────────────────────────────────────────

class FitbitHourlyDataset(Dataset):
    """
    Returns two augmented views (independent random crop + token drop) per subject.
    Each view: (WIN_HOURS, D_IN) float tensor, NaN-free.
    """
    def __init__(self, pids, drop_rate=0.15, seed=SEED):
        self.pids      = pids
        self.drop_rate = drop_rate
        self.rng       = np.random.default_rng(seed)

        # Load all arrays into memory (each ~40KB, 7K subjects = ~280MB)
        self.data = {}
        for pid in pids:
            p = CACHE / f'{pid}.npy'
            if p.exists():
                arr = np.load(p)   # (N_hours, 9), NaN for missing values
                if len(arr) >= WIN_HOURS:
                    self.data[pid] = arr

        self.valid_pids = list(self.data.keys())
        print(f'Loaded {len(self.valid_pids)}/{len(pids)} subjects '
              f'(≥{WIN_HOURS}h of data)')

    def _make_token(self, arr: np.ndarray) -> np.ndarray:
        """(WIN_HOURS, 9) → (WIN_HOURS, 18): impute NaN, append mask."""
        mask = np.isnan(arr).astype(np.float32)          # 1 = was NaN
        arr  = np.where(np.isnan(arr), 0.0, arr)
        return np.concatenate([arr, mask], axis=-1)       # (WIN_HOURS, 18)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Random token dropping: zero out D_IN features + set mask bits to 1."""
        x = x.copy()
        drop = self.rng.random(len(x)) < self.drop_rate
        x[drop, :N_FEAT] = 0.0
        x[drop, N_FEAT:] = 1.0    # mark dropped as missing
        return x

    def __len__(self):
        return len(self.valid_pids)

    def __getitem__(self, idx):
        pid = self.valid_pids[idx]
        arr = self.data[pid]       # (N_hours, 9)

        # Two independent 168-hour crops (different weeks) so the positive
        # pair must share subject-level signal, not one week's wear/missingness
        # fingerprint — further perturbed by independent token dropping.
        max_start = len(arr) - WIN_HOURS
        start1 = self.rng.integers(0, max_start + 1)
        start2 = self.rng.integers(0, max_start + 1)
        base1  = self._make_token(arr[start1: start1 + WIN_HOURS])   # (168, 18)
        base2  = self._make_token(arr[start2: start2 + WIN_HOURS])   # (168, 18)

        view1 = torch.tensor(self._augment(base1), dtype=torch.float32)
        view2 = torch.tensor(self._augment(base2), dtype=torch.float32)
        return view1, view2


def _worker_init_fn(worker_id):
    """Reseed each DataLoader worker's RNG so augmentation doesn't repeat
    the same sequence every epoch (workers are re-forked from the same
    parent Dataset state each epoch when persistent_workers=False)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


# ── Model ──────────────────────────────────────────────────────────────────────

class FitbitEncoder(nn.Module):
    """
    Hourly Fitbit sequence → fixed-dim embedding via Transformer + CLS token.
    """
    def __init__(self, d_in=D_IN, d_model=128, n_heads=4, n_layers=4,
                 emb_dim=128, dropout=0.1, max_len=WIN_HOURS):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)

        # Sinusoidal positional embedding (fixed, not learned — hour position)
        pe = self._make_sinusoidal(max_len, d_model)
        self.register_buffer('pe', pe)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, emb_dim))

    @staticmethod
    def _make_sinusoidal(max_len, d_model):
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe  = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, D_in) → (B, emb_dim)"""
        B  = x.shape[0]
        t  = self.input_proj(x) + self.pe[:, :x.shape[1]]         # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, d_model)
        t   = torch.cat([cls, t], dim=1)                           # (B, T+1, d_model)
        out = self.transformer(t)                                   # (B, T+1, d_model)
        emb = self.head(out[:, 0])                                  # CLS → (B, emb_dim)
        return emb


class FitbitWBM(nn.Module):
    def __init__(self, d_in=D_IN, d_model=128, n_heads=4, n_layers=4,
                 emb_dim=128, proj_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = FitbitEncoder(d_in, d_model, n_heads, n_layers, emb_dim, dropout)
        # Projection head for contrastive (discarded after pretraining)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, proj_dim))

    def forward(self, x):
        return self.proj(self.encoder(x))


# ── Contrastive loss (InfoNCE / NT-Xent) ──────────────────────────────────────

def infonce_loss(z1, z2, temperature=0.1):
    """NT-Xent loss over a batch of positive pairs (z1[i], z2[i])."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B  = z1.shape[0]

    # (2B, 2B) similarity matrix
    z   = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature

    # Mask self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)
    loss   = F.cross_entropy(sim, labels)
    return loss


# ── Normalisation (fit on training set, per feature) ──────────────────────────

def fit_normalizer(valid_pids, sample_n=500):
    """Compute per-feature mean/std on non-missing values from a sample."""
    rng   = np.random.default_rng(SEED)
    samp  = rng.choice(valid_pids, min(sample_n, len(valid_pids)), replace=False)
    feats = []
    for pid in samp:
        arr = np.load(CACHE / f'{pid}.npy')
        feats.append(arr)
    feats  = np.concatenate(feats, axis=0)   # (N_total_hours, 5)
    means  = np.nanmean(feats, axis=0)
    stds   = np.nanstd(feats, axis=0).clip(min=1e-6)
    return means, stds


# ── Probe ──────────────────────────────────────────────────────────────────────

def load_probe_targets():
    lt  = pd.read_csv(DEMO, low_memory=False)
    age = (lt[lt['eventname'] == '2_year_follow_up_y_arm_1']
             [['src_subject_id', 'interview_age']]
           .assign(interview_age=lambda d: pd.to_numeric(d['interview_age'], errors='coerce'))
           .dropna())
    nb  = pd.read_csv(COGN, low_memory=False)
    nb2 = nb[nb['eventname'] == '2_year_follow_up_y_arm_1'][
              ['src_subject_id', 'nihtbx_cryst_agecorrected']].copy()
    nb2['cryst'] = pd.to_numeric(nb2['nihtbx_cryst_agecorrected'], errors='coerce')
    return age, nb2[['src_subject_id', 'cryst']]


@torch.no_grad()
def encode_dataset(encoder, valid_pids, means, stds, device, win=WIN_HOURS):
    """Encode all subjects using a center crop → (N, emb_dim)."""
    encoder.eval()
    embs, out_pids = [], []
    for pid in valid_pids:
        arr = np.load(CACHE / f'{pid}.npy')
        if len(arr) < win:
            continue
        # Center crop
        start = (len(arr) - win) // 2
        crop  = arr[start: start + win].astype(np.float32)
        mask  = np.isnan(crop).astype(np.float32)
        crop  = np.where(np.isnan(crop), 0.0, (crop - means) / stds)
        x     = torch.tensor(np.concatenate([crop, mask], axis=-1)).unsqueeze(0).to(device)
        emb   = encoder(x).cpu().numpy()
        embs.append(emb[0])
        out_pids.append(pid)
    return np.array(embs), out_pids


def linear_probe(encoder, valid_pids, means, stds, age_df, cog_df, device):
    embs, epids = encode_dataset(encoder, valid_pids, means, stds, device)
    sid_df = pd.DataFrame({'src_subject_id': epids, '_idx': np.arange(len(epids))})
    results = {}
    for name, tdf, tcol in [('brain_age', age_df, 'interview_age'),
                              ('cryst',    cog_df, 'cryst')]:
        m = sid_df.merge(tdf, on='src_subject_id').dropna(subset=[tcol])
        X, y = embs[m['_idx'].values], m[tcol].values.astype(np.float32)
        kf, yp = KFold(5, shuffle=True, random_state=SEED), np.zeros(len(y))
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            yp[te] = Ridge(alpha=100).fit(sc.transform(X[tr]), y[tr]).predict(
                         sc.transform(X[te]))
        results[name] = round(pearsonr(y, yp)[0], 4)
    return results


# ── Pre-training ───────────────────────────────────────────────────────────────

def pretrain(args):
    CKPT.mkdir(parents=True, exist_ok=True)
    log_file = open(CKPT / 'fitbit_wbm.log', 'a')
    sys.stdout = _Tee(sys.stdout, log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # All available subjects
    all_pids = sorted(p.stem for p in CACHE.glob('*.npy')
                      if not p.stem.endswith('_ts'))

    age_df, cog_df = load_probe_targets()

    dataset = FitbitHourlyDataset(all_pids, drop_rate=args.drop_rate)
    valid_pids = dataset.valid_pids

    # Normalisation stats
    means, stds = fit_normalizer(valid_pids)

    # Pre-normalise cached arrays (update dataset to use normalised values)
    print('Normalising cached arrays...')
    for pid in valid_pids:
        arr = dataset.data[pid]
        dataset.data[pid] = (arr - means) / stds   # NaN stays NaN

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=4,
                        worker_init_fn=_worker_init_fn)

    model = FitbitWBM(D_IN, args.d_model, args.n_heads, args.n_layers,
                      args.emb_dim, args.proj_dim, args.dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nFitbit WBM encoder  N={len(valid_pids)}  win={WIN_HOURS}h')
    print(f'd_model={args.d_model}  n_layers={args.n_layers}  params={n_params:,}')
    print(f'drop_rate={args.drop_rate}  temp={args.temperature}  bs={args.batch_size}\n')

    best_r, best_ep, patience_cnt = -1, 0, 0
    probe_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss   = infonce_loss(z1, z2, args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1
        sched.step()

        if epoch % probe_every == 0 or epoch == args.epochs:
            probe = linear_probe(model.encoder, valid_pids, means, stds,
                                 age_df, cog_df, device)
            avg_r = (probe['brain_age'] + probe['cryst']) / 2
            print(f'ep {epoch:4d}  loss={total_loss/n:.4f}  '
                  f'brain_age={probe["brain_age"]:.4f}  cryst={probe["cryst"]:.4f}')
            if avg_r > best_r + 1e-4:
                best_r = avg_r; best_ep = epoch; patience_cnt = 0
                torch.save(model.encoder.state_dict(), CKPT / 'fitbit_encoder_wbm.pt')
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  Early stop at ep {epoch} (best ep {best_ep})')
                    break

    print(f'\nBest ep {best_ep}  avg probe r={best_r:.4f}')
    print(f'Encoder saved → {CKPT / "fitbit_encoder_wbm.pt"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--d_model',     type=int,   default=128)
    p.add_argument('--emb_dim',     type=int,   default=128)
    p.add_argument('--proj_dim',    type=int,   default=64)
    p.add_argument('--n_heads',     type=int,   default=4)
    p.add_argument('--n_layers',    type=int,   default=4)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--drop_rate',   type=float, default=0.15,
                   help='fraction of hourly tokens randomly dropped per view')
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--batch_size',  type=int,   default=256)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--epochs',      type=int,   default=2000)
    p.add_argument('--patience',    type=int,   default=50)
    args = p.parse_args()
    pretrain(args)
