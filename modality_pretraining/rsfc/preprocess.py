#!/usr/bin/env python3 -u
"""
Extract Y0-Y6 rsFC data from corr.mat, matching the 352-ROI set used in Y0 npy.

Strategy:
  1. Read vol_info.mat → map column index → (subject_id, session)
  2. Match Y0 roi_names (352 ROIs) to corr.mat roinames (582 ROIs) → ROI index map
  3. Find which of the 169,653 pairs in corr.mat correspond to the 61,776 pairs in Y0 npy
  4. For each session, extract those rows and save as npy

Sessions in corr.mat: baseline (Y0), 2year (Y2), 4year (Y4), 6year (Y6)

Output:
  rsFC/y0_X.npy   : (N_y0, 61776) float32  — re-extracted for consistency check
  rsFC/y2_X.npy   : (N_y2, 61776) float32
  rsFC/y4_X.npy   : (N_y4, 61776) float32
  rsFC/y6_X.npy   : (N_y6, 61776) float32
  (+ corresponding _pids.npy for each)
"""

import h5py
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

CORR = Path(os.environ.get('LUMEN_RSFC_CORR_MAT', 'data/rsfc/corr.mat'))
VOLI = Path(os.environ.get('LUMEN_RSFC_VOLUME_INFO', 'data/rsfc/vol_info.mat'))
RSFC = Path(os.environ.get('LUMEN_RSFC_DIR', 'data/rsfc'))


def read_str_vec(f, key):
    """Read a (1, N) object array of h5py references → list of strings."""
    arr = f[key]
    out = []
    for i in range(arr.shape[1]):
        ref = arr[0, i]
        s   = ''.join(chr(c) for c in f[ref][:].flatten())
        out.append(s)
    return out


def read_roi_names(f):
    arr = f['roinames']
    out = []
    for i in range(arr.shape[0]):
        ref = arr[i, 0]
        s   = ''.join(chr(c) for c in f[ref][:].flatten())
        out.append(s)
    return out


def main():
    # ── 1. Load vol_info: column index → (pid, session) ───────────────────────
    print('Reading vol_info.mat...')
    with h5py.File(VOLI, 'r') as f:
        pids_raw = read_str_vec(f, 'participant_id')   # 'sub-XXXXXXXX'
        visit_ids = read_str_vec(f, 'visitidvec')      # 'SITE_INVXXXXXX_baseline/2year/4year'

    pids_raw  = np.array(pids_raw)
    visit_ids = np.array(visit_ids)

    # Parse session from visitid (last token after _)
    sessions = np.array([v.split('_')[-1] for v in visit_ids])

    # Unique sessions
    from collections import Counter
    print('Session counts:', Counter(sessions))

    # ── 2. Load corr.mat ROI names and pair indices ─────────────────────────
    print('\nReading corr.mat metadata...')
    with h5py.File(CORR, 'r') as f:
        corr_roi_names = read_roi_names(f)                    # list of 582 strings
        roi1 = f['roi1vec'][0, :].astype(int) - 1            # 0-based ROI indices
        roi2 = f['roi2vec'][0, :].astype(int) - 1
    corr_roi_names = np.array(corr_roi_names)

    print(f'corr.mat: {len(corr_roi_names)} ROIs, {len(roi1)} pairs')

    # ── 3. Map Y0 ROI names → corr.mat ROI indices ──────────────────────────
    y0_roi_names = np.load(RSFC / 'roi_names.npy', allow_pickle=True)   # 352 names
    y0_roi_pairs = np.load(RSFC / 'roi_pairs.npy')                       # (61776, 2)

    # Map: y0 roi name → index in corr.mat
    corr_name_to_idx = {n: i for i, n in enumerate(corr_roi_names)}
    y0_to_corr = np.array([corr_name_to_idx[n] for n in y0_roi_names])   # (352,)
    print(f'All Y0 ROI names found in corr.mat: {len(y0_to_corr) == len(y0_roi_names)}')

    # ── 4. Find which corr.mat pairs match Y0 pairs ──────────────────────────
    # Y0 pairs are in y0 roi-space (0..351); translate to corr.mat roi-space
    y0_r1_corr = y0_to_corr[y0_roi_pairs[:, 0]]   # (61776,) corr.mat roi indices
    y0_r2_corr = y0_to_corr[y0_roi_pairs[:, 1]]

    # Build lookup: frozenset(i,j) → pair index in corr.mat
    print('Building corr.mat pair lookup...')
    corr_pair_to_idx = {}
    for k in range(len(roi1)):
        key = (min(roi1[k], roi2[k]), max(roi1[k], roi2[k]))
        corr_pair_to_idx[key] = k

    # Find the corr.mat row index for each Y0 pair
    print('Mapping Y0 pairs to corr.mat rows...')
    y0_pair_rows = np.array([
        corr_pair_to_idx[(min(y0_r1_corr[k], y0_r2_corr[k]),
                           max(y0_r1_corr[k], y0_r2_corr[k]))]
        for k in range(len(y0_roi_pairs))
    ])
    print(f'Mapped {len(y0_pair_rows)} pairs. Sample: {y0_pair_rows[:5]}')

    # ── 5. Extract Y2 / Y4 columns from corrmat ──────────────────────────────
    for sess_name, sess_label in [('y0', 'baseline'), ('y2', '2year'),
                                   ('y4', '4year'), ('y6', '6year')]:
        col_mask = sessions == sess_label
        col_idx  = np.where(col_mask)[0]
        sess_pids = pids_raw[col_idx]

        print(f'\n=== {sess_name.upper()} ({sess_label}): {len(col_idx)} sessions ===')

        # Read corr.mat in chunks (it's large); extract y0_pair_rows rows & col_idx cols
        n_pairs = len(y0_pair_rows)
        n_sess  = len(col_idx)
        X_out   = np.full((n_sess, n_pairs), np.nan, dtype=np.float32)

        print(f'Extracting {n_pairs} pairs × {n_sess} sessions from corrmat...')
        with h5py.File(CORR, 'r') as f:
            corrmat = f['corrmat']   # (169653, 33915), rows=pairs, cols=sessions
            # Read in chunks of 5000 pairs to manage memory
            chunk = 5000
            for start in range(0, n_pairs, chunk):
                end     = min(start + chunk, n_pairs)
                pair_chunk = y0_pair_rows[start:end]
                # h5py requires sorted indices for fancy indexing
                sort_order = np.argsort(pair_chunk)
                sorted_pairs = pair_chunk[sort_order]
                data = corrmat[sorted_pairs][:, col_idx]   # (chunk, n_sess)
                # Restore original order
                inv_order = np.argsort(sort_order)
                X_out[:, start:end] = data[inv_order].T
                if start % 20000 == 0:
                    print(f'  ... pairs {start}/{n_pairs}', flush=True)

        # Validate: check overlap with Y0 for subjects that appear in both
        print(f'NaN rate: {np.isnan(X_out).mean():.4f}')
        print(f'Non-NaN subjects: {(~np.isnan(X_out).all(axis=1)).sum()}')

        # Save
        out_X   = RSFC / f'{sess_name}_X.npy'
        out_pid = RSFC / f'{sess_name}_pids.npy'
        np.save(out_X,   X_out)
        np.save(out_pid, sess_pids)
        print(f'Saved → {out_X}  (shape {X_out.shape})')
        print(f'Saved → {out_pid}')

    # ── Sanity check: verify Y0 extraction matches existing npy ──────────────
    print('\n=== Sanity check Y0 ===')
    col_mask = sessions == 'baseline'
    col_idx  = np.where(col_mask)[0]
    sess_pids = pids_raw[col_idx]
    y0_existing = np.load(RSFC / 'y0_X.npy')
    y0_pids_ex  = np.load(RSFC / 'y0_pids.npy', allow_pickle=True)
    n_check = 5
    print(f'Checking {n_check} subjects...')
    with h5py.File(CORR, 'r') as f:
        corrmat = f['corrmat']
        for check_pid in y0_pids_ex[:n_check]:
            # find in corr.mat columns
            col_pos = np.where(sess_pids == check_pid)[0]
            if len(col_pos) == 0:
                print(f'  {check_pid}: not found in corr.mat baseline cols')
                continue
            c = col_pos[0]
            row_vals = corrmat[y0_pair_rows[:10], col_idx[c]]
            # find in Y0 existing npy
            ex_pos = np.where(y0_pids_ex == check_pid)[0]
            if len(ex_pos) == 0: continue
            ex_vals = y0_existing[ex_pos[0], :10]
            match = np.allclose(row_vals, ex_vals, atol=1e-4, equal_nan=True)
            print(f'  {check_pid}: values match = {match}  '
                  f'corr={row_vals[:3].round(4)}  npy={ex_vals[:3].round(4)}')


if __name__ == '__main__':
    main()
