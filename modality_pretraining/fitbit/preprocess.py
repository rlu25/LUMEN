#!/usr/bin/env python3 -u
"""
Aggregate per-subject minute-level Fitbit data to hourly features.

Input:  $LUMEN_FITBIT_RAW_DIR/
        NDAR_INV{pid}_{heartrate_1min,minuteSteps,...}.csv

Output: fitbit_hourly_cache/
        {pid}.npy       (N_hours, 5) float32  — [steps, calories, intensity, met, hr]
                        NaN where HR not available; steps/cal/intensity/met always present
        manifest.csv    subject_id, n_hours, start_hour, end_hour
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

RAW   = Path(os.environ.get('LUMEN_FITBIT_RAW_DIR', 'data/fitbit_raw'))
CACHE = Path(os.environ.get('LUMEN_FITBIT_CACHE', 'data/fitbit_hourly_cache'))
N_WORKERS = 32

FEAT_COLS = ['steps', 'calories', 'intensity', 'met', 'hr',
             'wake_min', 'light_min', 'deep_min', 'rem_min']   # 9 features


def process_subject(pid: str):
    out_path = CACHE / f'{pid}.npy'
    ts_path  = CACHE / f'{pid}_ts.npy'
    if out_path.exists():
        arr = np.load(out_path)
        if arr.shape[1] == len(FEAT_COLS):   # re-process if old 5-feature cache
            return (pid, len(arr), None, None)

    try:
        step = pd.read_csv(RAW / f'{pid}_minuteStepsNarrow.csv',
                           parse_dates=['Wear_Time']).set_index('Wear_Time').rename(columns={'Steps': 'steps'})
        cal  = pd.read_csv(RAW / f'{pid}_minuteCaloriesNarrow.csv',
                           parse_dates=['Wear_Time']).set_index('Wear_Time').rename(columns={'Calories': 'calories'})
        intn = pd.read_csv(RAW / f'{pid}_minuteIntensitiesNarrow.csv',
                           parse_dates=['Wear_Time']).set_index('Wear_Time').rename(columns={'Intensity': 'intensity'})
        met  = pd.read_csv(RAW / f'{pid}_minuteMETsNarrow.csv',
                           parse_dates=['Wear_Time']).set_index('Wear_Time').rename(columns={'METs': 'met'})
        hr   = pd.read_csv(RAW / f'{pid}_heartrate_1min.csv',
                           parse_dates=['Wear_Time']).set_index('Wear_Time').rename(columns={'Value': 'hr'})

        df = step.join([cal, intn, met], how='outer').join(hr, how='left')

        hourly = df.resample('h').agg({
            'steps':     'sum',
            'calories':  'sum',
            'intensity': 'mean',
            'met':       'mean',
            'hr':        'mean',
        })

        # Sleep stages from 30-second data → minutes per hour
        ss_path = RAW / f'{pid}_30secondSleepStages.csv'
        if ss_path.exists():
            ss = pd.read_csv(ss_path, parse_dates=['Wear_Time'])
            ss['hour'] = ss['Wear_Time'].dt.floor('h')
            sleep_h = ss.groupby('hour').apply(lambda g: pd.Series({
                'wake_min':  (g['SleepStage'] == 'wake').sum()  / 2.0,
                'light_min': (g['SleepStage'] == 'light').sum() / 2.0,
                'deep_min':  (g['SleepStage'] == 'deep').sum()  / 2.0,
                'rem_min':   (g['SleepStage'] == 'rem').sum()   / 2.0,
            }), include_groups=False)
            hourly = hourly.join(sleep_h, how='left')
        else:
            for c in ['wake_min', 'light_min', 'deep_min', 'rem_min']:
                hourly[c] = np.nan

        # Keep only hours where device was worn (steps file defines wear)
        worn = df['steps'].resample('h').count() > 0
        hourly = hourly[worn]

        arr = hourly[FEAT_COLS].values.astype(np.float32)   # (N, 9), NaN for missing
        ts  = hourly.index.astype(np.int64).values // 10**9

        np.save(out_path, arr)
        np.save(ts_path, ts)
        return (pid, len(arr), str(hourly.index.min()), str(hourly.index.max()))

    except Exception as e:
        return (pid, -1, str(e), None)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    pids = sorted(set(
        f.name.split('_heartrate_1min')[0]
        for f in RAW.glob('*_heartrate_1min.csv')
    ))
    print(f'Subjects to process: {len(pids)}')

    rows = []
    with Pool(N_WORKERS) as pool:
        for i, (pid, n, start, end) in enumerate(pool.imap_unordered(process_subject, pids)):
            rows.append({'subject_id': pid, 'n_hours': n, 'start': start, 'end': end})
            if (i + 1) % 500 == 0:
                ok = sum(1 for r in rows if r['n_hours'] > 0)
                print(f'  {i+1}/{len(pids)}  ok={ok}', flush=True)

    df = pd.DataFrame(rows).sort_values('subject_id')
    df.to_csv(CACHE / 'manifest.csv', index=False)

    ok = (df['n_hours'] > 0).sum()
    print(f'\nDone: {ok}/{len(pids)} succeeded')
    print(f'Hours/subject: median={df[df.n_hours>0].n_hours.median():.0f}  '
          f'min={df[df.n_hours>0].n_hours.min()}  max={df[df.n_hours>0].n_hours.max()}')
    print(f'Cache: {CACHE}')


if __name__ == '__main__':
    main()
