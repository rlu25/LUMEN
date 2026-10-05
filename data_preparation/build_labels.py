#!/usr/bin/env python3
"""
Build downstream task label CSV.
Output: labels.csv in this repository. Review the generated table before using
        it in downstream experiments.
Columns: subid, visit, age_months, sex, bmi, bmi_z, bmi_biv, obesity,
         g_factor, internalizing, externalizing, sui

visit: Y0 / Y2 / Y4 / Y6 only. General phenotype rows come from
       ABCD_longitudinal/phenotype at ses-00A/02A/04A/06A; SUI additionally
       reads ses-03A/05A to cover the post-Y2 and post-Y4 outcome windows. The previous
       version of this script
       read an older set of tabular CSV files,
       an older pull that only goes up to a 4-year follow-up and only
       annually (Y0-Y4, no Y6) -- that source simply doesn't have Y6 data at
       all. ABCD_longitudinal/phenotype is a newer, richer pull that already
       has Y6. Y3 and Y5 are used only to construct the post-Y2 and post-Y4
       SUI outcome windows; odd-year rows are not emitted as model input or
       general label rows.
subid: NDAR_INV* format (participant_id in the source tables is already
       this format, unlike the old TabularBehavior/core src_subject_id
       column, so no prefix rewriting is needed here)

Sources (all files below ``LUMEN_PHENOTYPE_DIR``):
  - age_months    : ab_p_demo.parquet   ab_p_demo_age (reported in YEARS,
                    not months -- multiplied by 12 here to match this
                    project's age_months convention elsewhere)
  - sex (0/1)     : ab_g_stc.parquet    ab_g_stc__cohort_sex; 1(M)->0, 2(F)->1;
                    one row per subject (no session_id), broadcast to all visits
  - bmi           : ph_y_anthr.parquet  ph_y_anthr__weight_mean (lbs),
                    ph_y_anthr__height_mean (in); bmi = weight*703/height**2;
                    finite positive measurements are retained. Extreme values
                    are flagged with bmi_biv rather than deleted by a fixed
                    raw-BMI cutoff.
  - bmi_z         : 2022 Extended CDC BMI-for-age-and-sex z-score. The CDC 2000
                    LMS equation is used through P95; the Extended CDC sigma
                    method is used above P95 so severe obesity is not compressed.
  - bmi_biv       : CDC biological-implausibility flag based on modified BMI z:
                    -1 below -4, 0 within range, +1 above +8. A flag prompts
                    review but does not automatically discard the measurement.
  - obesity (0/1) : BMI >= the age- and sex-specific CDC 95th percentile.
  - g_factor      : computed HERE now (previously merged from an external
                    snapshot -- see git history / old docstring). PC1 of 5 NIH
                    Toolbox age-corrected subtests (Flanker, PicSeq, PicVocab,
                    Pattern Comparison, Reading -- List Sorting dropped, see
                    comment below), pooled Y0+Y2+Y4+Y6,
                    z-scored, then projected per row. Source:
                    nc_y_nihtb.parquet.
                    NOTE: this replaces a merge from
                    dingyi/abcd_long/v14/compute_gfactor_longitudinal.py's
                    output, which had a visit-mislabeling bug -- its
                    VISIT_MAP mapped ses-04A (actually Y4 data) to the label
                    'Y2', and ses-06A (actually Y6 data) to the label 'Y4',
                    while never referencing ses-02A (the real Y2 data) at
                    all. Every prior g_factor value tagged Y2 was really Y4
                    data, and every Y4 value was really Y6 data; only Y0 was
                    correctly labeled. This script's own PCA below uses the
                    correct ses-00A/02A/04A/06A -> Y0/Y2/Y4/Y6 mapping.
  - internalizing : mh_p_cbcl.parquet   mh_p_cbcl__synd__int_tscore (same
                    underlying CBCL syndrome T-score as the old
                    cbcl_scr_syn_internal_t column, just this pull's naming)
  - externalizing : mh_p_cbcl.parquet   mh_p_cbcl__synd__ext_tscore
  - sui (0/1)     : qualifying substance use over two prospective windows:
                    after Y2 through Y4 (Y3/Y4 interviews, stored on Y4) and
                    after Y4 through Y6 (Y5/Y6 interviews, stored on Y6).
                    Positive means
                    >=1 standard alcohol drink, cannabis/nicotine use beyond
                    the low-level sip/puff screen, or any reported use of an
                    included other substance. Use before the start of a window
                    does not alter its target. A positive at either annual
                    interview is sufficient; a negative requires adequate
                    observations at both interviews.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT      = Path(__file__).resolve().parents[1]
PHENO     = Path(os.environ.get('LUMEN_PHENOTYPE_DIR', 'data/phenotype'))
OUT       = Path(__file__).parent / 'labels.csv'
LMS_TABLE = ROOT / 'data' / 'cdc_bmiagerev_lms.csv'

VISIT_MAP = {'ses-00A': 'Y0', 'ses-02A': 'Y2', 'ses-04A': 'Y4', 'ses-06A': 'Y6'}


def to_ndar_inv(participant_id):
    """This pull's participant_id is 'sub-XXXX'; every other script in this
    project (fitbit/bold/rsfc/smri) joins on 'NDAR_INVXXXX' instead."""
    return 'NDAR_INV' + participant_id.str[len('sub-'):]


def load(name, cols):
    df = pd.read_parquet(PHENO / f'{name}.parquet',
                         columns=['participant_id', 'session_id'] + cols)
    df = df[df['session_id'].isin(VISIT_MAP)].copy()
    df['visit'] = df['session_id'].map(VISIT_MAP)
    df['subid'] = to_ndar_inv(df['participant_id'])
    return df.drop(columns=['session_id', 'participant_id'])


# ── Age (years -> months) ────────────────────────────────────────────────────
lt = load('ab_p_demo', ['ab_p_demo_age'])
lt['age_months'] = lt['ab_p_demo_age'] * 12
lt = lt[['subid', 'visit', 'age_months']]

# ── Sex (one row per subject, no session_id -- broadcast to all visits) ─────
# ab_g_stc__cohort_sex is stored as string ('1'/'2'), not int -- confirmed via
# .dtype/value_counts; comparing against int literals silently matches nothing.
sex = pd.read_parquet(PHENO / 'ab_g_stc.parquet',
                      columns=['participant_id', 'ab_g_stc__cohort_sex'])
sex['subid'] = to_ndar_inv(sex['participant_id'])
sex['cohort_sex_num'] = pd.to_numeric(sex['ab_g_stc__cohort_sex'], errors='coerce')
sex = sex[sex.cohort_sex_num.isin([1, 2])].copy()
sex['sex'] = (sex['cohort_sex_num'] == 2).astype(float)   # 1(M)->0, 2(F)->1
sex = sex[['subid', 'sex']]

# ── BMI + BMI-for-age-and-sex z-score ───────────────────────────────────────
anthro = load('ph_y_anthr', ['ph_y_anthr__height_mean', 'ph_y_anthr__weight_mean'])
anthro['bmi'] = anthro['ph_y_anthr__weight_mean'] * 703 / anthro['ph_y_anthr__height_mean'] ** 2
anthro.loc[~np.isfinite(anthro.bmi) | (anthro.bmi <= 0), 'bmi'] = np.nan
bmi = anthro[['subid', 'visit', 'bmi']]

lms = pd.read_csv(LMS_TABLE)   # CDC LMS reference: Sex(1=M,2=F), Agemos, L, M, S


def bmi_z_score(age_months, sex01, bmi_val, lms):
    """2022 Extended CDC BMI-for-age-and-sex z-score.

    CDC 2000 LMS is retained at or below P95. Above P95, use the 2022
    Extended CDC distribution. For fractional-month ages CDC specifies
    sex-specific regressions for sigma (age is in years).
    """
    if pd.isna(age_months) or pd.isna(sex01) or pd.isna(bmi_val):
        return np.nan
    grid = lms[lms.Sex == (2 if sex01 == 1 else 1)]
    L = np.interp(age_months, grid.Agemos, grid.L)
    M = np.interp(age_months, grid.Agemos, grid.M)
    S = np.interp(age_months, grid.Agemos, grid.S)
    p95 = np.interp(age_months, grid.Agemos, grid.P95)
    if bmi_val > p95:
        age_years = age_months / 12.0
        if sex01 == 1:  # girls
            sigma = 0.8334 + 0.3712 * age_years - 0.0011 * age_years ** 2
        else:           # boys
            sigma = 0.3728 + 0.5196 * age_years - 0.0091 * age_years ** 2
        percentile = 90.0 + 10.0 * norm.cdf((bmi_val - p95) / sigma)
        probability = np.clip(percentile / 100.0, np.nextafter(0.0, 1.0),
                              np.nextafter(1.0, 0.0))
        return float(norm.ppf(probability))
    if L == 0:
        return float(np.log(bmi_val / M) / S)
    return float(((bmi_val / M) ** L - 1) / (L * S))


def bmi_biv_flag(age_months, sex01, bmi_val, lms):
    """CDC modified-z BIV flag: low <-4, high >+8, otherwise zero."""
    if pd.isna(age_months) or pd.isna(sex01) or pd.isna(bmi_val):
        return np.nan
    grid = lms[lms.Sex == (2 if sex01 == 1 else 1)]
    L = np.interp(age_months, grid.Agemos, grid.L)
    M = np.interp(age_months, grid.Agemos, grid.M)
    S = np.interp(age_months, grid.Agemos, grid.S)
    bmi_minus2 = M * (1 - 2 * L * S) ** (1 / L)
    bmi_plus2 = M * (1 + 2 * L * S) ** (1 / L)
    if bmi_val < M:
        modified_z = (bmi_val - M) / ((M - bmi_minus2) / 2)
    else:
        modified_z = (bmi_val - M) / ((bmi_plus2 - M) / 2)
    if modified_z < -4:
        return -1.0
    if modified_z > 8:
        return 1.0
    return 0.0


def bmi_obesity_flag(age_months, sex01, bmi_val, lms):
    """CDC obesity classification: BMI at or above sex/age-specific P95."""
    if pd.isna(age_months) or pd.isna(sex01) or pd.isna(bmi_val):
        return np.nan
    grid = lms[lms.Sex == (2 if sex01 == 1 else 1)]
    p95 = np.interp(age_months, grid.Agemos, grid.P95)
    return float(bmi_val >= p95)


# ── g-factor: PC1 of 5 NIH Toolbox subtests, pooled Y0+Y2+Y4+Y6 ────────────
# List Sorting (lswmt) was dropped from the original 6-subtest set: it's a
# genuine ABCD protocol gap at Y2 specifically (15/10,827 subjects have a
# score there, vs 9,100-11,700 at every other visit) -- not a data-quality
# issue to impute around. Using the same 5 subtests at every visit keeps the
# PCA's measurement model consistent across the whole longitudinal series,
# rather than silently swapping in a 6th subtest only where it happens to be
# available (which is very likely why the previous version of this pipeline
# skipped ses-02A/Y2 entirely instead of documenting the gap).
SUBTESTS = [
    'nc_y_nihtb__flnkr__agecor_score',
    'nc_y_nihtb__picsq__agecor_score',
    'nc_y_nihtb__picvcb__agecor_score',
    'nc_y_nihtb__pttcp__agecor_score',
    'nc_y_nihtb__readr__agecor_score',
]
nihtb = load('nc_y_nihtb', SUBTESTS)
nihtb = nihtb[nihtb[SUBTESTS].notna().all(axis=1)].reset_index(drop=True)

X_scaled = StandardScaler().fit_transform(nihtb[SUBTESTS].values.astype(np.float64))
pca = PCA(n_components=1).fit(X_scaled)
if pca.components_[0].mean() < 0:
    pca.components_[0] *= -1
gf = nihtb[['subid', 'visit']].copy()
gf['g_factor'] = pca.transform(X_scaled).ravel()
print(f'g_factor: {len(gf)} rows, PC1 variance explained '
      f'{pca.explained_variance_ratio_[0]*100:.1f}%')
for v, grp in gf.groupby('visit'):
    print(f'  {v}: {len(grp)}')

# ── CBCL ─────────────────────────────────────────────────────────────────────
cbcl = load('mh_p_cbcl', ['mh_p_cbcl__synd__int_tscore', 'mh_p_cbcl__synd__ext_tscore'])
cbcl = cbcl.rename(columns={
    'mh_p_cbcl__synd__int_tscore': 'internalizing',
    'mh_p_cbcl__synd__ext_tscore': 'externalizing',
})

# ── SUI: qualifying substance use during two prospective windows ───────────
# Longitudinal SUI fields describe use since the preceding yearly session.
# Therefore, Y3/Y4 cover the outcome window after Y2 through Y4, while Y5/Y6
# cover the outcome window after Y4 through Y6.
SUI_WINDOWS = {
    'Y4': ('ses-03A', 'ses-04A'),
    'Y6': ('ses-05A', 'ses-06A'),
}

# The screen variables are used only to establish whether a branched domain
# was adequately assessed. A low-level sip/puff alone is not a positive.
SUI_DOMAINS = {
    'alcohol': (
        'su_y_sui__use__alc__sip_001__l',
        ['su_y_sui__use__alc_001__l'],
    ),
    'cannabis': (
        'su_y_sui__use__mj__puff_001__l',
        [
            'su_y_sui__use__mj__blunt_001__l',
            'su_y_sui__use__mj__conc__smoke_001__l',
            'su_y_sui__use__mj__conc__vape_001__l',
            'su_y_sui__use__mj__drink_001__l',
            'su_y_sui__use__mj__edbl_001__l',
            'su_y_sui__use__mj__smoke_001__l',
            'su_y_sui__use__mj__synth_001__l',
            'su_y_sui__use__mj__tinc_001__l',
            'su_y_sui__use__mj__vape_001__l',
        ],
    ),
    'nicotine': (
        'su_y_sui__use__nic__puff_001__l',
        [
            'su_y_sui__use__nic__chew_001__l',
            'su_y_sui__use__nic__cig_001__l',
            'su_y_sui__use__nic__cigar_001__l',
            'su_y_sui__use__nic__hookah_001__l',
            'su_y_sui__use__nic__pipe_001__l',
            'su_y_sui__use__nic__rplc_001__l',
            'su_y_sui__use__nic__vape_001__l',
        ],
    ),
}

SUI_OTHER_USE_COLS = [
    'su_y_sui__use__cath_001__l',
    'su_y_sui__use__cbd_001__l',
    'su_y_sui__use__coc_001__l',
    'su_y_sui__use__dxm_001__l',
    'su_y_sui__use__ghb_001__l',
    'su_y_sui__use__hall_001__l',
    'su_y_sui__use__inh_001__l',
    'su_y_sui__use__ket_001__l',
    'su_y_sui__use__mdma_001__l',
    'su_y_sui__use__meth_001__l',
    'su_y_sui__use__opi_001__l',
    'su_y_sui__use__othdrg_002__l',
    'su_y_sui__use__qc_001__l',
    'su_y_sui__use__roid_001__l',
    'su_y_sui__use__rxopi_001__l',
    'su_y_sui__use__rxsed_001__l',
    'su_y_sui__use__rxstim_001__l',
    'su_y_sui__use__salv_001__l',
    'su_y_sui__use__shroom_001__l',
    'su_y_sui__use__vape__flav_001__l',
]


def build_sui_window_target(target_visit, window_sessions):
    domain_columns = [
        column
        for screen, use_columns in SUI_DOMAINS.values()
        for column in [screen, *use_columns]
    ]
    columns = domain_columns + SUI_OTHER_USE_COLS
    substance = pd.read_parquet(
        PHENO / 'su_y_sui.parquet',
        columns=['participant_id', 'session_id', *columns],
    )
    substance = substance[
        substance['session_id'].isin(window_sessions)
    ].copy()
    substance['subid'] = to_ndar_inv(substance['participant_id'])
    values = substance[columns].apply(pd.to_numeric, errors='coerce')

    positive = pd.Series(False, index=substance.index)
    observed = pd.Series(True, index=substance.index)
    for screen, use_columns in SUI_DOMAINS.values():
        screen_value = values[screen]
        domain_use = values[use_columns]
        positive |= domain_use.eq(1).any(axis=1)
        observed &= screen_value.notna() & (
            screen_value.eq(0) | domain_use.notna().all(axis=1)
        )

    other_use = values[SUI_OTHER_USE_COLS]
    positive |= other_use.eq(1).any(axis=1)
    observed &= other_use.notna().all(axis=1)
    substance['visit_sui'] = np.where(
        positive, 1.0, np.where(observed, 0.0, np.nan)
    )

    per_visit = substance.pivot(
        index='subid', columns='session_id', values='visit_sui'
    ).reindex(columns=window_sessions)
    window_positive = per_visit.eq(1).any(axis=1)
    window_negative = per_visit.notna().all(axis=1) & per_visit.eq(0).all(axis=1)
    target = pd.Series(
        np.where(window_positive, 1.0, np.where(window_negative, 0.0, np.nan)),
        index=per_visit.index,
        name='sui',
    )
    return target.reset_index().assign(visit=target_visit), per_visit


sui_targets = []
sui_per_window = {}
for target_visit, window_sessions in SUI_WINDOWS.items():
    target, per_visit = build_sui_window_target(target_visit, window_sessions)
    sui_targets.append(target)
    sui_per_window[target_visit] = per_visit
sui = pd.concat(sui_targets, ignore_index=True)

# ── Merge ─────────────────────────────────────────────────────────────────────
df = lt.merge(sex, on='subid', how='left') \
       .merge(bmi, on=['subid', 'visit'], how='outer') \
       .merge(gf,   on=['subid', 'visit'], how='outer') \
       .merge(cbcl, on=['subid', 'visit'], how='outer') \
       .merge(sui,  on=['subid', 'visit'], how='outer')

df['bmi_z'] = df.apply(
    lambda r: bmi_z_score(r.age_months, r.sex, r.bmi, lms), axis=1)
df['bmi_biv'] = df.apply(
    lambda r: bmi_biv_flag(r.age_months, r.sex, r.bmi, lms), axis=1)
df['obesity'] = df.apply(
    lambda r: bmi_obesity_flag(r.age_months, r.sex, r.bmi, lms), axis=1)

df = df.sort_values(['subid', 'visit']).reset_index(drop=True)
df = df[['subid', 'visit', 'age_months', 'sex', 'bmi', 'bmi_z', 'bmi_biv',
         'obesity', 'g_factor', 'internalizing', 'externalizing', 'sui']]

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f'\nSaved {len(df):,} rows -> {OUT}')
print(f'\nNon-null per column:')
print(df.notna().sum().to_string())
print(f'\nRows per visit:')
print(df.visit.value_counts().sort_index())
print(f'\nSUI qualifying-use ascertainment:')
for target_visit, window_sessions in SUI_WINDOWS.items():
    per_visit = sui_per_window[target_visit]
    for session in window_sessions:
        s = per_visit[session]
        n = s.notna().sum()
        print(
            f'  {session}: N={n}  pos={int((s == 1).sum())} '
            f'({100 * (s == 1).sum() / n:.1f}%)'
        )
    s = df.loc[df.visit.eq(target_visit), 'sui']
    n = s.notna().sum()
    print(
        f'  {window_sessions[0]}->{window_sessions[-1]} target on {target_visit}: '
        f'N={n}  pos={int((s == 1).sum())} '
        f'({100 * (s == 1).sum() / n:.1f}%)'
    )
print(f'\nSample (Y0):')
print(df[df.visit == 'Y0'].head(5).to_string())
