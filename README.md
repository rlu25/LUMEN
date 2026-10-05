# LUMEN

Code for **A Shared Longitudinal Multimodal
Foundation Representation for Prospective Adolescent Outcome Prediction**.

LUMEN pretrains modality encoders independently, fuses longitudinal modality
representations into a task-independent participant vector, and uses that
shared vector for prospective outcome prediction. Inputs are resting-state
functional connectivity (rsFC), task-fMRI BOLD time series, structural MRI
(sMRI), Fitbit activity, and phenotype-independent genetic tokens. The primary
prediction contract uses Y0/Y2/Y4 inputs to predict Y6 outcomes.

## Repository layout

```text
.
├── data/                         Public, non-participant reference files
├── data_preparation/             Label and split construction
├── data_splits/                  Generated split tables (not distributed)
├── modality_pretraining/
│   ├── rsfc/                     rsFC masked reconstruction
│   ├── fitbit/                   Fitbit preprocessing and pretraining
│   ├── bold/                     BOLD masked/contrastive pretraining
│   ├── genetics/                 Phenotype-independent genetic tokens
│   └── smri/                     sMRI checkpoint instructions and MAPSeg source link
├── multimodal_fusion/            Shared multimodal foundation model
├── downstream_prediction/        Frozen-head training and Stage-2 fine-tuning
├── CHECKPOINTS.md                Released artifacts, provenance, and hashes
├── requirements.txt              Core dependencies
└── requirements-genetics.txt     Additional genetics dependencies
```

Participant-level inputs, labels, split tables, embeddings, manifests, and
predictions are intentionally excluded. Pretrained checkpoints and normalizers are distributed through
[Google Drive](https://drive.google.com/drive/folders/1Z-HgIAm5jvh6RdsZmjV4UkYkKAZxg9q1?usp=sharing), separately from this repository.

## Installation

Python 3.10 or later is recommended. CUDA-capable PyTorch is
required for practical pretraining and BOLD extraction.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the pretrained artifacts from [Google Drive](https://drive.google.com/drive/folders/1Z-HgIAm5jvh6RdsZmjV4UkYkKAZxg9q1?usp=sharing)
and place them at the repository-relative paths listed in
[CHECKPOINTS.md](CHECKPOINTS.md) before running extraction or downstream
prediction. Keep the downloaded checkpoints and normalizers outside Git.

For the genetics workflow:

```bash
python -m pip install -r requirements-genetics.txt
```

Install Nucleotide Transformer separately and set `LUMEN_NT_REPOSITORY` to
that checkout.

## Controlled data and paths

ABCD participant-level data are controlled-access and are not redistributed.
Obtain authorization through the ABCD Study data-sharing process and comply
with the applicable data-use agreement.

Run commands from the repository root. Default inputs live under `data/` and
can be replaced with environment variables:

```bash
export LUMEN_PHENOTYPE_DIR=/path/to/phenotype
export LUMEN_DEMOGRAPHICS_CSV=/path/to/abcd_y_lt.csv
export LUMEN_RSFC_DIR=/path/to/rsfc
export LUMEN_BOLD_DIR=/path/to/bold
export LUMEN_SMRI_EMBEDDING_DIR=/path/to/smri_embeddings
export LUMEN_FITBIT_RAW_DIR=/path/to/fitbit_raw
export LUMEN_FITBIT_CACHE=/path/to/fitbit_hourly_cache
export LUMEN_FITBIT_TABLE=/path/to/nt_y_fitb_act_d.csv
export LUMEN_GENETICS_ROOT=/path/to/genetics
export LUMEN_NT_REPOSITORY=/path/to/nucleotide-transformer
export LUMEN_LIFTOVER_CHAIN=/path/to/hg19ToHg38.over.chain.gz
```

Expected high-level inputs:

- rsFC: `y0_X.npy` through `y6_X.npy`, matching `*_pids.npy`, and
  `roi_pairs.npy`.
- BOLD: `sub-<id>_Y0.npy` through `sub-<id>_Y6.npy`, each with 366 ROIs.
- Fitbit: source tables and/or the preprocessed hourly cache.
- sMRI: `NDAR_INV<id>_Y<visit>.npy`, each a 512-dimensional vector.
- Genetics: PLINK/imputed VCF/reference FASTA inputs.
- Phenotypes and demographics: the controlled tables consumed by
  `data_preparation/`.

## 1. Build labels and splits

```bash
python -m data_preparation.build_labels
python -m data_preparation.create_master_split
```

Review `data_preparation/labels.csv` and
`data_splits/master_subject_split.csv` before model training. The master split
keeps families together; training families update weights, validation families
select checkpoints, and test families remain held out.

An optional family-safe site holdout can be generated with:

```bash
python -m data_preparation.create_site_holdout_split
```

## 2. Pretrain modality encoders

### rsFC

```bash
python -m modality_pretraining.rsfc.pretrain \
  --epochs 2000 --batch-size 32 --lr 1e-4 --mask-ratio 0.30
```

### Fitbit

```bash
python -m modality_pretraining.fitbit.preprocess
python -m modality_pretraining.fitbit.pretrain \
  --epochs 2000 --batch-size 256 --lr 3e-4 --drop-rate 0.15
```

### BOLD

```bash
python -m modality_pretraining.bold.pretrain \
  --epochs 2000 --batch-size 16 --lr 3e-4
```

### Structural MRI

The sMRI source implementation is not duplicated here. Use the upstream
[MAPSeg repository](https://github.com/XuzheZ/MAPSeg/tree/main). Download the project checkpoint from Google Drive and place it at
`modality_pretraining/smri/checkpoints/best_model.pth`; see
`modality_pretraining/smri/README.md` and `CHECKPOINTS.md`.

The multimodal code consumes already exported 512-dimensional sMRI vectors via
`LUMEN_SMRI_EMBEDDING_DIR`; it does not invoke MAPSeg internally.

### Genetics

```bash
python -m modality_pretraining.genetics.prepare_panel
python -m modality_pretraining.genetics.build_sequences
python -m modality_pretraining.genetics.extract_delta_embeddings
python -m modality_pretraining.genetics.build_regional_features
python -m modality_pretraining.genetics.pretrain
python -m modality_pretraining.genetics.extract_subject_tokens
```

The SNP panel and all normalization/PCA fitting are derived from training
families only. Genetic inputs to fusion are phenotype-independent.

## 3. Extract modality representations

```bash
python -m multimodal_fusion.extract_modality_embeddings rsfc fitbit bold smri
```

This loads the released encoder-only rsFC, Fitbit, and BOLD checkpoints and
links external sMRI vectors into `multimodal_fusion/embeddings/`. The genetics
export command above writes generic genetic tokens to the same contract.

## 4. Train and export the shared representation

With all five modalities:

```bash
python -m multimodal_fusion.train \
  --modalities rsfc fitbit smri bold genetic
```

Without genetics:

```bash
python -m multimodal_fusion.train \
  --modalities rsfc fitbit smri bold
```

Stage 2 summarizes visits within each modality, fuses the available modality
summaries with a learned `[FUSE]` token, and reconstructs masked inputs. Export
prospective Y0/Y2/Y4 subject vectors with:

```bash
python -m multimodal_fusion.extract_subject_embeddings
```

The extractor writes `multimodal_fusion/subject_embeddings/`, metadata, a
manifest, and a `.complete` marker. Downstream training rejects an incomplete
handoff.

## 5. Train downstream tasks

Train the five frozen-representation heads:

```bash
python -m downstream_prediction.train \
  bmi g_factor internalizing externalizing sui
```

The outcomes are BMI z-score, general cognition, internalizing symptoms,
externalizing symptoms, and substance-use initiation (`sui`). Validation loss
selects each checkpoint; the selected model is then evaluated on the requested
evaluation split.

To fine-tune Stage 2 with a fresh task predictor:

```bash
python -m downstream_prediction.finetune \
  bmi g_factor internalizing externalizing sui \
  --fusion-checkpoint multimodal_fusion/checkpoints/best.pt \
  --reference-output-dir downstream_prediction/runs \
  --head-init fresh
```

This keeps the modality token producers and reconstruction decoder frozen
while training the Stage-2 encoder and a newly initialized predictor. The
reference directory must be a completed local frozen-head run from the command
immediately above; its cohort and predictions enforce a matched comparison.

## Reproducibility safeguards

- Normalization is fit on training participants only.
- Related participants remain in the same partition.
- Y6 is excluded from prospective model inputs.
- Missing visits/modalities are attention-masked, not treated as observed
  zeros.
- Checkpoint selection uses validation objectives, not held-out test scores.
- Generic fusion accepts only phenotype-independent genetic tokens.
- Derived participant-level data and predictions remain outside Git.

## Released weights

Download pretrained weights and normalizers from
[Google Drive](https://drive.google.com/drive/folders/1Z-HgIAm5jvh6RdsZmjV4UkYkKAZxg9q1?usp=sharing). Checkpoint binaries are not included
in this GitHub repository. See [CHECKPOINTS.md](CHECKPOINTS.md) for destination
paths, artifact roles, and SHA-256 hashes.

## Citation

Manuscript citation information will be added when available. If you use the
sMRI implementation, also cite MAPSeg as requested by its upstream repository.
