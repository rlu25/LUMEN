# LUMEN

This repository accompanies the manuscript **“A Shared Longitudinal Multimodal Foundation Representation for Prospective Adolescent Outcome Prediction.”**

## Overview

LUMEN is a self-supervised framework for learning a shared participant representation from incomplete longitudinal multimodal data. It integrates five complementary data modalities:

- resting-state functional connectivity (rsFC),
- task-fMRI BOLD time series,
- structural MRI (sMRI),
- Fitbit activity, and
- genetics.

Modality-specific encoders first transform each input into a compact representation. The longitudinal multimodal fusion model then summarizes the available visits within each modality and combines the modality summaries through a learned fusion token. The resulting task-independent participant vector is shared across downstream prediction tasks.

The study uses information from baseline, year 2, and year 4 to prospectively predict year-6 outcomes:

- body mass index,
- general cognition,
- internalizing symptoms,
- externalizing symptoms, and
- suicidal ideation.

## Framework

The workflow contains five stages:

1. **Modality-specific representation learning:** independently pretrain encoders using masked reconstruction or contrastive learning objectives.
2. **Longitudinal multimodal fusion:** summarize available visits within each modality and learn one shared participant representation.
3. **Prospective outcome prediction:** train outcome-specific prediction heads using the shared representation.
4. **Modality and visit attribution:** evaluate fixed models under different input coalitions and calculate exact Shapley values.
5. **Sensitivity analysis:** evaluate predictive value beyond measured demographic and socioeconomic covariates.

## Repository status

This repository currently provides information about the project. Source code, configuration files, and detailed execution instructions will be added after the manuscript and release materials have completed review.

Participant-level data, model checkpoints, embeddings, and predictions are not included.

## Data availability

This study uses controlled-access data from the Adolescent Brain Cognitive Development (ABCD) Study. Qualified investigators can find current access information through the [ABCD Study data-sharing page](https://abcdstudy.org/scientists/data-sharing/).

ABCD participant-level data cannot be redistributed through this repository.

## Citation

The complete manuscript citation and DOI will be added when available.
