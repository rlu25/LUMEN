"""Default paths and dimensions for the ABCD genome-wide pipeline."""

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
CHECKPOINTS = HERE / "checkpoints"

GENETICS_ROOT = Path(os.environ.get("LUMEN_GENETICS_ROOT", "data/genetics"))
PLINK_PREFIX = GENETICS_ROOT / "genotype_microarray" / "merged_chroms"
IMPUTED_VCF_DIR = GENETICS_ROOT / "genotype_microarray" / "imputed"
REFERENCE_FASTA = GENETICS_ROOT / "sequencing" / "supportfiles" / "GRCh38.no_alt_analysis_set.fa"
NT_REPOSITORY = Path(
    os.environ.get("LUMEN_NT_REPOSITORY", "external/nucleotide-transformer")
)
LIFTOVER_CHAIN = Path(
    os.environ.get(
        "LUMEN_LIFTOVER_CHAIN",
        str(NT_REPOSITORY / "abcd_pipeline_new/liftover_data/hg19ToHg38.over.chain.gz"),
    )
)
SPLIT_FILE = ROOT / "data_splits" / "master_subject_split.csv"
FUSION_GENETIC_DIR = ROOT / "multimodal_fusion" / "embeddings" / "genetic"

AUTOSOMES = tuple(str(i) for i in range(1, 23))
SEQUENCE_LENGTH = 1000
NT_MODEL_NAME = "50M_multi_species_v2"
NT_EMBEDDING_LAYER = 12
NT_DIM = 512
PCA_DIM = 64
REGION_MODEL_DIM = 256
N_LATENTS = 32
MAX_REGIONS = 4096
