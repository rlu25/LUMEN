#!/usr/bin/env python3 -u
"""Build authoritative GRCh38 REF/ALT sequence windows for the fixed SNP panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pysam
from pyliftover import LiftOver

from .config import (
    ARTIFACTS, IMPUTED_VCF_DIR, LIFTOVER_CHAIN, REFERENCE_FASTA,
    SEQUENCE_LENGTH,
)


def liftover_position(liftover, chromosome: str, position37: int):
    contig = f"chr{chromosome}"
    mapped = liftover.convert_coordinate(contig, position37 - 1)
    if not mapped or mapped[0][0] != contig or mapped[0][2] != "+":
        return None
    return int(mapped[0][1]) + 1


def fetch_snp(vcf, contig: str, position: int):
    for record in vcf.fetch(contig, position - 1, position):
        if record.pos != position or len(record.alts or ()) != 1:
            continue
        ref, alt = record.ref, record.alts[0]
        if len(ref) == len(alt) == 1:
            return ref.upper(), alt.upper()
    return None


def dosage_orientation(a0: str, a1: str, ref: str, alt: str):
    """Return whether a1-count dosage must be flipped to become ALT dosage.

    pandas-plink's default matrix counts BIM a1. Strand-ambiguous A/T and C/G
    sites are dropped because allele letters alone cannot establish orientation.
    """
    a0, a1, ref, alt = (str(x).upper() for x in (a0, a1, ref, alt))
    if {ref, alt} in ({"A", "T"}, {"C", "G"}):
        return None
    complement = str.maketrans("ACGT", "TGCA")
    if (a0, a1) == (ref, alt):
        return False
    if (a0, a1) == (alt, ref):
        return True
    if (a0.translate(complement), a1.translate(complement)) == (ref, alt):
        return False
    if (a0.translate(complement), a1.translate(complement)) == (alt, ref):
        return True
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=ARTIFACTS / "panel.csv")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "sequences")
    parser.add_argument("--chromosomes", nargs="*", default=[str(i) for i in range(1, 23)])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.panel, dtype={"chrom": str})
    fasta = pysam.FastaFile(str(REFERENCE_FASTA))
    liftover = LiftOver(str(LIFTOVER_CHAIN))
    half = SEQUENCE_LENGTH // 2

    for chromosome in args.chromosomes:
        output = args.output_dir / f"chr{chromosome}.parquet"
        if output.exists() and not args.overwrite:
            print(f"chr{chromosome}: already exists, skipping")
            continue
        rows = panel[panel["chrom"] == chromosome]
        vcf_path = IMPUTED_VCF_DIR / f"chr{chromosome}_dose.vcf.gz"
        if rows.empty or not vcf_path.exists():
            print(f"chr{chromosome}: no panel rows or VCF")
            continue
        vcf = pysam.VariantFile(str(vcf_path))
        contig = f"chr{chromosome}"
        output_rows, dropped = [], 0
        for number, row in enumerate(rows.itertuples(index=False), 1):
            position38 = liftover_position(liftover, chromosome, int(row.pos_grch37))
            alleles = fetch_snp(vcf, contig, position38) if position38 is not None else None
            if alleles is None:
                dropped += 1; continue
            ref, alt = alleles
            start = position38 - 1 - half
            sequence = fasta.fetch(contig, start, start + SEQUENCE_LENGTH).upper()
            if len(sequence) != SEQUENCE_LENGTH or sequence[half] != ref:
                dropped += 1; continue
            flip = dosage_orientation(row.plink_a0, row.plink_a1, ref, alt)
            if flip is None:
                dropped += 1; continue
            output_rows.append({
                "variant_index": int(row.variant_index),
                "region_id": int(row.region_id),
                "chrom": chromosome,
                "pos_grch37": int(row.pos_grch37),
                "pos_grch38": position38,
                "rsid": row.rsid,
                "ref": ref,
                "alt": alt,
                "dosage_flip": bool(flip),
                "ref_seq": sequence,
                "alt_seq": sequence[:half] + alt + sequence[half + 1:],
            })
            if number % 10000 == 0:
                print(f"chr{chromosome}: {number:,}/{len(rows):,}", flush=True)
        pd.DataFrame(output_rows).to_parquet(output, index=False)
        print(f"chr{chromosome}: saved={len(output_rows):,}, dropped={dropped:,} -> {output}")


if __name__ == "__main__":
    main()
