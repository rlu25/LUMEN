#!/usr/bin/env python3 -u
"""Run the frozen Nucleotide Transformer and save ALT-minus-REF SNP vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import (ARTIFACTS, NT_EMBEDDING_LAYER, NT_MODEL_NAME,
                     NT_REPOSITORY)

sys.path.insert(0, str(NT_REPOSITORY))
from nucleotide_transformer.pretrained import get_pretrained_model


def embed(forward, parameters, tokenizer, key, sequences):
    token_ids = [item[1] for item in tokenizer.batch_tokenize(sequences)]
    outputs = forward.apply(parameters, key, jnp.asarray(token_ids, dtype=jnp.int32))
    values = np.asarray(outputs[f"embeddings_{NT_EMBEDDING_LAYER}"])
    return values[:, 1:, :].mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, default=ARTIFACTS / "sequences")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "delta")
    parser.add_argument("--chromosomes", nargs="*", default=[str(i) for i in range(1, 23)])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading frozen {NT_MODEL_NAME}")
    parameters, forward, tokenizer, nt_config = get_pretrained_model(
        model_name=NT_MODEL_NAME,
        embeddings_layers_to_save=(NT_EMBEDDING_LAYER,),
    )
    forward = hk.transform(forward)
    key = jax.random.PRNGKey(0)
    print(f"NT embed_dim={nt_config.embed_dim}")

    for chromosome in args.chromosomes:
        source = args.sequence_dir / f"chr{chromosome}.parquet"
        output = args.output_dir / f"chr{chromosome}.npz"
        if not source.exists():
            continue
        if output.exists() and not args.overwrite:
            print(f"chr{chromosome}: already exists, skipping")
            continue
        table = pd.read_parquet(source)
        chunks = []
        for start in range(0, len(table), args.batch_size):
            batch = table.iloc[start:start + args.batch_size]
            ref = embed(forward, parameters, tokenizer, key, batch["ref_seq"].tolist())
            alt = embed(forward, parameters, tokenizer, key, batch["alt_seq"].tolist())
            chunks.append((alt - ref).astype(np.float16))
            if (start // args.batch_size) % 100 == 0:
                print(f"chr{chromosome}: {min(start + args.batch_size, len(table)):,}/{len(table):,}", flush=True)
        delta = np.concatenate(chunks) if chunks else np.empty((0, nt_config.embed_dim), np.float16)
        np.savez_compressed(
            output,
            variant_index=table["variant_index"].to_numpy(np.int64),
            region_id=table["region_id"].to_numpy(np.int64),
            dosage_flip=table["dosage_flip"].to_numpy(bool),
            rsid=table["rsid"].astype(str).to_numpy(),
            delta=delta,
        )
        print(f"chr{chromosome}: saved {delta.shape} -> {output}")


if __name__ == "__main__":
    main()
