#!/usr/bin/env python3 -u
"""Extract frozen shared subject embeddings for prospective prediction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
from modality_pretraining.common import load_split_map

from multimodal_fusion.data import DEFAULT_EMBEDDING_DIR, MultiModalDataset, collate_tokens
from multimodal_fusion.model import MultimodalFusionMAE


PROSPECTIVE_VISITS = ("Y0", "Y2", "Y4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=HERE / "checkpoints" / "best.pt"
    )
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "subject_embeddings"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {args.output_dir}\n"
            "Choose a new --output-dir."
        )

    completion_marker = args.checkpoint.parent / ".complete"
    if not completion_marker.exists():
        raise FileNotFoundError(
            f"Incomplete fusion training: {completion_marker} is missing"
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MultimodalFusionMAE(**checkpoint["model_config"])
    if model.pooling_mode != "fusion_token":
        raise ValueError(
            f"{args.checkpoint} uses pooling_mode={model.pooling_mode!r}; "
            "foundation extraction requires 'fusion_token'"
        )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    split_map = load_split_map()
    datasets = []
    for split in ("train", "val", "test"):
        dataset = MultiModalDataset(
            args.embedding_dir, split_map, split,
            tuple(checkpoint["modalities"]), PROSPECTIVE_VISITS,
            normalizer=checkpoint["normalizer"],
        )
        datasets.append((split, dataset))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as manifest:
        writer = csv.writer(manifest)
        writer.writerow(["subid", "split", "n_tokens", "embedding_file"])
        with torch.no_grad():
            for split, dataset in datasets:
                loader = DataLoader(
                    dataset, args.batch_size, shuffle=False, num_workers=args.workers,
                    collate_fn=collate_tokens,
                )
                offset = 0
                for raw, mod_ids, visit_ids, slot_ids, padding in loader:
                    raw, mod_ids = raw.to(device), mod_ids.to(device)
                    visit_ids, slot_ids = visit_ids.to(device), slot_ids.to(device)
                    padding = padding.to(device)
                    embeddings = model.embed_subject(raw, mod_ids, visit_ids, slot_ids, padding)
                    embeddings = embeddings.cpu().numpy().astype(np.float32)
                    token_counts = (~padding).sum(dim=1).cpu().numpy()
                    subjects = dataset.subjects[offset:offset + len(embeddings)]
                    for subid, value, n_tokens in zip(subjects, embeddings, token_counts):
                        filename = f"{subid}.npy"
                        np.save(args.output_dir / filename, value)
                        writer.writerow([subid, split, int(n_tokens), filename])
                    offset += len(embeddings)
    metadata = {
        "representation": "shared_foundation_embedding",
        "input_visits": PROSPECTIVE_VISITS,
        "output_dim": int(model.d_model),
        "subject_pooling": "learned_fusion_token",
        "checkpoint": str(args.checkpoint.resolve()),
        "embedding_dir": str(args.embedding_dir.resolve()),
        "modalities": tuple(checkpoint["modalities"]),
        "split_file": str(HERE.parent / "data_splits" / "master_subject_split.csv"),
        "n_subjects": sum(len(ds) for _, ds in datasets),
    }
    with (args.output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    (args.output_dir / ".complete").write_text("complete\n")
    print(f"Saved {sum(len(ds) for _, ds in datasets):,} subject embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
