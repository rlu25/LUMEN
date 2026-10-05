# Structural MRI checkpoint

Download the project checkpoint from [Google Drive](https://drive.google.com/drive/folders/1Z-HgIAm5jvh6RdsZmjV4UkYkKAZxg9q1?usp=sharing)
and place it at this path relative to this directory:

```text
checkpoints/best_model.pth
```

Use the upstream [MAPSeg repository](https://github.com/XuzheZ/MAPSeg/tree/main)
for the sMRI source implementation and its environment/setup instructions. The
source tree is not duplicated in LUMEN. Checkpoint binaries are excluded from GitHub.

The LUMEN fusion workflow does not load this checkpoint directly. It expects
already exported 512-dimensional vectors named
`NDAR_INV<id>_Y<visit>.npy` in the directory configured by
`LUMEN_SMRI_EMBEDDING_DIR`.

Checkpoint SHA-256:

```text
df5de21e00bd34cab07d84f0a1422907d60c7861199ea2b77641b9b96e9490e7  checkpoints/best_model.pth
```
