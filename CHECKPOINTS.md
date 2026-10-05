# Released checkpoints

Download the checkpoints and normalizers from [Google Drive](https://drive.google.com/drive/folders/1Z-HgIAm5jvh6RdsZmjV4UkYkKAZxg9q1?usp=sharing).
Binary artifacts are not included in this GitHub repository. After downloading,
place them at the paths below, relative to the repository root. `best.pt` denotes validation selection, not selection on the
held-out test set.

## Modality pretraining

| Modality | Released artifact | Best epoch | Validation objective |
|---|---|---:|---:|
| rsFC | `modality_pretraining/rsfc/best.pt` | 47 | 0.9859607107 |
| Fitbit | `modality_pretraining/fitbit/best.pt` | 296 | 0.0736675672 |
| BOLD | `modality_pretraining/bold/best.pt` | 498 | 0.1901222456 |
| Genetics | `modality_pretraining/genetics/best.pt` | 371 | 0.3864824395 |
| sMRI | `modality_pretraining/smri/checkpoints/best_model.pth` | not recorded here | not recorded here |

Also download the encoder-only rsFC, Fitbit, and BOLD weights used by the
fusion input exporter, and the training-derived rsFC and Fitbit normalizers.
Their destination paths are listed in the SHA-256 section below.

The sMRI checkpoint is distributed in the Google Drive folder. The associated source implementation is maintained upstream in
[XuzheZ/MAPSeg](https://github.com/XuzheZ/MAPSeg/tree/main) and is not copied
into this repository. LUMEN's fusion code consumes exported 512-dimensional
sMRI vectors; it does not load `best_model.pth` directly.

## Multimodal fusion

`multimodal_fusion/checkpoints/best.pt` is the validation-selected shared
fusion checkpoint (epoch 109; validation objective 0.2645976084).
`multimodal_fusion/checkpoints/model.pt` is the corresponding compact model
state used by the original workflow. The loss log is included in this repository. Restore the `.complete` marker
from the download alongside the fusion checkpoints when required by the workflow.

## Downstream prediction

Validation-selected frozen heads for `bmi`, `g_factor`, `internalizing`,
`externalizing`, and `sui` should be downloaded to
`downstream_prediction/checkpoints/<task>/best.pt`. Their non-participant-level
loss and metric summaries are included. Stage-2-fine-tuned checkpoints are a
separate experimental contract and are not mixed into this frozen-head set.

## SHA-256

```text
ef6d73977e8925f4170ef3ee84ecaf9c13e84e16016ef65ee301ee524e72b93f  modality_pretraining/rsfc/best.pt
f5caadf59cd4bac8ad60416a8200df428304aa4fa123896e41c4dfe1d3abf247  modality_pretraining/rsfc/encoder.pt
a9419e63820b74d484ada2b2a0e974a473b14908b34d66e697563ac436274bad  modality_pretraining/rsfc/normalizer.npz
4ae1d0d77496872e913456c82aab5af46c27decab0642fe44ae7cf82924ff9f1  modality_pretraining/fitbit/best.pt
ee48cf57e57fb728a98eafd622cfaddec96c17c59eccf92d259bc95861181b3e  modality_pretraining/fitbit/encoder.pt
ec102a114d6bff908cee8b1f54465fa6faae2461bee9f4cc23f0927b58eb1c8f  modality_pretraining/fitbit/normalizer.npz
629f5992670f7bb75005bfe9caa38015474b70c3c7dd93736aa381d4a7a88f6a  modality_pretraining/bold/best.pt
454c94d0369ace7f44aa3d74b4a93e2e3389b4d7b1ebbf0703e777c2166e6671  modality_pretraining/bold/encoder.pt
5deb206335110a78129411560e6e1b88389bfcf41f3deac32462d8c7813efe10  modality_pretraining/genetics/best.pt
df5de21e00bd34cab07d84f0a1422907d60c7861199ea2b77641b9b96e9490e7  modality_pretraining/smri/checkpoints/best_model.pth
9c2b04faad40c073c8e16f27aaab3d02429e5ee91ebe3017e66d133e3e38bd65  multimodal_fusion/checkpoints/best.pt
c84781b37d42a42df8b8483e564332f4fdd3701382417bb90bb586bf06940673  multimodal_fusion/checkpoints/model.pt
02b2d923725538db541d6df679aa8cc7e11b29ef480e6a0495b2fa9b2f53006f  downstream_prediction/checkpoints/bmi/best.pt
b0f23ed21363d873e842cdf970d977391b784a6b35c5983814864d81dcb684c8  downstream_prediction/checkpoints/g_factor/best.pt
58c340868a9a6ec3e4263c36eca6a595483411e99bbab4d17081940dfc071612  downstream_prediction/checkpoints/internalizing/best.pt
2b59ccb503139342fdedba0efd93880975602f8b6e1623fe098b9bbfd849418e  downstream_prediction/checkpoints/externalizing/best.pt
d373410c200313a130c6eb72b27a0f0a686f720481c15d24d4e67024817acd9c  downstream_prediction/checkpoints/sui/best.pt
```
