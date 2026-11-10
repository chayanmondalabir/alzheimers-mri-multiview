# Weights

The trained networks are **not stored in git** (about 650 MB). They are attached to the
[GitHub Release `v1.0`](../../../releases) as `weights.zip` and contain:

| File | What it is |
|---|---|
| `{axial,coronal,sagittal}_seed{0-4}.pt` | 15 ResNet-18 networks (5 seeds per plane), trained on all 198 OASIS-1 subjects aged 60+ |
| `config.json` | slicing settings used at training time |
| `template.npy` | mean of 60 OASIS-1 volumes, used to centre new scans |
| `extra.json` | typical brain size and the weak-alignment cut-off |

Download and unpack them with `python scripts/fetch_weights.py`.
