# Tree-structured CAD reconstruction and controlled generation

This repository contains the training and evaluation code for the paper
"Tree-structured command sequences for parametric CAD reconstruction and
controlled generation".

The workflow starts from
MCL-lite JSON trees and covers data augmentation, vectorization, autoencoder
training, latent encoding, diffusion training, operation-presence guidance,
sampling, decoding and numerical evaluation.

The package does not contain datasets, checkpoints, logs or generated models.
The converter from the original commercial CAD format to MCL-lite JSON is not
part of this repository.

The repository metadata is stored at the root. The runnable code package is in
`pcmracgbotcs/`.

## Environment

We recommend creating a new Conda environment for this project.

```bash
cd pcmracgbotcs
conda create -n mcl python=3.7.12 -y
conda activate mcl
pip install -r requirements.txt
```

The published environment uses PyTorch 1.10.2 with CUDA 11.3. GPU training
requires a PyTorch build compatible with the local CUDA driver.

The expected data layout inside `pcmracgbotcs/` is:

```text
data/
  cad_vec/
    0000/00000007_lite.h5
    ...
  split.json
```

## Paper-to-code map

Paths below are relative to `pcmracgbotcs/`.

| Paper component | Code |
|---|---|
| Tree sequence and `ENDCHILD` / `ENDROOT` encoding | `vectorize.py`, `devectorize.py`, `cadlib/macro.py` |
| Prism-to-Revol and Prism-to-Thrusections augmentation | `augment/augment_revol.py`, `augment/augment_thrusections.py` |
| Attention-residual autoencoder | `model/autoencoder.py`, `model/layers/` |
| Autoencoder training and reconstruction output | `train_autoencoder.py`, `reconstruct.py` |
| Reconstruction accuracy | `evaluate/evaluate_reconstruction.py` |
| Latent diffusion model | `model/diffusion.py`, `model/noise_predictor.py` |
| Noisy-latent operation-presence classifier | `model/operation_classifier.py`, `train_operation_classifier.py` |
| Classifier-guided sampling and decoding | `sample_guided.py`, `decode_latents.py` |

## Dataset

*Tree-structured command sequences in JSON format.*
<https://doi.org/10.5281/zenodo.23833977>

*Sequence format conversion script and visualization software: to be released.*

## Full workflow

Run the following commands from `pcmracgbotcs/`.

Only required paths and experiment-specific values are passed on the command
line. Data paths use `data`, and training hyperparameters use the paper defaults.

The augmentation scripts use the paper seed by default. Replace `START_ID` and
`END_ID` with the top-level folder IDs in your dataset.

```bash
python augment/augment_revol.py \
  --start-folder START_ID --end-folder END_ID

python augment/augment_thrusections.py \
  --start-folder START_ID --end-folder END_ID

python vectorize.py \
  --input raw_json raw_json_Revol raw_json_thrusections

python gen_split.py
```

Train the autoencoder:

```bash
python train_autoencoder.py
```

Generate reconstruction HDF5 files on the test split:

```bash
python reconstruct.py \
  --ckpt runs/mcl/model/latest.pth
```

Reconstruction files are written to `recon/test` by default.

Evaluate reconstruction accuracy:

```bash
python evaluate/evaluate_reconstruction.py
```

The evaluator writes only `ACC_cmd` and `ACC_param` to `metrics.json` and
`metrics.csv`. `ACC_cmd` covers positions through the first EOS and includes
that EOS. `ACC_param` uses only command-specific non-padding parameter slots.

Encode every split into the trained latent space:

```bash
python encode_latents.py \
  --ae_ckpt_path runs/mcl/model/latest.pth
```

Train the unconditional latent diffusion model:

```bash
python train_latent_diffusion.py --ae_ckpt latest
```

Train the classifier for the three operation-presence bits:

```bash
python train_operation_classifier.py --ae_ckpt latest
```

The command writes the step-matched checkpoint used in the paper evaluation to
`runs/mcl/latent_diffusion_latest/classifier_model/op_classifier_step10000.pth`.

Evaluate the classifier and write numerical reports:

```bash
python evaluate/evaluate_presence_classifier.py \
  --z_path runs/mcl/results/all_zs_ckptlatest.h5 \
  --classifier_ckpt runs/mcl/latent_diffusion_latest/classifier_model/op_classifier_step10000.pth
```

The evaluator writes `metrics.json`, `metrics.csv` and
`confusion_counts.csv`.

Generate Prism latents with classifier guidance:

```bash
python sample_guided.py \
  --ae_ckpt latest --op_type prism \
  --classifier_ckpt runs/mcl/latent_diffusion_latest/classifier_model/op_classifier_step10000.pth \
  --n_samples 500
```

Use `revol` or `thrusections` for the other targets. Decode and devectorize:

```bash
python decode_latents.py \
  --latent_path runs/mcl/latent_diffusion_latest/results/prism_scale1.0_n500.h5 \
  --ae_ckpt_path runs/mcl/model/latest.pth

python devectorize.py --input runs/mcl/latent_diffusion_latest/results/prism_scale1.0_n500_decoded
```

