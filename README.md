# CEPI

Implementation of camera-calibrated cross-exposure posterior-predictive RAW restoration.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The following input route generates paired RAW mosaics from a declared camera-noise process. These are generated inputs. `source.json` records assumptions and seed; the selected perceptual output is `rgb_l1`, with no LPIPS weights required. Real camera exports can instead use the manifest interface described below.

```bash
python -m cepi.synthetic --output data/generated --groups 30 --size 64 --seed 17
cepi prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Download [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) and [ELD](https://github.com/Vandermode/ELD). Export registered, black-level-corrected linear mosaics to NPZ files with `source,target,clean` [height,width], boolean `mask_source,mask_target`, scalar source/target `exposure` and `gain`, scalar integer `camera`, and source/target `noise` [shot_a,read_b,row_variance,quantization_variance]. Also include integer `cfa` [2,2] (RGB channel indices), `white_balance` [3] and `color_matrix` [3,3]. The JSON manifest needs `id,path,scene,split`; use `train`, `validation` and `test`, keeping every scene in one split. Fields use prefixes such as `source_exposure`. Exposures/gains are normalized to the clean radiance reference so `h(r)=exposure*gain*r`. Noise coefficients must use those normalized observation units. Frozen calibration comes from independent camera measurements, not test images.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
cepi train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
```

## Evaluation

```bash
cepi evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
cepi compare --inputs runs/reproduce --output outputs/comparison
cepi plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `model.pt`, `history.csv`, `scenes.csv`, `calibration_bins.csv`, `metrics.json`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training and full evaluation were not run for this review. The GitHub Actions workflow checks packaging and source code.
