# CEPI

PyTorch implementation of camera-calibrated cross-exposure posterior-predictive RAW restoration.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Registered RAW pairs

Download [SID](https://github.com/cchen156/Learning-to-See-in-the-Dark) and [ELD](https://github.com/Vandermode/ELD). Export registered, black-level-corrected linear mosaics to NPZ files with `source,target,clean` [height,width], boolean `mask_source,mask_target`, scalar source/target `exposure` and `gain`, scalar integer `camera`, and source/target `noise` [shot_a,read_b,row_variance,quantization_variance]. Fields use prefixes such as `source_exposure`. Exposures/gains are normalized to the clean radiance reference so `h(r)=exposure*gain*r`. Noise coefficients must use those normalized observation units. Frozen calibration comes from independent camera measurements, not test images.

Each file also has `cfa` (Bayer 2x2 or X-Trans 6x6 color IDs R=0/G=1/B=2), RGB `white_balance` and a linear camera-to-sRGB `color_matrix` [3,3]. Supply a JSON manifest of `{id,path,scene,split}` records, split by physical scene before forming brackets/crops. Masks exclude hot, clipped and registration-invalid pixels; saturation/clipping are additionally excluded by code.

```bash
cepi prepare --input data/manifest.json --data data/raw
cepi train --data data/raw --variant cepi --output runs/full
cepi evaluate --data data/raw --checkpoint runs/full/model.pt --output runs/full
cepi compare --inputs runs/full --output outputs/comparison
cepi plot --output outputs/comparison
```

## Objective and controls

A shared compact encoder/decoder predicts positive radiance mean and variance from one RAW frame, exposure/gain and camera embedding. Bidirectional predictive Gaussian NLL propagates posterior variance through the target camera map and adds calibrated shot/read/row/quantization variance. Masked clean-radiance L1 anchors both directional means. The Gaussian NLL omits the additive constant `0.5*log(2*pi)` consistently.

Train separate `supervised`, `noise2noise`, `same_exposure` and `deterministic` runs for objective controls. Deterministic removes only propagated latent variance; same-exposure uses the observed source itself as the predictive target. Evaluate a selected checkpoint with `--control wrong_exposure`, `wrong_gain` or `homoscedastic` to change only the predictive observation law. These post-selection law controls leave the restoration mean unchanged, so their PSNR is identical while predictive calibration changes. To test metadata-conditioned restoration fidelity, modify source metadata in a separate test manifest. External IDR/NM1H training systems are not aliases of these objective controls.

## Outputs and conventions

`configs/default.json` controls widths, camera vocabulary, epochs, AdamW learning rate, paired crop size, predictive weight, variance floor and device. Training writes final `model.pt` and loss history; it never selects from test PSNR. Evaluation stores restored radiance/variance/sRGB arrays, per-scene PSNR and the configured LPIPS/RGB-L1 distance, predictive NLL, absolute innovation mean, variance deviation from one, 90% coverage and calibration error across 10%–90% central intervals. Calibration bins retain gain and signal-intensity membership. Summaries average brackets within scenes and then weight scenes equally.

RAW PSNR uses unit radiance range on valid paired pixels. The shared deterministic ISP performs normalized local CFA interpolation, white balance, color matrix and sRGB transfer; invalid predicted pixels are filled with the common reference before LPIPS, giving no metric weight to restoration at excluded locations. This compact ISP and backbone are explicit implementation choices, and their perceptual scores should not be mixed with a different ISP. The optional `perceptual_metric="lpips"` uses published AlexNet weights. The registered `rgb_l1` setting uses no weights and writes `rgb_l1`, not LPIPS. Comparison CSV/JSON and PNG/SVG plots use generated measurements.

## Registered compact reproduction setting

Rendered radiance scenes pass through calibrated shot/read/row/quantization noise with independent source/target draws. Clean radiance precedes the sensor process; scene groups never cross splits. The compact setting reports rgb_l1 after the common ISP, with its exact metric name and no pretrained perceptual weights.

Generate the declared input process, preprocess it, then use the existing method entry points:

```bash
python -m cepi.synthetic --output data/generated --groups 30 --size 64 --seed 17
cepi prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
cepi train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
cepi evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

`source.json` records the data-generating assumptions and seed. The configuration records `input_source`. All inputs for this selected setting are generated locally using installed package dependencies.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
cepi-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
