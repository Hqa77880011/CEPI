"""Render independent scenes through a declared calibrated RAW sensor law."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def sensor(radiance, exposure, gain, noise, rng):
    """Independent shot/read pixels, shared row noise and uniform quantization."""
    signal = exposure * gain * radiance
    shot, read, row, quant = noise
    value = signal + rng.normal(size=signal.shape) * np.sqrt(shot * signal + read)
    value += rng.normal(size=(len(signal), 1)) * np.sqrt(row)
    value += rng.uniform(-np.sqrt(3 * quant), np.sqrt(3 * quant), signal.shape)
    valid = (value > 0) & (value < 1)
    return np.clip(value, 0, 1).astype(np.float32), valid


def scene(size, camera, rng):
    yy, xx = np.mgrid[:size, :size] / size
    rgb = []
    for _ in range(3):
        smooth = gaussian_filter(rng.normal(size=(size, size)), size / 12)
        smooth /= max(float(smooth.std()), 1e-6)
        pattern = 0.12 * np.sin(2 * np.pi * (rng.uniform(1, 4) * xx + rng.uniform(1, 4) * yy))
        rgb.append(np.clip(0.35 + 0.08 * smooth + pattern, 0.04, 0.8))
    rgb = np.stack(rgb, -1)
    cfa = np.array([[0, 1], [1, 2]], dtype=np.int64)
    tile = np.tile(cfa, (int(np.ceil(size / 2)),) * 2)[:size, :size]
    clean = np.take_along_axis(rgb, tile[..., None], -1)[..., 0].astype(np.float32)
    noise = np.array([0.0015 * (camera + 1), 2e-5, 8e-6, (1 / 1024) ** 2 / 12], np.float32)
    source_exposure, target_exposure = rng.uniform(0.2, 0.5), rng.uniform(0.8, 1.1)
    source_gain, target_gain = 1.0, rng.uniform(0.9, 1.1)
    source, mask_source = sensor(clean, source_exposure, source_gain, noise, rng)
    target, mask_target = sensor(clean, target_exposure, target_gain, noise, rng)
    return dict(source=source, target=target, clean=clean, mask_source=mask_source,
                mask_target=mask_target, source_exposure=source_exposure,
                target_exposure=target_exposure, source_gain=source_gain,
                target_gain=target_gain, source_noise=noise, target_noise=noise,
                camera=camera, cfa=cfa, white_balance=np.ones(3, np.float32),
                color_matrix=np.eye(3, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 6 or args.size < 16:
        parser.error("Use at least six scenes and size sixteen")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, group]))
        split = "train" if group < 0.6 * args.groups else "validation" if group < 0.8 * args.groups else "test"
        values = scene(args.size, group % 2, rng)
        name = f"scene_{group:04d}.npz"
        np.savez_compressed(output / name, **values)
        records.append(dict(id=f"scene-{group}", scene=f"scene-{group}", path=name, split=split))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="rendered RAW with independently sampled calibrated sensor noise",
        seed=args.seed, groups=args.groups, size=args.size,
        truth="clean radiance before exposure, gain, noise and clipping",
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
