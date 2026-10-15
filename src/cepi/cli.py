import argparse
from pathlib import Path
from statistics import NormalDist
import numpy as np
import torch
from scipy.ndimage import convolve
from .model import Restoration, predictive, nll
from .common import load, save, table, plot


def prepare(a, c):
    records = load(a.input)
    groups = {}
    result = []
    for record in records:
        if record["scene"] in groups and groups[record["scene"]] != record["split"]:
            raise ValueError("Scene crosses splits")
        groups[record["scene"]] = record["split"]
        path = Path(record["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        data = np.load(path)
        for key in [
            "source",
            "target",
            "clean",
            "mask_source",
            "mask_target",
            "source_exposure",
            "target_exposure",
            "source_gain",
            "target_gain",
            "source_noise",
            "target_noise",
            "camera",
            "cfa",
            "white_balance",
            "color_matrix",
        ]:
            if key not in data:
                raise ValueError("Missing RAW field: " + key)
        result.append(dict(record, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", result)


def tensors(path, c, rng=None):
    raw = np.load(path)
    arrays = {k: raw[k] for k in raw.files}
    if rng is not None:
        h, w = arrays["source"].shape
        size = min(c["crop"], h, w)
        top = int(rng.integers(h - size + 1))
        left = int(rng.integers(w - size + 1))
        for key in ["source", "target", "clean", "mask_source", "mask_target"]:
            arrays[key] = arrays[key][top : top + size, left : left + size]
    result = {}
    for k, v in arrays.items():
        if k in ["cfa", "white_balance", "color_matrix"]:
            continue
        tensor = torch.as_tensor(np.asarray(v).copy(), device=c["device"])
        if k in ["mask_source", "mask_target"]:
            result[k] = tensor.bool()[None, None]
        elif k in ["source", "target", "clean"]:
            result[k] = tensor.float()[None, None]
        elif k == "camera":
            result[k] = tensor.long().reshape(1)
        else:
            result[k] = tensor.float().unsqueeze(0)
    result["mask_source"] &= (result["source"] > 0) & (result["source"] < 1)
    result["mask_target"] &= (result["target"] > 0) & (result["target"] < 1)
    return result, arrays


def forward(model, b, prefix):
    return model(b[prefix], b[prefix + "_exposure"], b[prefix + "_gain"], b["camera"])


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "train"]
    if not records:
        raise ValueError("No training scenes")
    model = Restoration(c["width"], c["cameras"]).to(c["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    history = []
    for epoch in range(c["epochs"]):
        total = 0.0
        rng.shuffle(records)
        model.train()
        for record in records:
            b, _ = tensors(record["path"], c, rng)
            ms, vs = forward(model, b, "source")
            mt, vt = forward(model, b, "target")
            rec = (
                (ms - b["clean"]).abs()[b["mask_source"]].mean()
                + (mt - b["clean"]).abs()[b["mask_target"]].mean()
            ) / 2
            ps, pv = predictive(
                ms,
                vs,
                b["target_exposure"],
                b["target_gain"],
                b["target_noise"],
                c["variance_floor"],
                a.variant,
            )
            pt, tv = predictive(
                mt,
                vt,
                b["source_exposure"],
                b["source_gain"],
                b["source_noise"],
                c["variance_floor"],
                a.variant,
            )
            pred = nll(b["target"], ps, pv, b["mask_target"]) + nll(
                b["source"], pt, tv, b["mask_source"]
            )
            if a.variant == "supervised":
                loss = rec
            elif a.variant == "noise2noise":
                loss = (
                    (
                        ms
                        - b["target"]
                        / (b["target_exposure"] * b["target_gain"])[:, None, None, None]
                    )
                    .square()[b["mask_target"]]
                    .mean()
                )
            elif a.variant == "same_exposure":
                same, var = predictive(
                    ms,
                    vs,
                    b["source_exposure"],
                    b["source_gain"],
                    b["source_noise"],
                    c["variance_floor"],
                    "cepi",
                )
                loss = rec + c["predictive_weight"] * nll(
                    b["source"], same, var, b["mask_source"]
                )
            else:
                loss = rec + c["predictive_weight"] * pred
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        history.append(dict(epoch=epoch + 1, loss=total / len(records)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(model=model.cpu().state_dict(), config=c, variant=a.variant),
        output / "model.pt",
    )
    table(output / "history.csv", history)


def isp(mosaic, cfa, wb, matrix):
    h, w = mosaic.shape
    tile = np.tile(
        cfa, (int(np.ceil(h / cfa.shape[0])), int(np.ceil(w / cfa.shape[1])))
    )[:h, :w]
    kernel = np.ones((2 * max(cfa.shape) + 1,) * 2)
    rgb = []
    for color in range(3):
        mask = (tile == color).astype(float)
        rgb.append(
            convolve(mosaic * mask, kernel, mode="mirror")
            / np.maximum(convolve(mask, kernel, mode="mirror"), 1e-8)
        )
    linear = np.stack(rgb, -1) * np.asarray(wb)[None, None]
    linear = np.maximum(linear @ np.asarray(matrix).T, 0)
    return np.clip(
        np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.maximum(linear, 1e-12) ** (1 / 2.4) - 0.055,
        ),
        0,
        1,
    ).astype("float32")


def evaluate(a, c):
    import lpips

    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    model = Restoration(c["width"], c["cameras"]).to(c["device"])
    model.load_state_dict(state["model"])
    model.eval()
    perceptual = lpips.LPIPS(net="alex").to(c["device"]).eval()
    rows = []
    bins = []
    variant = state["variant"] if a.control == "none" else a.control
    for record in load(Path(a.data) / "manifest.json"):
        if record["split"] != "test":
            continue
        b, raw = tensors(record["path"], c)
        with torch.no_grad():
            mean, var = forward(model, b, "source")
            pm, pv = predictive(
                mean,
                var,
                b["target_exposure"],
                b["target_gain"],
                b["target_noise"],
                c["variance_floor"],
                variant,
            )
            valid = b["mask_target"] & b["mask_source"]
            innovation = ((b["target"] - pm) / pv.sqrt())[valid].cpu().numpy()
            predictive_loss = float(nll(b["target"], pm, pv, valid))
            estimate = mean[0, 0].cpu().numpy()
            mask = valid[0, 0].cpu().numpy()
            clean = raw["clean"]
            mse = np.mean((estimate[mask] - clean[mask]) ** 2)
            estimated = isp(
                np.where(mask, estimate, clean),
                raw["cfa"],
                raw["white_balance"],
                raw["color_matrix"],
            )
            reference = isp(
                clean, raw["cfa"], raw["white_balance"], raw["color_matrix"]
            )
            rgb = lambda x: (
                torch.from_numpy(x).permute(2, 0, 1)[None].to(c["device"]) * 2 - 1
            )
            perceptual_distance = float(
                perceptual(rgb(estimated), rgb(reference)).mean()
            )
        levels = np.arange(0.1, 1, 0.1)
        coverage = np.array(
            [
                np.mean(np.abs(innovation) <= NormalDist().inv_cdf((1 + p) / 2))
                for p in levels
            ]
        )
        rows.append(
            dict(
                id=record["id"],
                scene=record["scene"],
                camera=int(raw["camera"]),
                method=variant,
                psnr=float(-10 * np.log10(max(mse, 1e-12))),
                lpips=perceptual_distance,
                predictive_nll=predictive_loss,
                innovation_mean_error=abs(float(np.mean(innovation))),
                innovation_variance_error=abs(float(np.var(innovation)) - 1),
                coverage90=float(np.mean(np.abs(innovation) <= 1.6448536269514722)),
                calibration_ece=float(np.mean(np.abs(coverage - levels))),
                exposure_ratio=float(raw["target_exposure"] / raw["source_exposure"]),
            )
        )
        signal = pm[valid].detach().cpu().numpy()
        for low, high in zip([0, 0.05, 0.2, 0.5], [0.05, 0.2, 0.5, np.inf]):
            selected = (signal >= low) & (signal < high)
            if selected.any():
                bins.append(
                    dict(
                        id=record["id"],
                        gain=float(raw["target_gain"]),
                        signal_low=low,
                        count=int(selected.sum()),
                        mean=float(innovation[selected].mean()),
                        variance=float(innovation[selected].var()),
                        coverage90=float(
                            np.mean(np.abs(innovation[selected]) <= 1.6448536269514722)
                        ),
                    )
                )
        output = Path(a.output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / (str(record["id"]) + ".npz"),
            radiance=estimate,
            variance=var[0, 0].detach().cpu().numpy(),
            srgb=estimated,
        )
    if not rows:
        raise ValueError("No test scenes")
    keys = [k for k in rows[0] if k not in ("id", "scene", "camera", "method")]
    scene_means = [
        {key: np.mean([r[key] for r in rows if r["scene"] == scene]) for key in keys}
        for scene in sorted({r["scene"] for r in rows})
    ]
    summary = dict(
        method=variant, **{key: np.mean([r[key] for r in scene_means]) for key in keys}
    )
    table(Path(a.output) / "scenes.csv", rows)
    table(Path(a.output) / "calibration_bins.csv", bins)
    save(Path(a.output) / "metrics.json", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/raw")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument("--inputs", nargs="*")
    p.add_argument(
        "--variant",
        choices=["cepi", "supervised", "noise2noise", "same_exposure", "deterministic"],
        default="cepi",
    )
    p.add_argument(
        "--control",
        choices=["none", "wrong_exposure", "wrong_gain", "homoscedastic"],
        default="none",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [load(Path(d) / "metrics.json") for d in a.inputs]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
