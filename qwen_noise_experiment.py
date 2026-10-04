"""Place beside vlm.py; run directly, without importing or running main.py.

Gaussian pixel noise: round(255 * clip(x + sigma*z, 0, 1)), x in [0, 1].
Reuses vlm.extract_embeddings: final layer, final input token, no pooling.
Qwen3-VL-4B has 2560 features; the width is checked, never forced to 4096.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SIGMAS = (0.0, 0.1, 0.2, 0.5)
METRICS = ("l2", "relative_l2", "cosine_distance")
PROMPT = "Describe the visual style, colors, typography and objects in this movie poster."


def perturb(pixels, noise, sigma):
    """Perturb original RGB pixels before Qwen resizing/normalization."""
    return Image.fromarray(np.rint(255 * np.clip(pixels + sigma * noise, 0, 1)).astype(np.uint8))


def distances(features, clean):
    features, clean = np.asarray(features, dtype=np.float64), np.asarray(clean, dtype=np.float64)
    if not (np.isfinite(features).all() and np.isfinite(clean).all()):
        raise ValueError("Non-finite embeddings.")
    norms, clean_norm = np.linalg.norm(features, axis=1), np.linalg.norm(clean)
    if clean_norm == 0 or np.any(norms == 0):
        raise ValueError("Zero-norm embedding; cosine distance is undefined.")
    l2 = np.linalg.norm(features - clean, axis=1)
    cosine = 1 - np.clip((features @ clean) / (norms * clean_norm), -1, 1)
    return dict(zip(METRICS, (l2, l2 / clean_norm, cosine)))


def experiment(df, embed, repeats, seed):
    """One clean reference/poster; independent noise draws across repeats.

    Within a repeat use the same z for all sigmas, isolating noise amplitude.
    Re-evaluate sigma=0 each repeat to measure the inference noise floor.
    """
    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        path = str(row["image_path"])
        with Image.open(path) as image:
            clean_image = image.convert("RGB")
        pixels = np.asarray(clean_image, dtype=np.float32) / 255
        clean = embed([clean_image])[0]
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, repeat]))
            noise = rng.standard_normal(pixels.shape).astype(np.float32)
            images = [perturb(pixels, noise, sigma) for sigma in SIGMAS]
            values = distances(embed(images), clean)
            for j, sigma in enumerate(SIGMAS):
                raw = pixels + sigma * noise
                rows.append({
                    "poster": index, "image_path": path, "repeat": repeat,
                    "sigma": sigma, "feature_dim": len(clean),
                    "pixel_rmse": float(np.sqrt(np.mean((np.asarray(images[j]) / 255 - pixels) ** 2))),
                    "clipped_fraction": float(np.mean((raw < 0) | (raw > 1))),
                    **{key: float(value[j]) for key, value in values.items()},
                })
        print(f"Poster {index + 1}/{len(df)} complete", flush=True)
    return pd.DataFrame(rows)


def summarize(results):
    # Posters are the sampling units: do not treat repeated draws as new posters.
    grouped = results.groupby(["poster", "sigma"])[list(METRICS)]
    poster_means = grouped.mean()
    summary = poster_means.groupby("sigma").agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}_across_posters" for metric, stat in summary.columns]
    within = grouped.std().groupby("sigma").mean().add_suffix("_mean_within_poster_sd")
    summary = summary.join(within)
    repeat_means = results.groupby(["repeat", "sigma"])[list(METRICS)].mean()
    trends = []
    for metric in METRICS:
        curves = results.pivot(index=["poster", "repeat"], columns="sigma", values=metric)
        # Exclude the trivial clean->noisy step when assessing monotonicity.
        curves = curves.loc[:, list(SIGMAS[1:])]
        deltas = np.diff(curves.to_numpy(), axis=1)
        for j, (low, high) in enumerate(zip(SIGMAS[1:-1], SIGMAS[2:])):
            paired = (poster_means[metric].unstack()[high] - poster_means[metric].unstack()[low])
            trends.append({
                "metric": metric, "sigma_low": low, "sigma_high": high,
                "mean_paired_change": paired.mean(),
                "sd_paired_change_across_posters": paired.std(),
                "fraction_trials_increasing": float(np.mean(deltas[:, j] > 1e-8)),
                "fraction_trials_nondecreasing": float(np.mean(deltas[:, j] >= -1e-8)),
                "fraction_trials_monotonic_over_nonzero_sigmas": float(np.mean(np.all(deltas >= -1e-8, axis=1))),
            })
    return summary, repeat_means, pd.DataFrame(trends)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="movies.csv")
    parser.add_argument("--poster-dir", default="posters")
    parser.add_argument("--local-posters", action="store_true", help="Use existing images only; no CSV or downloads.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all posters; use 10 for a smoke run.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true", help="Optional CUDA/bitsandbytes quantization.")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("qwen_noise_results"))
    args = parser.parse_args()
    if args.repeats < 2 or args.batch_size < 1 or args.limit < 0 or args.seed < 0:
        parser.error("Require repeats >= 2, batch-size >= 1, limit >= 0 and seed >= 0.")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from config import set_seed
    from vlm import extract_embeddings

    set_seed(args.seed)
    if args.local_posters:
        paths = sorted(p.resolve() for p in Path(args.poster_dir).rglob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        df = pd.DataFrame({"image_path": [str(p) for p in paths]})
    else:
        from dataset import prepare_data
        datasets, _ = prepare_data(args.csv, args.poster_dir, args.seed)
        df = pd.concat(datasets, ignore_index=True)
    if df.empty:
        parser.error("No posters found.")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    df = df.reset_index(drop=True)
    if args.load_in_4bit and not torch.cuda.is_available():
        parser.error("This script's 4-bit mode requires CUDA; omit --load-in-4bit.")
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16 if torch.cuda.is_available() else torch.float32)
    kwargs = {"device_map": "auto", "torch_dtype": dtype}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **kwargs).eval()
    model.config.use_cache = False
    # Keep the repository's image-processing budget for comparable extraction.
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=256*28*28, max_pixels=256*28*28)
    processor.tokenizer.padding_side = "left"  # [:, -1, :] must select a real token.
    text_config = getattr(model.config, "text_config", model.config)
    width = text_config.hidden_size
    print(f"Final-token feature dimension: {width} (no pooling)", flush=True)

    def embed(images):
        # qwen_vl_utils accepts PIL objects in the existing image_path column.
        # Labels are unused here, but required by extract_embeddings's interface.
        batch = pd.DataFrame({"image_path": images, "labels": [[0] for _ in images]})
        features, _ = extract_embeddings(model, processor, batch, args.prompt, batch_size=args.batch_size)
        if features.shape != (len(images), width):
            raise ValueError(f"Unexpected embedding shape: {features.shape}")
        return features

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sigmas": SIGMAS, "feature_dim": width, "n_posters": len(df), "dtype": str(dtype),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "versions": {name: version(name) for name in ("torch", "transformers", "qwen-vl-utils", "numpy", "pandas", "Pillow")},
        "noise": "iid Gaussian RGB in [0,1] before resizing; shared draw across sigmas; clipping and uint8 rounding",
        "representation": "hidden_states[-1][:, -1, :] with fixed prompt and left padding",
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    df[["image_path"]].to_csv(args.output_dir / "posters.csv", index=False)
    results = experiment(df, embed, args.repeats, args.seed)
    results.to_csv(args.output_dir / "distances.csv", index=False)
    summary, repeat_means, trends = summarize(results)
    summary.to_csv(args.output_dir / "summary.csv")
    repeat_means.to_csv(args.output_dir / "repeat_means.csv")
    trends.to_csv(args.output_dir / "trends.csv", index=False)
    print(summary.to_string())
    print(trends.to_string(index=False))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
