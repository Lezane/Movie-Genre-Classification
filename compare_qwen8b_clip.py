"""Add-on experiment; all original repository files remain unchanged."""
import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from joblib import parallel_config
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, CLIPModel

from config import set_seed
from dataset import prepare_data
from vlm import extract_embeddings
from main import save_results


class QwenFeatureForward(torch.nn.Module):
    """Use the original extractor, avoiding unnecessary logits and KV caching."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    @property
    def device(self):
        return self.backbone.device

    def forward(self, **inputs):
        return self.backbone(**inputs, use_cache=False, logits_to_keep=1)


def fit_original_heads(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl,
                       seed=50, mlp_max_iter=500):
    """Original vlm.py LR/MLP section, with a configurable iteration cap.

    LR uses training-only scaling. MLP uses the original raw features. Candidate
    grids, thresholds and validation micro-F1 selection are unchanged. Starting
    best_acc at -1 also allows a valid head when every validation score is zero.
    """
    results = {}
    # 4. Logistic Regression Head
    print("\n[VLM] Tuning Logistic Regression...", flush=True)

    # Fit scaling on training data only.
    # Keep the original embeddings unchanged for the MLP section.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_emb)
    X_val = scaler.transform(val_emb)
    X_test = scaler.transform(test_emb)

    print(
        f"Training samples: {X_train.shape[0]}, "
        f"features: {X_train.shape[1]}, "
        f"genres: {train_lbl.shape[1]}",
        flush=True,
    )

    best_c, best_acc, best_lr = None, -1.0, None

    # Train two genre classifiers concurrently, with one CPU thread
    # per worker to avoid competing thread pools.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for c in [0.1, 0.3, 1, 3]:
            start = perf_counter()
            print(f"Starting C={c}...", flush=True)

            clf = OneVsRestClassifier(
                LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=300,
                    tol=1e-3,
                    class_weight="balanced",
                ),
                n_jobs=2,
            )
            clf.fit(X_train, train_lbl)

            val_preds = clf.predict(X_val)
            val_acc = f1_score(
                val_lbl,
                val_preds,
                average="micro",
                zero_division=0,
            )

            iterations = [
                int(est.n_iter_.max())
                for est in clf.estimators_
                if hasattr(est, "n_iter_")
            ]
            max_iterations = max(iterations, default=0)

            print(
                f"C={c} | "
                f"Time: {perf_counter() - start:.1f}s | "
                f"Validation micro-F1: {val_acc:.4f} | "
                f"Max iterations: {max_iterations}",
                flush=True,
            )

            if val_acc > best_acc:
                best_acc, best_c, best_lr = val_acc, c, clf

    print(
        f"Best C={best_c} | Validation micro-F1={best_acc:.4f}",
        flush=True,
    )

    test_preds_lr = best_lr.predict(X_test)
    results["VLM + LogReg"] = {
        "Micro F1": f1_score(
            test_lbl,
            test_preds_lr,
            average="micro",
            zero_division=0,
        ),
        "Exact Match": accuracy_score(test_lbl, test_preds_lr),
    }

    # 5. MLP Head
    print("\n[VLM] Tuning MLP...")
    best_hidden, best_acc, best_mlp = (128,), -1.0, None
    for hidden in [(128,), (256,), (128, 64)]:
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=mlp_max_iter, random_state=seed)
        clf.fit(train_emb, train_lbl)
        val_acc = f1_score(val_lbl, clf.predict(val_emb), average='micro', zero_division=0)
        # val_acc = accuracy_score(val_lbl, clf.predict(val_emb))  ## If subset accuracy is the selection criteria
        if val_acc > best_acc:
            best_acc, best_hidden, best_mlp = val_acc, hidden, clf

    test_preds_mlp = best_mlp.predict(test_emb)
    results['VLM + MLP'] = {
        "Micro F1": f1_score(test_lbl, test_preds_mlp, average='micro', zero_division=0),
        "Exact Match": accuracy_score(test_lbl, test_preds_mlp)
    }

    return results


def extract_clip_embeddings(model, processor, dataframe, batch_size=64):
    embeddings = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(dataframe), batch_size), desc="CLIP features"):
            images = []
            for path in dataframe.iloc[start:start + batch_size]['image_path']:
                with Image.open(path) as image:
                    images.append(image.convert('RGB'))
            pixels = processor(images=images, return_tensors='pt')['pixel_values']
            pixels = pixels.to(device=model.device, dtype=model.dtype)
            embeddings.append(model.get_image_features(pixel_values=pixels).cpu().float().numpy())
    return np.concatenate(embeddings), np.asarray(dataframe['labels'].tolist(), dtype=np.float32)


def run_encoder(encoder, datasets, prompt, seed=50, qwen_batch_size=1, mlp_max_iter=500):
    train = pd.concat(datasets[:3], ignore_index=True)
    validation, test = datasets[3], datasets[4]
    if encoder == 'qwen':
        if not torch.cuda.is_available():
            raise RuntimeError('Qwen 4-bit mode requires a CUDA GPU. Select L4 in Colab.')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Same 4-bit configuration as the original experiment.
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        backbone = AutoModelForImageTextToText.from_pretrained(
            'Qwen/Qwen3-VL-8B-Instruct', quantization_config=quantization,
            torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
        model = QwenFeatureForward(backbone)
        del backbone
        processor = AutoProcessor.from_pretrained(
            'Qwen/Qwen3-VL-8B-Instruct', min_pixels=256*28*28, max_pixels=256*28*28)
        processor.tokenizer.padding_side = 'left'
        def extract(frame):
            return extract_embeddings(model, processor, frame, prompt, batch_size=qwen_batch_size)
    elif encoder == 'clip':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', torch_dtype=dtype).to(device)
        processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
        def extract(frame):
            return extract_clip_embeddings(model, processor, frame)
    else:
        raise ValueError('encoder must be qwen or clip')
    model.eval()
    model.requires_grad_(False)
    start = perf_counter()
    try:
        train_emb, train_lbl = extract(train)
        val_emb, val_lbl = extract(validation)
        test_emb, test_lbl = extract(test)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'{encoder} extraction: {perf_counter() - start:.1f}s (model loading excluded)', flush=True)
    with threadpool_limits(limits=2):
        return fit_original_heads(train_emb, train_lbl, val_emb, val_lbl, test_emb, test_lbl,
                                  seed=seed, mlp_max_iter=mlp_max_iter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='movies.csv')
    parser.add_argument('--poster-dir', default='posters')
    parser.add_argument('--output-dir', type=Path, default=Path('qwen8b_clip_results'))
    parser.add_argument('--encoders', nargs='+', choices=['qwen', 'clip'], default=['qwen', 'clip'])
    parser.add_argument('--qwen-batch-size', type=int, default=1)
    parser.add_argument('--mlp-max-iter', type=int, default=500, help='Original cap; use 100 for a shorter run')
    parser.add_argument('--pilot-per-split', type=int, default=0, help='Cap each of the five chunks; 0 uses all data')
    args = parser.parse_args()
    if args.qwen_batch_size < 1 or args.mlp_max_iter < 1 or args.pilot_per_split < 0:
        parser.error('Batch size and iterations must be positive; pilot size must be nonnegative')
    seed = 50
    set_seed(seed)
    datasets, mlb = prepare_data(args.csv, args.poster_dir, seed)
    if args.pilot_per_split:
        datasets = [chunk.head(args.pilot_per_split).copy() for chunk in datasets]
        print('PILOT ONLY: these scores are not a full-dataset benchmark.', flush=True)
    if any(chunk.empty for chunk in datasets):
        raise ValueError('Every split must contain at least one usable poster')
    valid_genres = ', '.join(mlb.classes_)
    prompt_json = f"""You are an expert art director and visual analyst.
Your task is to predict a movie's genres based strictly on the visual composition of its poster.

Step 1: Analyze the visual elements, including the color palette, lighting (e.g., dark/moody vs. bright), typography (e.g., distressed vs. elegant), and iconography/props.
Step 2: Based on this visual evidence, select 1 to 3 genres that best fit.

Rules:
- You must choose ONLY from this exact list: {valid_genres}.
- Ignore your outside knowledge of the movie's plot.
- Ignore the meaning of titles or taglines; evaluate only the aesthetic style of the fonts.

Output your response strictly as a JSON object with the following structure. Do not include markdown formatting or outside text:
{{
  "visual_analysis": "A brief 1-2 sentence description of the visual cues you observed.",
  "genres": ["Genre1", "Genre2"]
}}"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([frame.assign(split=name) for frame, name in zip(
        datasets, ['train', 'train', 'train', 'validation', 'test'])]).to_csv(
            args.output_dir / 'split_manifest.csv', index=False)
    results = {}
    for encoder in dict.fromkeys(args.encoders):
        scores = run_encoder(encoder, datasets, prompt_json, seed,
                             args.qwen_batch_size, args.mlp_max_iter)
        label = 'Qwen3-VL-8B' if encoder == 'qwen' else 'CLIP ViT-B/32'
        for name, values in scores.items():
            results[name.replace('VLM', label)] = {
                'Micro F1': values['Micro F1'], 'Subset Accuracy': values['Exact Match']}
        save_results(results, args.output_dir / 'results.md', args.output_dir / 'results.png')
        import matplotlib.pyplot as plt
        plt.close('all')
    print('Saved:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
