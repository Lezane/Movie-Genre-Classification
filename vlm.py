import json
import re
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from time import perf_counter
from sklearn.preprocessing import StandardScaler
from joblib import parallel_config

def run_zero_shot(model, processor, df_subset, prompt, max_tokens=256, batch_size=32):
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(df_subset), batch_size), desc="Zero-Shot Inferencing"):
            batch = df_subset.iloc[i:i+batch_size]
            messages = [[{"role": "user", "content": [{"type": "image", "image": row['image_path']}, {"type": "text", "text": prompt}]}] for _, row in batch.iterrows()]
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            
            trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            preds = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
            predictions.extend([p.strip() for p in preds])
    return predictions

def parse_json_predictions(predictions, valid_classes):
    parsed = []
    for pred in predictions:
        genres = []
        try:
            match = re.search(r'\{.*\}', pred, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                genres = data.get("genres", [])
        except Exception:
            pass
            
        if not genres:
            array_match = re.search(r'"genres"\s*:\s*\[(.*?)\]', pred, re.DOTALL | re.IGNORECASE)
            if array_match:
                genres = re.findall(r'"([^"]+)"', array_match.group(1))
        
        if isinstance(genres, list):
            parsed.append([g for g in genres if g in valid_classes])
        else:
            parsed.append([])
    return parsed

def extract_embeddings(model, processor, df_subset, prompt, batch_size=16):
    embeddings, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(df_subset), batch_size), desc="Extracting Features"):
            batch = df_subset.iloc[i:i+batch_size]
            messages = [[{"role": "user", "content": [{"type": "image", "image": row['image_path']}, {"type": "text", "text": prompt}]}] for _, row in batch.iterrows()]
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            
            embeddings.append(outputs.hidden_states[-1][:, -1, :].cpu().float())
            labels.extend([torch.tensor(row['labels'], dtype=torch.float32) for _, row in batch.iterrows()])
            
    return torch.cat(embeddings).numpy(), torch.stack(labels).numpy()

def run_vlm_experiments(datasets, mlb, seed):
    df_1, df_2, df_3, df_4, df_5 = datasets
    valid_genres = ", ".join(mlb.classes_)
    
    print("Loading Qwen-VL Model (Takes time)...")
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=256*28*28)
    processor.tokenizer.padding_side = "left"

    results = {}

    # 1. Zero-Shot Naive
    print("\n[VLM] Running Zero-Shot Naive on df_1...")
    prompt_naive = f"You are an expert movie reviewer. Look at this movie poster.\nBased ONLY on the visual style, predict the movie genres.\nChoose from the following list: {valid_genres}.\nOutput ONLY a comma-separated list of genres. Do not explain."
    preds_naive = run_zero_shot(model, processor, df_1, prompt_naive, max_tokens=50)
    parsed_naive = [[p.strip() for p in pred.split(',') if p.strip() in mlb.classes_] for pred in preds_naive]
    
    true_matrix_zs = mlb.transform(df_1['Genre'].apply(lambda x: x.split(', ')))
    zs_pred_matrix = mlb.transform(parsed_naive)
    results['Zero-Shot (Naive)'] = {
        "Micro F1": f1_score(true_matrix_zs, zs_pred_matrix, average='micro', zero_division=0),
        "Exact Match": accuracy_score(true_matrix_zs, zs_pred_matrix)
    }

    # 2. Zero-Shot JSON
    print("\n[VLM] Running Zero-Shot JSON on df_1...")
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
    preds_json = run_zero_shot(model, processor, df_1, prompt_json, max_tokens=300)
    parsed_json = parse_json_predictions(preds_json, mlb.classes_)
    zs_json_matrix = mlb.transform(parsed_json)
    results['Zero-Shot (JSON)'] = {
        "Micro F1": f1_score(true_matrix_zs, zs_json_matrix, average='micro', zero_division=0),
        "Exact Match": accuracy_score(true_matrix_zs, zs_json_matrix)
    }

    # Free up memory before extracting massive feature mappings
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Feature Extraction
    df_train = pd.concat([df_1, df_2, df_3]).reset_index(drop=True)
    df_val = df_4.copy()
    df_test = df_5.copy()
    
    print("\n[VLM] Extracting Embedded Features...")
    train_emb, train_lbl = extract_embeddings(model, processor, df_train, prompt_json)
    val_emb, val_lbl = extract_embeddings(model, processor, df_val, prompt_json)
    test_emb, test_lbl = extract_embeddings(model, processor, df_test, prompt_json)

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
    best_hidden, best_acc, best_mlp = (128,), 0.0, None
    for hidden in [(128,), (256,), (128, 64)]:
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=seed)
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
