import os
import requests
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MultiLabelBinarizer
from config import set_seed

def download_image(row, poster_dir):
    """Download poster images to local storage."""
    url, movie_id = row['Poster_Url'], row['movie_id']
    save_path = os.path.join(poster_dir, f"{movie_id}.jpg")
    
    if pd.isna(url) or os.path.exists(save_path):
        return
        
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
    except Exception:
        pass 

def prepare_data(csv_path='movies.csv', poster_dir='./posters', seed=50):
    """Clean data, trigger concurrent downloads, and encode genres."""
    print("Preparing data and downloading posters...")
    os.makedirs(poster_dir, exist_ok=True)
    df = pd.read_csv(csv_path, engine='python')
    
    df.dropna(subset=['Poster_Url', 'Genre'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['movie_id'] = df.index
    df['image_path'] = os.path.abspath(poster_dir) + '/' + df['movie_id'].astype(str) + '.jpg'

    # Download in parallel using 32 threads
    records = df.to_dict('records')
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(download_image, r, poster_dir) for r in records]
        for f in futures:
            f.result()

    # Filter out missing images due to dead links
    missing_mask = ~df['image_path'].apply(os.path.exists)
    if missing_mask.sum() > 0:
        df = df[~missing_mask].reset_index(drop=True)
        
    # Apply MultiLabel Binarizer
    mlb = MultiLabelBinarizer()
    df['labels'] = mlb.fit_transform(df['Genre'].apply(lambda x: x.split(', '))).tolist()
    
    # Shuffle and split into 5 chunks with fixed reproducible seed
    set_seed(seed)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    datasets = np.array_split(df_shuffled, 5)
    
    return datasets, mlb