import pandas as pd
import matplotlib.pyplot as plt
from config import set_seed
from dataset import prepare_data
from resnet import run_resnet
from vlm import run_vlm_experiments

def save_results(results, txt_path, png_path):
    df = pd.DataFrame(results).T
    
    # Save nicely formatted markdown table 
    with open(txt_path, "w") as f:
        f.write("# Model Evaluation Results\n\n")
        f.write(df.to_markdown())
        f.write("\n")
    
    # Save figure plot for instant analysis
    ax = df.plot(kind='bar', figsize=(10, 6), rot=25)
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"\n✅ Results successfully exported to {txt_path} and {png_path}")

def main():
    SEED = 50
    set_seed(SEED)
    
    # Phase 1: Data Preparation
    print("=== Phase 1: Data Preparation ===")
    datasets, mlb = prepare_data(csv_path='movies.csv', poster_dir='./posters', seed=SEED)
    
    all_results = {}
    
    # Phase 2: ResNet Baseline
    print("\n=== Phase 2: ResNet-18 Baseline ===")
    resnet_metrics = run_resnet(datasets, num_classes=len(mlb.classes_))
    all_results['ResNet-18'] = resnet_metrics
    
    # Phase 3: VLM Experiments
    print("\n=== Phase 3: VLM Zero-Shot & Feature Extraction ===")
    vlm_metrics = run_vlm_experiments(datasets, mlb, SEED)
    all_results.update(vlm_metrics)
    
    # Phase 4: Export Results
    print("\n=== Phase 4: Exporting Final Results ===")
    save_results(all_results, "results.md", "results.png")

if __name__ == "__main__":
    main()