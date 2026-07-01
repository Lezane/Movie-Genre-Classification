# Visual Movie Genre Classification 🎬

This repository predicts a movie's genres purely based on its poster composition. It contrasts two approaches:
1. **ResNet-18 Baseline:** A conventional CNN directly trained end-to-end on image spatial pixel data.
2. **Qwen3-VL-4B-Instruct:** A Vision-Language Model serving via explicit Zero-Shot prompting alongside a dense representational feature extractor (probing linear logistic regression and MLP classifications).

## Getting Started

1. Ensure your original dataset `movies.csv` is seated inside the project's root folder.
2. Install the necessary requirements:
```bash
pip install -r requirements.txt