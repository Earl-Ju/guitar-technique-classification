# Guitar Technique Classification 🎸

**A work-in-progress project to explore Music Information Retrieval (MIR) by automatically identifying guitar playing techniques (e.g., bending, vibrato, slides) from audio signals.**

---

## Project Description

This is my first computational music project.  
As a guitar player and maker with over 20 years of experience, I am leveraging my domain knowledge to explore how machine learning can be used to understand and categorize nuances in musical performance.

🎯 **Goal**: Build a tool that can serve as a practice aid for guitar students or an analytical tool for musicologists.

---

## Current Status & Learning Goals

- **Learning the fundamentals**: Python, Machine Learning, and Digital Signal Processing (DSP).  
- **Toolchain setup**: Exploring [Librosa](https://librosa.org) for audio feature extraction and [Scikit-learn](https://scikit-learn.org/) for building models.  
- **Data collection**: Experimenting with datasets (e.g., [IDMT-SMT-Guitar](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/guitar.html)) and planning to record my own samples.  

---

## Next Steps

1. Extract MFCC, spectral contrast, and chroma features from sample audio.  
2. Implement a baseline classifier (e.g., SVM) to test feasibility.  
3. Experiment with different neural network architectures (CNN, CRNN).  

---

## Repository Structure

```commandline
guitar-technique-classification/
├── data/               # Local datasets (ignored by Git)
│   ├── raw/            # Raw audio + annotations
│   ├── processed/      # Extracted features (CSV, NPY)
│   └── external/       # External datasets (Zenodo, others)
│
├── notebooks/          # Experiments & exploration
│   ├── 01_feature_extraction.ipynb
│   ├── 02_baseline_SVM.ipynb
│   └── 03_CNN_CRNN.ipynb
│
├── src/                # Core source code
│   ├── features.py     # Feature extraction
│   ├── dataset.py      # Dataset loading & splitting
│   ├── models.py       # SVM / CNN / CRNN definitions
│   ├── train.py        # Training pipeline
│   └── evaluate.py     # Evaluation (Accuracy, F1, Confusion Matrix)
│
├── results/            # Experimental results (ignored by Git)
│   ├── svm_baseline/
│   ├── cnn/
│   └── crnn/
│
├── quick_test/         # Quick test scripts (e.g., first_audio_analysis.py)
│
├── environment.yml     # Conda environment setup
├── LICENSE             # MIT License
└── README.md           # Project documentation


``` 
## Environment

The project uses **Conda** for reproducibility.

```bash
conda env create -f environment.yml
conda activate guitar-technique-classification
```

### Key dependencies
- Python 3.10.*
- numpy, scipy, scikit-learn
- librosa, matplotlib
- jupyter, gradio

---

## About Me

I am a music industry professional and Master Builder of electric guitars, now transitioning into the field of music technology.

- 💡 **Research interest:** AI-assisted music education & intelligent instrument design
- 📫 **Connect with me:** [LinkedIn / personal page — optional]
- ⚠️ Please check and respect dataset licenses (e.g., non-commercial clauses).

---

## Quickstart Demo

To quickly test the pipeline without needing a real dataset, you can generate a minimal dummy dataset and run training:

```bash
# 1. Generate dummy dataset (creates simple sine waves for "bending" and "vibrato")
python generate_dummy_data.py

# 2. Run training & evaluation on the dummy dataset
python -m src.train --data_dir data/processed --out_dir results/quick_test

```
```commandline
This will:

Create a toy dataset under data/processed/ with 2 classes (bending, vibrato).

Extract features (MFCCs).

Train and evaluate a simple SVM classifier.

Save results (metrics, logs, models) into results/quick_test/.
```










