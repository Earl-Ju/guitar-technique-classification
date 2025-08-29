# Guitar Technique Classification 🎸

A computational music project using **Music Information Retrieval (MIR)** and **deep learning** to classify guitar playing techniques.  
The goal is to detect techniques from monophonic notes/short phrases in guitar solos.  

Currently supported **6 techniques**:  
- `normal` (standard note)  
- `bend`  
- `vibrato`  
- `slide`  
- `harmonics`  
- `dead-note`

---

## Project Description

This project explores **MIR-based methods** for guitar technique classification.  
We begin with a **traditional feature + SVM baseline** (reference ≈ 67–70% accuracy for 6 classes)  
and then move to **CNN/CRNN architectures** for better performance.  

### Motivation
- Build a tool that can serve as a **practice aid for guitar learners**.  
- Provide an **analytical resource for musicologists** studying expressive performance.  
- Explore connections with **pitch/note tracking** for future improvements (TENT-inspired idea).

---

## Datasets

1. **IDMT-SMT-Guitar**  
   - Contains ≈4700 annotated note events with multiple techniques.  
   - License: non-commercial use.  
   - ✅ Primary dataset for baseline & CNN/CRNN experiments.  

2. **Multimodal Guitar Technique Dataset (IIT Demokritos)**  
   - Audio + video, 9 technique classes.  
   - Used for transfer learning / extension.  

3. **Guitar-TECHS (2025, arXiv)**  
   - New dataset with extended technique coverage.  
   - For future experiments.  

⚠️ Please check and respect dataset licenses (e.g., non-commercial clauses).

---

### Repository Structure

```commandline
guitar-technique-classification/
├── data/ # Local datasets (ignored by Git)
│ ├── raw/ # Raw audio + annotations
│ ├── processed/ # Extracted features (CSV, NPY)
│ └── external/ # External datasets (Zenodo, others)
│
├── notebooks/ # Experiments & exploration
│ ├── 01_feature_extraction.ipynb
│ ├── 02_baseline_SVM.ipynb
│ └── 03_CNN_CRNN.ipynb
│
├── src/ # Core source code
│ ├── features.py # Feature extraction
│ ├── dataset.py # Dataset loading & splitting
│ ├── models.py # SVM / CNN / CRNN definitions
│ ├── train.py # Training pipeline
│ └── evaluate.py # Evaluation (Accuracy, F1, Confusion Matrix)
│
├── results/ # Experimental results (ignored by Git)
│ ├── svm_baseline/
│ ├── cnn/
│ └── crnn/
│
├── environment.yml # Conda environment setup
├── first_audio_analysis.py # Quick test script
├── README.md # Project documentation
└── LICENSE # MIT License
```







