# Conditional Deep Generative Normative Modeling for Fetal Brain Anomaly Detection

## Overview  
This repository contains the implementation of **Conditional Cyclic Variational Autoencoding Generative Adversarial Network (CCVAEGAN)** for detecting structural and developmental anomalies in fetal brain MRIs. The model leverages conditional generative models and cyclic consistency training to produce high-quality normative fetal brain references and achieve superior anomaly detection performance.

---

## Features  
- **Deep Generative Model**: Combines Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs).  
- **Conditional Learning**: Incorporates gestational age (GA) conditioning via the brain age estimator.  
- **Cyclic Consistency Training**: Ensures bidirectional reconstruction to enhance image fidelity.  
- **Anomaly Detection**: Quantitative anomaly scores for identifying and localizing brain abnormalities.  
- **Multisite Validation**: Demonstrates generalizability on datasets from multiple institutions.  

---

## Highlights  
- **Model Performance**: Near-perfect AUROC > 0.99 across various anomaly metrics.  
- **Visualization**: Anomaly maps for intuitive localization of structural deviations.  
- **Generalizability**: Effective performance on external site datasets.  

---

## Repository Structure (including desired data/output structure)  
```
├── data/                           # Data directory (not included)
│   ├── train/                      # Training images (TD, i.e., normal)
│   └── test/                       # Test images (TD and anomalies)
│
├── models/                         # Trained model weights (To be uploaded)
│   └── CCVAEGAN                    # Pretrained CCVAEGAN weights
|        ├── encoder.pth
|        ├── decoder.pth
|        └── dis_GAN.pth            
│
├── src/                            # Source code
│   ├── main.py                     # Main code to run CCVAEGAN framework
│   ├── train_framework.py          # Training implementation
│   ├── evaluation_framework.py     # Evaluation and anomaly detection including visualization implementation
|   └── utils/                            
│       ├── eval_utils.py           # Evaluation functions
│       ├── loss.py                 # Loss functions
│       ├── metric.py               # Evaluation metric
│       └── util.py                 # General Helper functions
├── notebooks/                      # Jupyter notebooks for experiments
│   └── Outcome_analysis.ipynb      # Analysis of anomaly score
│
├── results/                        # Results directory (Not included)
│
├── enviroment.yml                  # Environment Libraries for Anaconda enviroment
└── README.md                       # Project documentation
```

---

## Requirements  
- Python 3.9  
- PyTorch 2.2.1  
- CUDA (for GPU acceleration): cudatoolkit 11.8.0, cudnn 8.9.2
- For detailed package information, please check enviroment.yml file.

---

## Setup  
### 1. Clone the repository  
```bash
git clone https://github.com/VictorSungminYou/Fetal_Brain_AnomalyDetection
cd CCVAEGAN_Fetal_Brain_AnomalyDetection
```

### 2. Install dependencies  
```bash
conda env create -f enviroment.yml
```

### 3. Prepare the dataset  
- Fetal brain MRIs are expected in the `/data/` directory.  
- Ensure appropriate preprocessing (motion correction, brain extraction).

---

## Usage  
### 1. Train the Model  
Train CCVAEGAN using the provided script:  
```bash
python Src/main.py --task Training ...
```

### 2. Test and Evaluate  
Run anomaly detection on test data:  
```bash
python Src/main.py --task Evaluation ...
```

### 3. Analysis of output score files  
Use Jupyter notebooks to perform analysis including post-organization:  

---
