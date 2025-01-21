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
├── Weights/                        # Trained model weights (To be uploaded)
│   ├── mixed_cohort                # Pretrained model weights from experiments
│   │   └── CCVAEGAN                    
|   │       ├── encoder.pth
|   │       ├── decoder.pth
|   │       └── dis_GAN.pth            
|    └── external_validation
|        └── CCVAEGAN
|           ├── encoder.pth
|           ├── decoder.pth
|           └── dis_GAN.pth            
│
├── src/                            # Source code
│   ├── main.py                     # Main code to run CCVAEGAN framework
│   ├── train_framework.py          # Training implementation
│   ├── evaluation_framework.py     # Evaluation and anomaly detection including visualization implementation
│   ├── models/                         # Trained model weights
│   │   ├── CCVAEGAN.py                 # Model strucutre
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
git clone https://github.com/VictorSungminYou/Fetal-Brain-AnomalyDetection
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

## Singularity Deloyment
Use following Singularity image to run CCVAEGAN with pre-trained weights.

https://www.dropbox.com/scl/fi/cq42n9of4kkdgahuzfd1f/fetal_anomaly_detection.sif?rlkey=f3jgm0mprg7qdbef2rwnitg8h&st=tyh938zj&dl=0  

### It requires following data structure.
```
├── MRI/                           # Data directory contains MRI files
└── Info.csv                       # CSV file contains demographic information (mainly GA and Sex)
```

### How to run
```
singularity run --no-home -B data_path:/data --nv path_to_SIF model path_to_Info.csv anomaly_map_option vis_threshold_MAE vis_threshold_MSE GPU_id
```
path_to_Info.csv : The file contains demographic information with subject key. Refer the format of example file (Info_example.csv)

anomaly_map_option: option to plot anomaly maps (‘center’: only center slide (idx=14), ‘all’: save all 30 slices (it take much longer), ‘no’: Compute score only without saving anomaly map)

vis_threshold_MAE: visualization range of MAE anomaly map (‘auto’ is allowed)

vis_threshold_MSE: visualization range of MSE anomaly map (‘auto’ is allowed)

GPU_id: The GPU ID number to run the model (0, 1, ...)

#### Example
```
singularity run --no-home -B ./:/data --nv fetal_anomaly_detection.sif /data/Demo_info.csv 'center' 0.4, 0.08 0
```

