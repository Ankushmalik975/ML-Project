# ML-Project
# Sensor Fusion Gesture Classification (CMI/BFRB)

This repository contains a complete deep-learning pipeline for classifying human gestures and behaviors using **multi‑modal sensor data** (IMU + TOF + Thermal). The project includes data preprocessing, feature engineering, model design, K-Fold training, and inference server setup for Kaggle competitions.

---

 🚀 Features

* **Multi‑modal sensor fusion** using IMU + TOF/THM data.
* **Deep CNN + Attention architecture** for time‑series classification.
* **Separable convolutions** for efficient feature extraction.
* **Squeeze‑and‑Excitation (SE) channel attention**.
* **Temporal Multi‑Head Attention** for sequence modeling.
* **Comprehensive data augmentation** (noise, scaling, shifting, dropout).
* **StandardScaler normalization + low‑pass filtering** for IMU signals.
* **Stratified K‑Fold training** with early stopping.
* **Model ensemble inference** for better accuracy.
* **Kaggle-compatible inference server** included.

---

📁 Project Structure

```
|-- train.py / main.py        # Full pipeline (same as code above)
|-- README.md                 # Project documentation
|-- models/                   # Saved model checkpoints
|-- input/                    # Dataset directory
|   |-- train.csv
|   |-- test.csv
|   |-- test_demographics.csv
|-- artifacts/                # Saved normalizers + metadata
```

---

 🧠 Model Architecture Overview

 1. **IMU Feature Engineering**

* Magnitude features (acc + rotation)
* 1D grouped convolution on accelerometer channels
* Combines raw + engineered features

### 2. **IMU CNN Pathway**

* 2× SeparableConv blocks (Depthwise + Pointwise)
* MaxPooling + SE Attention

### 3. **TOF/THM Pathway**

* Simpler Conv → BN → SiLU → Pool network

### 4. **Fusion + Temporal Attention**

* Concatenate IMU + TOF features
* Multi‑Head Self‑Attention

### 5. **Classifier Head**

* Global Average Pooling
* Dense layers with BatchNorm + SiLU
* Output logits

---

## 🛠️ How to Train

1. Place dataset inside:

```
/input/cmi-detect-behavior-with-sensor-data/
```

2. Ensure `TRAIN_MODE = True` in the config.
3. Run:

```bash
python main.py
```

The script will:

* preprocess and normalize data
* train K-fold models
* save model checkpoints + metadata in the output directory

---

## 🔍 Inference

1. Set:

```
TRAIN_MODE = False
```

2. Place pretrained model files in:

```
/kaggle/input/cmi-models/
```

3. Run:

```bash
python main.py
```

The inference server will:

* load all folds
* preprocess inputs
* run ensemble prediction
* return gesture label

---

## 📦 Artifacts Saved

* `sensor_model_foldX.pth` – trained weights
* `sensor_feature_cols.npy` – feature names
* `behavior_classes.npy` – label encoder classes
* `data_normalizer.pkl` – StandardScaler

---

## 📊 Data Preprocessing

### Includes:

* Forward/backward fill for missing data
* Low-pass Butterworth filter for IMU
* StandardScaler normalization
* Fixed-length padding/truncation (100 timesteps)

---

## 🧪 Data Augmentation

Applied only in training:

* Gaussian noise on IMU
* Random scaling
* Time shift (roll)
* Non‑IMU sensor dropout

---

## 🖥️ Kaggle Inference Compatibility

The project includes:

* Mock server for local testing
* Kaggle CMIInferenceServer integration

---

## ⚙️ Requirements

* Python 3.10+
* PyTorch
* Scikit-Learn
* NumPy / Pandas / Polars
* SciPy
* joblib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✨ Acknowledgements

This project is inspired by real‑time gesture classification challenges and built with performance, robustness, and clarity in mind.

---

## 📫 Contact

For doubts or improvements, feel free to create an issue or pull request on GitHub.
