# 🚗 AI Vision Project

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?style=for-the-badge&logo=tensorflow)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-blue?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-red?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-green?style=for-the-badge)
![Project Active](https://img.shields.io/badge/Project-Active-success?style=for-the-badge)

A modular **computer vision pipeline** for car understanding, including vehicle detection, color classification, brand recognition, license plate detection, and OCR.

---

## 🧠 Architecture Overview

```mermaid
flowchart TD

A[Input Image]

A --> B[YOLOv8m<br/>Vehicle Detection]
B --> C[Crop Vehicle]

C --> D1[EfficientNet<br/>Color Prediction]
C --> D2[PaddleOCR<br/>Text Recognition]
C --> D3[YOLOv8<br/>Plate Detection]
C --> D4[Brand Prediction]

D3 --> E[Crop Plate]


D1 --> F[Final Output]
E --> F
D2 --> F
D4 --> F
```

---
## Project Structure

- **assets/local-statices/**
- **input/** — input images for testing
- **models/** — model configs + pretrained weights (download externally)
- **notebooks/** — experiments, training notebooks
- **outputs/** — pipeline outputs (created with run)
- **src/pipelines/** — core pipeline modules:
  - `brand_predict.py` — vehicle brand prediction
  - `color_predict.py` — vehicle color prediction
  - `ocr.py` — license plate OCR
  - `plate_crop.py` — crop license plates
  - `vehicle_detect.py` — detect vehicles
- **requirements_tensorflow.txt** — TensorFlow dependencies (first virtual environment)
- **requirements_paddle.txt** — Paddle dependencies (second virtual environment)
- **run_pipeline.py** — main script to run the pipeline

---

## ⚙️ Installation

### 1. Clone repo
```bash
git clone https://github.com/abdalmjeedalshami/ai-doroob
cd ai-doroob
```

### 2. Create environments inside `venvs/`
```bash
mkdir venvs
python -m venv venvs/venv_tf
python -m venv venvs/venv_paddle
```

### 3. Install dependencies

**TensorFlow**
```bash
source venvs/venv_tf/bin/activate
pip install -r requirements_tensorflow.txt
source venns/venv_tensorflow/bin/deactivate
```

**Paddle**
```bash
source venvs/venv_paddle/bin/activate
pip install -r requirements_paddle.txt
source venvs/venv_paddle/bin/deactivate
```

---

## 🚨 Important Notes

- **Do NOT mix environments**
- Always activate the correct virtual environment before installing packages
- Do not upload `venvs/` or large outputs to GitHub

---

## 🤖 Models

### ✅ Pretrained models (currently used)
The project requires the following pretrained model weights. <b>These are not included in the repository</b> and must be downloaded from <b>Google Drive</b>:

[**Download Pretrained Models**](https://drive.google.com/drive/folders/1Kx7fkI68of8oPhWpgpJ6HVY1ez3ZUASm?usp=drive_link)

After downloading, replace the `models/` folder. The pipeline will automatically use them.

- `yolov8m.pt` → vehicle detection
- `license_plate_detector.pt` → license plate detection

---

### 🚘 Car Brand Model

- Architecture: **ResNet18 (pretrained=True)**
- Fully trainable and reproducible

To retrain or recreate:

➡️ Run the notebook inside:
```
src/training/training-brand-model/notebook.ipynb
```

Requirements:
- Dataset must be inside the same folder
- Follow the included README in that directory

---

## ▶️ Usage

Run the full pipeline:

```bash
python run_pipeline.py
```

Put your test images inside:
```
input/
```

---

## 🧰 Tools

- Python 3.9+
- TensorFlow / Paddle
- YOLOv8
- PaddleOCR
- Jupyter Notebook
- VS Code / PyCharm

---

## 📬 Contact

GitHub:
https://github.com/abdalmjeedalshami/ai-doroob
