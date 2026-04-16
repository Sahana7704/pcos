# 🔐 PCOS Image Security — DCT Watermarking & Tamper Detection

A deep learning pipeline for **PCOS (Polycystic Ovary Syndrome) detection** from ultrasound images, combined with **invisible DCT-domain watermarking** for medical image integrity verification.

---

## 🧠 Project Overview

This project combines two goals:

1. **Medical AI** — Classify ultrasound images as PCOS or Non-PCOS using a fine-tuned VGG16 model.
2. **Image Security** — Embed invisible spread-spectrum watermarks into medical images using DCT (Discrete Cosine Transform) and detect any tampering via SHA-256 hash verification.

---

## ✨ Features

- **DCT Watermarking** — Invisible watermarks embedded in the frequency domain (green channel) of each image
- **SHA-256 Hash Registry** — Each watermarked image is registered by hash for tamper detection
- **VGG16 Classifier** — Transfer learning on ImageNet weights, fine-tuned for binary PCOS classification
- **CLAHE Preprocessing** — Contrast-limited adaptive histogram equalization for ultrasound images
- **Pseudo-3D Volume Viewer** — Interactive HTML canvas viewer showing depth slices of ultrasound images
- **Streamlit App** — Web UI for uploading images and checking tamper status in real time

---

## 📁 Project Structure

```
pcos-security/
│
├── PCOS_SECURITY.ipynb     # Main Colab notebook (full pipeline)
├── app.py                  # Streamlit tamper-detection web app
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to exclude from Git
└── README.md               # This file
```

> **Note:** The dataset (`infected/` and `noninfected/` image folders) is **not** included in this repo. See the setup section below.

---

## 🗂️ Dataset

This project uses the **PCOS Ultrasound Image Dataset** from Kaggle:

- Folder structure expected:
  ```
  MyDrive/PCOS/
  ├── infected/        ← PCOS images
  └── noninfected/     ← Non-PCOS images
  ```

Upload your dataset to **Google Drive** and update the `DATASET_DIR` path in the notebook.

---

## 🚀 Getting Started

### 1. Run the Notebook (Google Colab)

1. Upload `PCOS_SECURITY.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Mount your Google Drive
3. Set `DATASET_DIR` to your PCOS image folder path
4. Run all cells top to bottom

The pipeline will:
- Load and sample the dataset
- Apply DCT watermarks to all images
- Train the VGG16 classifier
- Evaluate and visualize results
- Save the model, confusion matrix, and training curves to Drive

### 2. Export the Hash Registry

After running the notebook, export the hash registry so the Streamlit app can use it:

```python
import json
with open("/content/drive/MyDrive/PCOS/hash_registry.json", "w") as f:
    json.dump(hash_registry, f, indent=2)
```

Download `hash_registry.json` from Drive and place it alongside `app.py`.

### 3. Run the Streamlit App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run it **inside Colab** using ngrok (already in the notebook).

---

## ⚙️ Configuration

Key parameters in the notebook (edit at the top of Cell 2):

| Parameter         | Default         | Description                              |
|-------------------|-----------------|------------------------------------------|
| `DATASET_DIR`     | `/content/drive/MyDrive/PCOS` | Path to your dataset on Drive |
| `IMG_SIZE`        | `(224, 224)`    | Input image size for VGG16               |
| `EPOCHS`          | `20`            | Max training epochs                      |
| `BATCH_SIZE`      | `16`            | Training batch size                      |
| `LEARNING_RATE`   | `1e-4`          | Adam optimizer learning rate             |
| `SAMPLE_FRACTION` | `0.5`           | Fraction of each class to use (1.0 = all)|
| `DEPTH_SLICES`    | `5`             | Number of pseudo-3D depth slices         |

---

## 🔬 How the Watermarking Works

The `DCTWatermark` class embeds an invisible watermark using **spread-spectrum DCT steganography**:

1. Convert the image to grayscale
2. Split into 8×8 pixel blocks
3. Apply DCT to each block
4. Encode the image ID as bits and modulate the `(4,4)` DCT coefficient (`±15.0` strength)
5. Apply inverse DCT and write back to the **green channel**

Tamper detection is done via **SHA-256 hash comparison** — any pixel-level modification changes the hash.

---

## 📊 Model Architecture

```
Input (224×224×1)
  → Lambda (repeat to 3 channels)
  → VGG16 (frozen ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(256, ReLU) + Dropout(0.5)
  → Dense(128, ReLU)
  → Dense(1, Sigmoid)
```

Trained with **Binary Cross-Entropy** loss and **Adam** optimizer.

---

## 📦 Requirements

See `requirements.txt`. Key packages:

- `tensorflow >= 2.10`
- `opencv-python`
- `scikit-learn`
- `streamlit`
- `Pillow`
- `matplotlib`
- `numpy`

---

## 📄 References

- Eswaraiah, R., & Sreenivasa Reddy, E. — *Medical Image Watermarking Technique for Accurate Tamper Detection in ROI and Exact Recovery of ROI*
- Zhou et al. (2025) — Pseudo-3D ultrasound visualization

---

## 📝 License

This project is for academic and research purposes.
