# Skin Disease Detection System
Deep Learning-Based Multi-Class Skin Disease Classification Web Application

## Table of Contents
- [Overview](#overview)
- [Instalation](#Instalasi)
- [Project Structure](#project-structure)
- [Running the application](#running-the-application)
- [Evaluation Results](#evaluation-results)
- [License & Credits](#license--credits)

---

## Overview

Repositori ini berisi aplikasi web berbasis Streamlit untuk mengklasifikasikan citra lesi kulit ke dalam 22 jenis penyakit kulit menggunakan model deep learning. Aplikasi ini memungkinkan pengguna untuk mengunggah gambar lesi kulit, memilih dari tiga model yang telah dilatih, serta memperoleh prediksi secara instan lengkap dengan distribusi probabilitas kelas dan penjelasan visual (heatmap Grad-CAM pada model yang mendukung).
Sistem ini dirancang khusus untuk tujuan edukasi dan demonstrasi dan tidak ditujukan untuk diagnosis medis.

### 1. Convolutional Neural Networks (CNNs)
Transfer learning dari model pre-trained ImageNet (input 224×224×3).

| Model              | Pre-trained Weights                          | Mode Training yang Diuji                                                                 |
|--------------------|----------------------------------------------|------------------------------------------------------------------------------------------|
| Simple CNN         | Tidak ada (from scratch)                     | Baseline CNN (training penuh dari awal)                                                   |
| ResNet-101         | `ResNet101_Weights.IMAGENET1K_V2`             | Transfer Learning (freeze backbone) • Fine-tuning parsial (unfreeze layer3 & layer4)     |
| ConvNeXt-Base      | `ConvNeXt_Base_Weights.DEFAULT`               | Transfer Learning (full fine-tuning) dengan custom classifier head + LayerNorm & Dropout |

**Notes:**
- ResNet-101 menggunakan strategi fine-tuning parsial dengan membekukan seluruh backbone,
  kemudian membuka kembali `layer3` dan `layer4` untuk meningkatkan adaptasi domain.
- ConvNeXt-Base menggunakan full fine-tuning dengan custom classifier head
  (Flatten → LayerNorm → MLP) untuk menghindari mismatch dimensi LayerNorm.
- Simple CNN digunakan sebagai baseline tanpa pre-training.


### 2. Dataset
Dataset yang digunakan pada penelitian ini adalah Skin Disease Dataset yang diperoleh dari Kaggle:
https://www.kaggle.com/datasets/pacificrm/skindiseasedataset

Dataset ini berisi citra penyakit kulit multikelas dengan total 22 kelas, yang dibagi ke dalam data training dan testing.
**Ringkasan Dataset**
| Keterangan            | Jumlah |
|-----------------------|--------|
| Total Kelas           | 22     |
| Total Data Training   | 13,898 |
| Total Data Testing    | 1,546  |
| Total Seluruh Data    | 15,444 |
| Jenis Data            | Citra RGB (skin disease images) |
| Sumber Dataset        | Kaggle – Skin Disease Dataset   |

**Distribusi Data per Kelas**
| Kelas                    | Train | Test |
|--------------------------|-------|------|
| Acne                    | 593   | 65   |
| Actinic Keratosis       | 748   | 83   |
| Benign Tumors           | 1,093 | 121  |
| Bullous                 | 504   | 55   |
| Candidiasis             | 248   | 27   |
| Drug Eruption           | 547   | 61   |
| Eczema                  | 1,010 | 112  |
| Infestations & Bites    | 524   | 60   |
| Lichen                  | 553   | 61   |
| Lupus                   | 311   | 34   |
| Moles                   | 361   | 40   |
| Psoriasis               | 820   | 88   |
| Rosacea                 | 254   | 28   |
| Seborrheic Keratoses    | 455   | 51   |
| Skin Cancer             | 693   | 77   |
| Sun & Sunlight Damage   | 312   | 34   |
| Tinea                   | 923   | 102  |
| Unknown / Normal        | 1,651 | 189  |
| Vascular Tumors         | 543   | 60   |
| Vasculitis              | 461   | 52   |
| Vitiligo                | 714   | 82   |
| Warts                   | 580   | 64   |
| **Total**               | **13,898** | **1,546** |

**Notes:**
- Dataset bersifat **imbalanced**, dengan kelas *Unknown / Normal* sebagai kelas dominan.
- Jumlah kelas yang relatif besar (22 kelas) menjadikan tugas klasifikasi bersifat **multi-class fine-grained**.
- Dataset digunakan untuk mengevaluasi performa:
  - CNN from scratch
  - Transfer learning (ResNet-101, ConvNeXt-Base)

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/rofiqsamanhudi/skin-disease-detection.git
cd skin desease detection
```

### 2.  Buat & aktifkan virtual environment (sangat disarankan)
```bash
python -m venv venv

# Windows Command Prompt
venv\Scripts\activate

# Windows PowerShell
venv\Scripts\Activate.ps1
# Jika muncul error policy:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

File `requirements.txt` pada repositori ini telah disesuaikan dengan library yang digunakan di dalam kode, termasuk **PyTorch**, **TorchVision**, **NumPy**, **Pandas**, **Scikit-learn**, serta library visualisasi dan utilitas pendukung lainnya.

Apabila Anda **tidak memiliki GPU (CUDA)**, silakan ubah baris instalasi **PyTorch** di dalam `requirements.txt` menjadi versi **CPU-only** sebagai berikut:


```bash
# Core
numpy
pandas

# Visualization
matplotlib
seaborn
pillow
tqdm

# Machine Learning
scikit-learn

# PyTorch (CPU only)
torch
torchvision
torchaudio
```


---

## Project structure

```
skin-disease-detection/
├── assets/                  # Training curves & confusion matrix images
├── dashboard/               # (Possibly additional dashboard components)
├── data/                    # Dataset (not included in repo)
├── model/                   # Trained model files (.pkl)
├── classdict.py             # Class name definitions
├── skindisease.ipynb        # Training notebook (Jupyter)
├── app.py                   # Main Streamlit application (this file)
├── .gitattributes
└── README.md                # This file
```

> **Catatan:** Struktur ini memudahkan pengelolaan dataset, fitur, model, dan eksperimen, sehingga seluruh pipeline end-to-end bisa dijalankan secara terorganisir.


---
## Running the Application

Seluruh sistem diimplementasikan sebagai aplikasi web interaktif berbasis Streamlit untuk klasifikasi penyakit kulit.
Aplikasi ini memungkinkan pengguna mengunggah citra lesi kulit dan memilih model deep learning yang tersedia untuk memperoleh prediksi secara langsung.
**`# Pastikan sudah berada di dalam virtual environment
streamlit run app.py`**
Setelah aplikasi berjalan:
1. Unggah gambar lesi kulit (format .jpg, .jpeg, atau .png)
2. Pilih model yang ingin digunakan
3. Klik tombol EXECUTE ANALYSIS
4. Sistem akan menampilkan hasil prediksi dan informasi pendukung

### Application Workflow

| Tahap                        | Deskripsi                                                                                 |
|------------------------------|-------------------------------------------------------------------------------------------|
| 1. Image Upload | Pengguna mengunggah citra lesi kulit melalui sidebar aplikasi|
| 2. Model Selection   | Pengguna memilih salah satu model klasifikasi yang tersedia                                       |
| 3. Inference   | Model melakukan inferensi pada citra input  |
| 4. Prediction Output | Sistem menampilkan kelas penyakit dengan probabilitas tertinggi                  |
| 5. Model Info Display    | Informasi performa dan konfigurasi training model ditampilkan        |

### Cara Menjalankan

```bash
# Pastikan sudah di dalam virtual environment
jupyter notebook skindisease.ipynb
```
Setelah notebook terbuka:
1. Klik Run -> Run All Cells (atau jalankan satu persatu untuk melihat prosesnya)
2. Tunggu hingga selesai, semua model akan otomatis disimpan beserya log dan grafiknya
---
## Evaluation Results
### Available Models
Aplikasi menyediakan tiga model deep learning dengan karakteristik dan performa yang berbeda:

| Model         | Deskripsi                                                                              |
| ------------- | -------------------------------------------------------------------------------------- |
| CNN Scratch   | Model CNN sederhana yang dilatih sepenuhnya dari awal tanpa pre-training               |
| ResNet-101    | Model berbasis transfer learning menggunakan bobot ImageNet dengan fine-tuning parsial |
| ConvNeXt-Base | Model modern ConvNeXt dengan custom classifier head dan full fine-tuning               |
---
### Model Leaderboard (Test Performance)
| Model          | Test Accuracy | Training Information                                    |
| -------------- | ------------- | ------------------------------------------------------- |
| **ResNet-101** | **79.17%**    | Total epochs trained: 45<br>Final Learning Rate: 1e-06  |
| ConvNeXt Base  | 77.43%        | Total epochs: 30<br>Final Learning Rate: 1e-04          |
| CNN Scratch    | 49.42%        | Total epochs trained: 200<br>Final Learning Rate: 6e-05 |
---
**Disclaimer:**  
Aplikasi ini dikembangkan hanya untuk tujuan edukasi dan demonstrasi.
Hasil prediksi tidak dapat dijadikan sebagai diagnosis medis dan tidak menggantikan konsultasi dengan tenaga kesehatan profesional.

## License & Credits
Proyek ini dikembangkan untuk tujuan edukasi dan eksperimen **Computer Vision**
**Kontributor:**
- Rofiq Samanhudi

**License**
Repository ini dilisensikan di bawah [MIT License](LICENSE), yang memungkinkan penggunaan pribadi, edukatif, dan redistribusi dengan menyertakan atribusi yang sesuai.
