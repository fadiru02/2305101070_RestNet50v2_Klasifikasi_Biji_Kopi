# ☕ Coffee Bean Classification API

API sederhana berbasis **FastAPI** untuk mendeteksi tingkat kematangan/roasting biji kopi (**Dark**, **Green**, **Light**, **Medium**) menggunakan model **ResNet50V2** hasil transfer learning.

---

## 📂 Struktur Folder

Pastikan struktur folder project kamu seperti berikut sebelum menjalankan server:

```
coffee-bean-api/
├── app/
│   └── ResNet50V2-Coffee-Beans.h5   # File model hasil training (.h5)
├── main.py                          # Kode utama FastAPI
├── requirements.txt                 # Daftar dependensi Python
└── README.md
```

> ⚠️ **Penting:** File model `ResNet50V2-Coffee-Beans.h5` **wajib** diletakkan di dalam folder `app/`, karena path model di `main.py` mengacu ke `app/ResNet50V2-Coffee-Beans.h5`.

---

## 🧰 Persyaratan (Requirements)

- Python **3.9 – 3.11** (disarankan, agar kompatibel dengan TensorFlow)
- pip
- (Opsional) GPU + CUDA jika ingin inferensi lebih cepat

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/coffee-bean-api.git
cd coffee-bean-api
```

### 2. Buat Virtual Environment (disarankan)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

Buat file `requirements.txt` (jika belum ada) dengan isi berikut:

```txt
fastapi
uvicorn[standard]
tensorflow
numpy
pillow
python-multipart
```

Kemudian install:

```bash
pip install -r requirements.txt
```

### 4. Letakkan File Model

Pastikan file `ResNet50V2-Coffee-Beans.h5` sudah berada di dalam folder `app/`.

### 5. Jalankan Server

```bash
uvicorn main:app --reload
```

Jika berhasil, kamu akan melihat log seperti ini di terminal:

```
🏆 Model ResNet50V2 & Weights berhasil dibangun dan dimuat melalui rekonstruksi!
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Secara default, server berjalan di:
```
http://127.0.0.1:8000
```

---

## 📖 Dokumentasi API (Swagger UI)

FastAPI otomatis menyediakan dokumentasi interaktif. Buka salah satu URL berikut di browser:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 Endpoint yang Tersedia

### 1. Cek Status Server

```
GET /
```

**Contoh Response:**
```json
{
  "status": "active",
  "message": "Server FastAPI Klasifikasi Biji Kopi Aktif!"
}
```

### 2. Prediksi Gambar Biji Kopi

```
POST /predict
```

**Request:** `multipart/form-data`

| Field  | Tipe   | Keterangan                          |
|--------|--------|--------------------------------------|
| `file` | File   | Gambar biji kopi (JPEG/PNG)          |

**Contoh menggunakan cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@contoh_biji_kopi.jpg"
```

**Contoh Response Sukses:**
```json
{
  "status": "success",
  "filename": "contoh_biji_kopi.jpg",
  "prediction": "Medium",
  "confidence": 0.9732,
  "probabilities": {
    "Dark": 0.0102,
    "Green": 0.0021,
    "Light": 0.0145,
    "Medium": 0.9732
  }
}
```

---

## 🌐 Integrasi dengan Frontend (Laravel, dll.)

CORS sudah diaktifkan untuk semua origin (`allow_origins=["*"]`), sehingga API ini bisa langsung diakses dari aplikasi frontend seperti Laravel, React, atau aplikasi mobile tanpa masalah CORS.

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---|---|
| `Model Machine Learning tidak tersedia di server` | Pastikan file `.h5` ada di path `app/ResNet50V2-Coffee-Beans.h5` |
| Gagal memuat model (`Gagal total memuat model`) | Cek kembali arsitektur model dan urutan layer menggunakan script `inspect_h5.py` |
| Error `ModuleNotFoundError` | Pastikan semua dependensi di `requirements.txt` sudah terinstall |
| Server lambat saat prediksi pertama kali | Wajar, karena TensorFlow melakukan inisialisasi graph di request pertama |

---

## 🛠️ Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) – Web framework
- [TensorFlow / Keras](https://www.tensorflow.org/) – Deep Learning
- [Pillow](https://python-pillow.org/) – Image processing
- [Uvicorn](https://www.uvicorn.org/) – ASGI server

---
