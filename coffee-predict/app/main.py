from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Coffee Bean Classification API",
    description="API untuk mendeteksi tingkat kematangan/roasting biji kopi menggunakan ResNet50V2",
    version="1.0.0"
)

# 2. Tambahkan CORSMiddleware agar Swagger UI dan Laravel dapat mengakses API tanpa kendala CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Rekonstruksi Arsitektur Model ResNet50V2 & Load Weights
MODEL_PATH = "app/ResNet50V2-Coffee-Beans.h5"
CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]

try:
    # Arsitektur ini direkonstruksi PERSIS sesuai notebook training aslinya:
    # Sequential([ base_model, GlobalAvgPool2D(), Dense(256, relu), Dropout(0.2), Dense(n_classes, softmax) ])
    #
    # Sebelumnya kita salah asumsi bahwa outputnya langsung base_model -> Dense(4),
    # padahal aslinya ada layer Dense(256, relu) + Dropout(0.2) di antaranya.
    # Itulah sebabnya file .h5 menyimpan 3 layer BERBOBOT (base_model, Dense(256), Dense(4))
    # sementara rekonstruksi lama cuma py 2 (base_model, Dense(4)) -> "expected 2, found 3".
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None,  # Tidak mendownload weights ImageNet karena kita pakai weights sendiri
        input_shape=(256, 256, 3),
        pooling=None
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(256, 256, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")
    ])

    # by_name=False (default): load_weights mencocokkan bobot berdasarkan URUTAN/STRUKTUR
    # layer, bukan berdasarkan nama persis. Jadi selama urutan & jumlah layer berbobotnya
    # sama seperti di atas, ini akan berhasil walau nama layernya beda dari aslinya.
    model.load_weights(MODEL_PATH)
    print("🏆 Model ResNet50V2 & Weights berhasil dibangun dan dimuat melalui rekonstruksi!")

except Exception as e:
    print(f"⚠️ Gagal memuat via rekonstruksi: {e}")
    print("🤖 Mencoba metode fallback (load_model biasa)...")
    try:
        # Jika metode di atas gagal karena perbedaan konfigurasi layer, gunakan pemuatan standar tanpa kompilasi
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("🏆 Model ResNet50V2 berhasil dimuat melalui metode fallback tanpa kompilasi!")
    except Exception as err:
        print(f"❌ Gagal total memuat model: {err}")
        print("💡 Jalankan inspect_h5.py untuk memastikan urutan & bentuk (shape) layer yang tersimpan.")
        model = None

# 4. Fungsi Prapemrosesan Gambar
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Mengubah gambar mentah (bytes) menjadi array yang siap
    dimasukkan ke dalam model ResNet50V2 (256x256 piksel & rescale 1./255).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 5. Endpoint Cek Status Server
@app.get("/")
def home():
    return {
        "status": "active",
        "message": "Server FastAPI Klasifikasi Biji Kopi Aktif!"
    }

# 6. Endpoint Utama Prediksi Gambar
@app.post("/predict")
async def predict_coffee_bean(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model Machine Learning tidak tersedia di server. Periksa log terminal.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Berkas yang diunggah harus berupa gambar (JPEG/PNG).")

    try:
        # Membaca file gambar yang dikirim dari Swagger / Laravel
        image_bytes = await file.read()

        # Eksekusi prapemrosesan gambar
        processed_image = preprocess_image(image_bytes)

        # Prediksi nilai probabilitas gambar menggunakan TensorFlow
        predictions = model.predict(processed_image)

        # Cari indeks dengan probabilitas tertinggi
        predicted_idx = int(np.argmax(predictions[0]))
        confidence_score = float(predictions[0][predicted_idx])

        # Petakan hasil ke dalam seluruh kelas kopi
        all_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": CLASS_NAMES[predicted_idx],
            "confidence": confidence_score,
            "probabilities": all_probabilities
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}")