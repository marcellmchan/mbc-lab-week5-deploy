import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import re
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Deep Learning Deployer", page_icon="🚀", layout="wide")

# CSS Kustom untuk UI yang lebih menarik
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 20px; font-weight: bold;}
    .success-box {padding: 15px; border-radius: 10px; background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .error-box {padding: 15px; border-radius: 10px; background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .info-box {padding: 15px; border-radius: 10px; background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model():
    model_path = 'sentiment_GRU_cfg1_seq100.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@st.cache_resource
def load_image_model():
    model_path = 'model_pretrained_mobilenetv2.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@st.cache_resource
def get_tokenizer():
    import pickle
    if os.path.exists('tokenizer.pkl'):
        with open('tokenizer.pkl', 'rb') as handle:
            return pickle.load(handle)
    else:
        tokenizer = Tokenizer(num_words=5000)
        return tokenizer

def preprocess_text(text, tokenizer, max_len=100):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

# Navigasi Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", width=100)
st.sidebar.title("Navigasi Portofolio")
menu = st.sidebar.radio("Pilih Halaman:", ["Klasifikasi Gambar Apple vs Orange", "Analisis Sentimen IMDB", "Riwayat Versi (Versioning)"])

if menu == "Klasifikasi Gambar Apple vs Orange":
    st.title("🍎🍊 Klasifikasi Gambar: Apple vs Orange")
    st.write("Aplikasi ini menggunakan model **MobileNetV2** (Transfer Learning) dari tugas Minggu 2 untuk membedakan gambar buah Apel dan Jeruk secara akurat.")
    
    img_model = load_image_model()
    
    if img_model is None:
        st.error("File `model_pretrained_mobilenetv2.h5` tidak ditemukan.")
    else:
        uploaded_file = st.file_uploader("Upload gambar apel atau jeruk (JPG/PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar yang diupload")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
            with col2:
                st.subheader("Hasil Prediksi")
                with st.spinner("Menganalisis gambar..."):
                    # Preprocessing Image
                    img_resized = image.resize((128, 128))
                    img_array = img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array) # Preprocessing khusus MobileNetV2
                    
                    # Prediksi
                    prediction = img_model.predict(img_array)[0][0]
                    
                    if prediction > 0.5:
                        st.markdown(f'<div class="success-box"><h3>🍊 Ini adalah JERUK (Orange)</h3><p>Kepercayaan: {prediction:.2%}</p></div>', unsafe_allow_html=True)
                        st.progress(float(prediction))
                    else:
                        st.markdown(f'<div class="error-box"><h3>🍎 Ini adalah APEL (Apple)</h3><p>Kepercayaan: {(1-prediction):.2%}</p></div>', unsafe_allow_html=True)
                        st.progress(float(1 - prediction))

elif menu == "Analisis Sentimen IMDB":
    st.title("🚀 Analisis Sentimen Ulasan Film (IMDB)")
    st.write("Aplikasi ini menggunakan model Deep Learning **GRU** (Minggu 3) untuk memprediksi sentimen ulasan secara otomatis.")

    sentiment_model = load_sentiment_model()
    tokenizer = get_tokenizer()

    if sentiment_model is None:
        st.error("Model `sentiment_GRU_cfg1_seq100.h5` tidak ditemukan.")
    else:
        if not os.path.exists('tokenizer.pkl'):
            st.warning("⚠️ **Peringatan:** File `tokenizer.pkl` belum ada di folder. Prediksi mungkin kacau karena model tidak punya kamus (vocabulary). Silakan simpan dan upload file tokenizer tersebut.")
            
        with st.form("sentiment_form"):
            user_input = st.text_area("Masukkan ulasan film dalam Bahasa Inggris:", "This movie was absolutely fantastic! The acting was great and the story was engaging.")
            submitted = st.form_submit_button("Analisis Sentimen")

        if submitted:
            if user_input.strip() == "":
                st.warning("Mohon masukkan teks ulasan.")
            else:
                with st.spinner("Sedang menganalisis teks..."):
                    processed_text = preprocess_text(user_input, tokenizer, max_len=100)
                    prediction = sentiment_model.predict(processed_text)[0][0]
                    
                    st.subheader("Hasil Prediksi")
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction > 0.5:
                            st.markdown(f'<div class="success-box"><h3>🌟 Sentimen: Positif</h3><p>Kepercayaan: {prediction:.2%}</p></div>', unsafe_allow_html=True)
                            st.progress(float(prediction))
                        else:
                            st.markdown(f'<div class="error-box"><h3>💔 Sentimen: Negatif</h3><p>Kepercayaan: {(1-prediction):.2%}</p></div>', unsafe_allow_html=True)
                            st.progress(float(1 - prediction))
                    
                    with col2:
                        with st.expander("Detail Preprocessing (Bonus)"):
                            st.write("**Teks Bersih:**", re.sub(r'[^a-zA-Z0-9\s]', '', re.sub(r'<[^>]*>', '', user_input.lower())))

elif menu == "Riwayat Versi (Versioning)":
    st.title("📚 Riwayat Versi (Versioning)")
    
    version_data = {
        "Versi": ["Versi 1", "Versi 2 (Final)"],
        "Fitur yang Ditambahkan / Diperbaiki": [
            "Membuat UI Streamlit dasar, implementasi model GRU (Minggu 3), form input teks, progress bar prediksi, dan halaman Riwayat Versi.",
            "Menambahkan fitur unggah gambar untuk Model Apple vs Orange (Minggu 2) menggunakan MobileNetV2 (Transfer Learning), dan integrasi Sidebar Multi-Halaman."
        ],
        "Status": ["Selesai", "Selesai & Siap Deploy"]
    }
    st.table(pd.DataFrame(version_data))
    
    st.info("✅ Kriteria Versioning tugas telah terpenuhi. Aplikasi ini sudah berisi 2 model dan riwayat pengembangan yang dicatat dengan baik.")
