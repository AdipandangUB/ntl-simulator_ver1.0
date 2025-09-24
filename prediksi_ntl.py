# ntl_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
import tempfile
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="NTL Analysis",
    page_icon="🌃",
    layout="wide"
)

def simulate_ntl_analysis(raster_paths, future_years, model_type, normalization):
    """Simulasi analisis NTL untuk demo"""
    try:
        # Data contoh untuk demo
        years = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
        radiances = np.array([25.1, 27.3, 29.8, 32.5, 35.2, 38.1])
        
        # Normalisasi
        if normalization == "MinMax":
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(radiances.reshape(-1, 1)).flatten()
        elif normalization == "Z-score":
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(radiances.reshape(-1, 1)).flatten()
        else:
            data_scaled = radiances
        
        # Model training
        if model_type == "Linear":
            model = LinearRegression()
        elif model_type == "Polynomial (degree 2)":
            poly = PolynomialFeatures(degree=2)
            years_poly = poly.fit_transform(years)
            model = LinearRegression()
            model.fit(years_poly, data_scaled)
        elif model_type == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "Lasso":
            model = Lasso(alpha=0.1)
        
        if model_type != "Polynomial (degree 2)":
            model.fit(years, data_scaled)
        
        # Prediksi
        future_years_arr = np.array(future_years).reshape(-1, 1)
        
        if model_type == "Polynomial (degree 2)":
            future_poly = poly.transform(future_years_arr)
            predictions = model.predict(future_poly)
        else:
            predictions = model.predict(future_years_arr)
        
        # Denormalize
        if normalization != "None":
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return years.flatten(), radiances, future_years, predictions, model
        
    except Exception as e:
        st.error(f"Error dalam simulasi: {str(e)}")
        return None

# UI Streamlit
st.title("🌃 Nighttime Lights Analysis")
st.markdown("Analisis dan prediksi nighttime lights menggunakan machine learning")

tab1, tab2 = st.tabs(["📈 Prediksi NTL", "ℹ️ Informasi"])

with tab1:
    st.header("Prediksi Nighttime Lights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Regresi",
            ["Linear", "Polynomial (degree 2)", "Ridge", "Lasso"]
        )
        
        normalization = st.selectbox(
            "Normalisasi Data",
            ["None", "MinMax", "Z-score"]
        )
    
    with col2:
        future_years_str = st.text_input(
            "Tahun Prediksi (pisahkan koma)",
            "2024,2025,2026,2027,2028"
        )
        
        # Parse tahun prediksi
        future_years = []
        for year_str in future_years_str.split(','):
            year_clean = year_str.strip()
            if year_clean.isdigit():
                future_years.append(int(year_clean))
    
    if st.button("🚀 Jalankan Analisis", type="primary"):
        if not future_years:
            st.error("Masukkan tahun prediksi yang valid")
        else:
            with st.spinner("Menjalankan analisis..."):
                result = simulate_ntl_analysis([], future_years, model_type, normalization)
                
                if result:
                    years, radiances, pred_years, predictions, model = result
                    
                    # Visualisasi
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot data historis
                    ax.scatter(years, radiances, color='blue', s=100, label='Data Historis', zorder=5)
                    ax.plot(years, radiances, 'b--', alpha=0.7)
                    
                    # Plot prediksi
                    ax.scatter(pred_years, predictions, color='red', s=100, label='Prediksi', zorder=5)
                    ax.plot(pred_years, predictions, 'r--', alpha=0.7)
                    
                    # Gabungkan untuk garis trend
                    all_years = np.concatenate([years, pred_years])
                    all_values = np.concatenate([radiances, predictions])
                    ax.plot(all_years, all_values, 'g-', alpha=0.5, linewidth=2, label='Trend')
                    
                    ax.set_xlabel('Tahun')
                    ax.set_ylabel('Radiansi NTL')
                    ax.set_title('Prediksi Nighttime Lights')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Tampilkan metrik
                    st.subheader("📊 Hasil Analisis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current = radiances[-1]
                        growth = ((predictions[0] - current) / current) * 100
                        st.metric("Radiansi Terakhir", f"{current:.2f}")
                    
                    with col2:
                        st.metric("Prediksi 2024", f"{predictions[0]:.2f}")
                    
                    with col3:
                        st.metric("Pertumbuhan", f"{growth:.1f}%")
                    
                    # Tabel prediksi
                    st.subheader("📈 Detail Prediksi")
                    pred_data = []
                    for year, pred in zip(pred_years, predictions):
                        pred_data.append({"Tahun": year, "Prediksi Radiansi": f"{pred:.2f}"})
                    
                    st.table(pred_data)

with tab2:
    st.header("Informasi Aplikasi")
    st.markdown("""
    ### Tentang Aplikasi NTL Analysis
    Aplikasi ini digunakan untuk menganalisis dan memprediksi nighttime lights menggunakan berbagai model machine learning.
    
    ### Model yang Tersedia:
    - **Linear Regression**: Model linier sederhana
    - **Polynomial Regression**: Model polinomial derajat 2
    - **Ridge Regression**: Regularisasi L2
    - **Lasso Regression**: Regularisasi L1
    
    ### Cara Penggunaan:
    1. Pilih model regresi yang diinginkan
    2. Pilih metode normalisasi data
    3. Masukkan tahun prediksi (pisahkan dengan koma)
    4. Klik "Jalankan Analisis"
    """)

# Footer
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Satellite Imagery Analysis - Ver 1.0 **")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Predictio Algorithm Creator) & Adipandang Yudono (WebGIS NTL Prediction Analytics Developer)**")
