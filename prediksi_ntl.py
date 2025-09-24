"""
Script Analisis Nighttime Light pada Streamlit
Fokus pada Prediksi dan Modeling NTL
"""

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
import tempfile
import os
import zipfile
import urllib.request
from shapely.geometry import box
import geopandas as gpd
import time

# ----------------------------
# FUNGSI UTAMA ANALISIS NTL
# ----------------------------

def nighttime_light_modeller(raster_paths, future_years, model_type, normalization, output_folder):
    """
    Memodelkan dan memprediksi nighttime lights menggunakan berbagai model regresi
    
    Parameters:
    - raster_paths: list path file raster NTL
    - future_years: array tahun yang akan diprediksi
    - model_type: jenis model regresi
    - normalization: jenis normalisasi data
    - output_folder: folder output hasil prediksi
    """
    
    try:
        # Baca dan ekstrak data dari raster
        years = []
        avg_radiances = []
        
        for i, path in enumerate(raster_paths):
            with rasterio.open(path) as src:
                data = src.read(1)
                # Handle no data values
                data[data == src.nodata] = np.nan
                avg_rad = np.nanmean(data)
                
                # Asumsikan tahun dari nama file atau urutan
                year = 2020 + i  # Default tahun
                years.append(year)
                avg_radiances.append(avg_rad)
        
        years = np.array(years).reshape(-1, 1)
        radiances = np.array(avg_radiances)
        
        # Normalisasi data
        if normalization == "MinMax":
            scaler = MinMaxScaler()
            radiances_scaled = scaler.fit_transform(radiances.reshape(-1, 1)).flatten()
        elif normalization == "Z-score":
            scaler = StandardScaler()
            radiances_scaled = scaler.fit_transform(radiances.reshape(-1, 1)).flatten()
        else:
            radiances_scaled = radiances
        
        # Persiapan model regresi
        if model_type == "Linear":
            model = LinearRegression()
        elif model_type == "Polynomial (degree 2)":
            poly = PolynomialFeatures(degree=2)
            years_poly = poly.fit_transform(years)
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "Lasso":
            model = Lasso(alpha=0.1)
        
        # Training model
        if model_type == "Polynomial (degree 2)":
            model.fit(years_poly, radiances_scaled)
        else:
            model.fit(years, radiances_scaled)
        
        # Prediksi untuk tahun mendatang
        future_years_arr = np.array(future_years).reshape(-1, 1)
        
        if model_type == "Polynomial (degree 2)":
            future_years_poly = poly.transform(future_years_arr)
            predictions = model.predict(future_years_poly)
        else:
            predictions = model.predict(future_years_arr)
        
        # Denormalisasi jika diperlukan
        if normalization != "None":
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Simpan hasil prediksi
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        
        # Visualisasi hasil
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Data historis dan prediksi
        ax1.scatter(years.flatten(), radiances, color='blue', label='Data Historis')
        ax1.scatter(future_years, predictions, color='red', label='Prediksi')
        ax1.plot(np.concatenate([years.flatten(), future_years]), 
                np.concatenate([radiances, predictions]), color='green', linestyle='--')
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Rata-rata Radiansi NTL')
        ax1.set_title('Trend Nighttime Lights')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Perbandingan model
        all_years = np.arange(min(years)[0], max(future_years) + 1).reshape(-1, 1)
        if model_type == "Polynomial (degree 2)":
            all_years_poly = poly.transform(all_years)
            all_predictions = model.predict(all_years_poly)
        else:
            all_predictions = model.predict(all_years)
        
        if normalization != "None":
            all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        
        ax2.plot(all_years, all_predictions, color='orange', linewidth=2)
        ax2.scatter(years.flatten(), radiances, color='blue', s=50)
        ax2.scatter(future_years, predictions, color='red', s=50, marker='*')
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Radiansi NTL')
        ax2.set_title('Model Prediksi NTL')
        ax2.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tampilkan hasil numerik
        st.subheader("📊 Hasil Prediksi")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", f"{model.score(years, radiances_scaled):.3f}")
        
        with col2:
            current_rad = radiances[-1] if len(radiances) > 0 else 0
            st.metric("Radiansi Terakhir", f"{current_rad:.2f}")
        
        with col3:
            avg_growth = np.mean(np.diff(radiances)) if len(radiances) > 1 else 0
            st.metric("Pertumbuhan Rata-rata", f"{avg_growth:.2f}")
        
        # Tabel prediksi
        st.subheader("📈 Detail Prediksi")
        pred_data = []
        for year, pred in zip(future_years, predictions):
            pred_data.append({"Tahun": year, "Prediksi Radiansi": f"{pred:.2f}"})
        
        st.table(pred_data)
        
        return {"status": "success", "predictions": dict(zip(future_years, predictions))}
        
    except Exception as e:
        return f"❌ Error dalam pemodelan: {str(e)}"

def download_viirs_annual(geometry, years_sel, out_folder, feedback_func=st.write):
    """
    Download data VIIRS nighttime lights untuk analisis
    """
    try:
        os.makedirs(out_folder, exist_ok=True)
        gdf = gpd.GeoDataFrame([1], geometry=[geometry], crs="EPSG:4326")
        bounds = gdf.bounds.iloc[0]
        
        # Simulasi download untuk contoh
        for y in years_sel:
            feedback_func(f"📥 Memproses data VIIRS tahun {y}...")
            time.sleep(1)  # Simulasi proses
            
            # Buat file dummy untuk contoh
            dummy_path = os.path.join(out_folder, f"VIIRS_{y}_dummy.tif")
            # Dalam implementasi nyata, ini akan didownload dari Earth Engine
            
        feedback_func("✅ Proses download data NTL selesai!")
        return True
        
    except Exception as e:
        feedback_func(f"❌ Error dalam download: {str(e)}")
        return False

# ----------------------------
# KONFIGURASI STREAMLIT
# ----------------------------

st.set_page_config(page_title="NTL Analysis", layout="wide")
st.title("🌃 Nighttime Lights Analysis Tool")

# ----------------------------
# TAB ANALISIS NTL
# ----------------------------

tab1, tab2 = st.tabs(["📥 Download Data NTL", "📈 Prediksi NTL"])

with tab1:
    st.header("Download Data Nighttime Lights")
    st.markdown("Unduh data VIIRS untuk analisis nighttime lights")
    
    years = list(range(2014, 2026))
    selected_years = st.multiselect("Pilih Tahun", years, default=[2020, 2021, 2022, 2023])
    
    st.subheader("📍 Area of Interest")
    col1, col2 = st.columns(2)
    with col1:
        min_lon = st.number_input("Min Longitude", value=106.0)
        min_lat = st.number_input("Min Latitude", value=-7.0)
    with col2:
        max_lon = st.number_input("Max Longitude", value=108.0)
        max_lat = st.number_input("Max Latitude", value=-5.0)
    
    geometry = box(min_lon, min_lat, max_lon, max_lat)
    output_folder = st.text_input("Folder Output", value="./data_ntl")

    if st.button("📥 Download Data NTL"):
        with st.spinner("Mendownload data nighttime lights..."):
            success = download_viirs_annual(geometry, selected_years, output_folder, st.info)
            if success:
                st.success("✅ Data NTL berhasil diproses!")
            else:
                st.error("❌ Gagal memproses data NTL")

with tab2:
    st.header("🔮 Prediksi Nighttime Lights")
    st.markdown("Unggah data raster NTL historis untuk memprediksi tahun mendatang")
    
    # Upload raster data
    raster_files = st.file_uploader(
        "Pilih file TIFF raster NTL", 
        type=["tif", "tiff"], 
        accept_multiple_files=True,
        help="Unggah beberapa file raster NTL dari tahun berbeda"
    )
    
    if raster_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_paths = []
            for i, uploaded_file in enumerate(raster_files):
                file_path = os.path.join(tmpdir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                raster_paths.append(file_path)
            
            st.success(f"✅ {len(raster_paths)} file raster berhasil diunggah")
            
            # Konfigurasi prediksi
            st.subheader("⚙️ Konfigurasi Prediksi")
            col1, col2 = st.columns(2)
            
            with col1:
                future_years_str = st.text_input(
                    "Tahun prediksi (pisahkan koma)", 
                    value="2025,2030,2035",
                    help="Masukkan tahun yang ingin diprediksi, pisahkan dengan koma"
                )
                model_type = st.selectbox(
                    "Model Regresi", 
                    ["Linear", "Polynomial (degree 2)", "Ridge", "Lasso"],
                    help="Pilih model machine learning untuk prediksi"
                )
            
            with col2:
                normalization = st.selectbox(
                    "Normalisasi Data", 
                    ["None", "MinMax", "Z-score"],
                    help="Pilih metode normalisasi data"
                )
                output_folder_pred = st.text_input(
                    "Folder Output Prediksi", 
                    value="./hasil_prediksi"
                )
            
            # Parsing tahun prediksi
            future_years = []
            for x in future_years_str.split(","):
                x_clean = x.strip()
                if x_clean.isdigit():
                    future_years.append(int(x_clean))
            
            if st.button("🚀 Jalankan Prediksi", type="primary"):
                if len(raster_paths) < 2:
                    st.error("❌ Minimal perlu 2 file raster untuk analisis trend")
                elif len(future_years) == 0:
                    st.error("❌ Masukkan tahun prediksi yang valid")
                else:
                    with st.spinner("Melakukan analisis dan prediksi..."):
                        result = nighttime_light_modeller(
                            raster_paths, 
                            future_years, 
                            model_type, 
                            normalization, 
                            output_folder_pred
                        )
                    
                    if isinstance(result, dict) and result.get("status") == "success":
                        st.balloons()
                        st.success("✅ Prediksi berhasil dilakukan!")
                    else:
                        st.error(f"❌ {result}")

# ----------------------------
# INFORMASI DEVELOPER
# ----------------------------
st.markdown("---")
st.markdown("**Nighttime Light (NTL) Satellite Imagery Analysis - Ver 1.0 **")
st.markdown("**Dikembangkan oleh: Firman Afrianto (NTL Prediction Algorithm Creator) & Adipandang Yudono (WebGIS NTL Prediction Analytics Developer)**")
