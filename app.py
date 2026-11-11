import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Thêm src vào sys.path để import các module mô phỏng
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import spherex_sim

st.set_page_config(
    page_title="10LY2NHUEr SPHEREx Sandbox",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 10LY2NHUEr - NASA SPHEREx Sandbox")
st.caption("Trạm trung chuyển dữ liệu & giả lập quang phổ sứ mệnh NASA SPHEREx | Team 10LY2NHUEr")

# Giữ thanh trượt thời gian ban đầu
with st.sidebar:
    st.header("⚙️ Thiết lập quan sát")
    year = st.slider("Mốc thời gian quan sát:", 2024, 2028, 2025)
    survey_cycle = st.selectbox(
        "Chu kỳ khảo sát toàn bầu trời (6 tháng/lần):",
        ["Survey 1 (Tháng 1-6)", "Survey 2 (Tháng 7-12)", "Survey 3 (Năm 2)", "Survey 4 (Năm 2)"]
    )
    st.info(f"🛰️ Chế độ giả lập: **Năm {year} - {survey_cycle}**")
    
    st.markdown("---")
    st.markdown("""
    **Thông số SPHEREx:**
    - Bước sóng: $0.75 - 5.00\\ \\mu m$
    - Số kênh quang phổ: 102 kênh (LVF)
    - Kính thiên văn: $20\\text{ cm}$
    - 3 mục tiêu khoa học: *Lạm phát vũ trụ, Ánh sáng nền ngoại thiên hà (EBL), Băng đá liên sao & Nước*
    """)

# Các thẻ chức năng chính
tab_ices, tab_galaxy, tab_sky, tab_info = st.tabs([
    "🧊 Băng đá liên sao (Origin of Water)",
    "🔭 Dịch chuyển đỏ thiên hà (Cosmology)",
    "🗺️ Khảo sát toàn bầu trời (Sky Targets)",
    "ℹ️ Thông tin sứ mệnh SPHEREx"
])

# ----------------- TAB 1: INTERSTELLAR ICES -----------------
with tab_ices:
    st.subheader("🧊 Giả lập phổ hấp thụ băng đá liên sao (Interstellar Ices)")
    st.write(
        "SPHEREx đo đạc phổ hấp thụ của các phân tử băng đá ($H_2O, CO_2, CO$) "
        "bám trên hạt bụi trong các đám mây phân tử dày đặc và đĩa tiền hành tinh."
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Tham số nguồn sáng & băng đá:**")
        teff = st.slider("Nhiệt độ hiệu dụng sao nền (T_eff, K):", 3000, 7000, 4500, step=250)
        tau_h2o = st.slider("Độ sâu quang học Băng Nước (H2O @ 3.05 µm):", 0.0, 2.0, 0.85, step=0.05)
        tau_co2 = st.slider("Độ sâu quang học Băng CO2 (CO2 @ 4.27 µm):", 0.0, 1.5, 0.50, step=0.05)
        tau_co = st.slider("Độ sâu quang học Băng CO (CO @ 4.67 µm):", 0.0, 1.0, 0.25, step=0.05)
        snr = st.slider("Tỷ số tín hiệu trên nhiễu (SNR):", 10, 100, 45, step=5)
        
    with col2:
        wavelengths = spherex_sim.generate_spherex_bands()
        continuum, transmitted, tau = spherex_sim.simulate_stellar_ice_spectrum(
            wavelengths, teff=teff, tau_h2o=tau_h2o, tau_co2=tau_co2, tau_co=tau_co, snr=snr
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavelengths, continuum, label="Phổ liên tục (Continuum)", color="gray", linestyle="--", alpha=0.7)
        ax.plot(wavelengths, transmitted, label="Phổ quan sát SPHEREx", color="#00bcd4", lw=2)
        ax.scatter(wavelengths, transmitted, color="#ff4081", s=18, zorder=4, label="102 kênh đo SPHEREx")
        
        # Đánh dấu các vạch băng đá
        ax.axvspan(2.8, 3.4, color="cyan", alpha=0.15, label="H2O ice (~3.05 µm)")
        ax.axvspan(4.2, 4.35, color="orange", alpha=0.2, label="CO2 ice (~4.27 µm)")
        ax.axvspan(4.6, 4.75, color="magenta", alpha=0.15, label="CO ice (~4.67 µm)")
        
        ax.set_xlabel("Bước sóng $\\lambda\\ (\\mu m)$", fontsize=11)
        ax.set_ylabel("Thông lượng chuẩn hóa", fontsize=11)
        ax.set_title("Quang phổ mô phỏng quan sát bởi SPHEREx", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

# ----------------- TAB 2: GALAXY REDSHIFT -----------------
with tab_galaxy:
    st.subheader("🔭 Mô phỏng dịch chuyển đỏ quang phổ thiên hà (Cosmology & EBL)")
    st.write(
        "Bằng cách ghi nhận quang phổ hồng ngoại của hàng trăm triệu thiên hà trên toàn bầu trời, "
        "SPHEREx lập bản đồ 3D cấu trúc quy mô lớn và đo đạc thông số phi Gauss $f_{NL}$ của thời kỳ lạm phát vũ trụ."
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        redshift = st.slider("Độ dịch chuyển đỏ (Redshift z):", 0.1, 4.0, 1.2, step=0.1)
        galaxy_snr = st.slider("SNR Thiên hà:", 10, 80, 35, step=5)
        
        flux, detected_lines = spherex_sim.simulate_galaxy_spectrum(wavelengths, redshift=redshift, snr=galaxy_snr)
        
        st.markdown("**Các vạch phát xạ nằm trong dải quan sát SPHEREx:**")
        if detected_lines:
            for name, obs_wl in detected_lines:
                st.write(f"- Vạch **{name}**: $\\lambda_{{obs}} = {obs_wl:.3f}\\ \\mu m$")
        else:
            st.warning("Không có vạch phát xạ chính nào lọt vào dải 0.75 - 5.0 µm ở redshift này.")
            
    with col2:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(wavelengths, flux, color="#4caf50", lw=2, label=f"Phổ thiên hà (z={redshift:.2f})")
        ax2.scatter(wavelengths, flux, color="#ff9800", s=15, zorder=3)
        
        for name, obs_wl in detected_lines:
            ax2.axvline(obs_wl, color="red", linestyle="--", alpha=0.7)
            ax2.text(obs_wl, np.max(flux) * 0.9, f" {name}", color="red", fontsize=10, rotation=90)
            
        ax2.set_xlabel("Bước sóng $\\lambda\\ (\\mu m)$", fontsize=11)
        ax2.set_ylabel("Thông lượng quang phổ", fontsize=11)
        ax2.set_title(f"Quang phổ thiên hà dịch chuyển đỏ $z={redshift:.2f}$ trên 102 kênh SPHEREx", fontsize=13, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

# ----------------- TAB 3: ALL-SKY TARGETS -----------------
with tab_sky:
    st.subheader("🗺️ Danh mục mục tiêu thiên văn mẫu trên tọa độ Ngân Hà")
    targets = spherex_sim.generate_mock_targets(n=60)
    df_targets = pd.DataFrame(targets)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        # Phân loại theo màu
        yso_mask = df_targets["type"] == "YSO / Dense Core"
        ax3.scatter(df_targets[yso_mask]["l"], df_targets[yso_mask]["b"], color="#ff5722", label="YSO / Đám mây bụi đĩa sao", s=40, alpha=0.8)
        ax3.scatter(df_targets[~yso_mask]["l"], df_targets[~yso_mask]["b"], color="#2196f3", label="Thiên hà ngoại quan sát được", s=40, alpha=0.8)
        
        ax3.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax3.set_xlim(0, 360)
        ax3.set_ylim(-90, 90)
        ax3.set_xlabel("Kinh độ Ngân Hà $l$ (độ)", fontsize=11)
        ax3.set_ylabel("Vĩ độ Ngân Hà $b$ (độ)", fontsize=11)
        ax3.set_title("Bản đồ phân bố mục tiêu khảo sát (Hệ tọa độ Ngân Hà)", fontsize=13, fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right")
        st.pyplot(fig3)
        plt.close(fig3)
        
    with col2:
        st.write("**Bảng danh mục mục tiêu (Mock Catalog):**")
        st.dataframe(df_targets[["target_id", "type", "ra", "dec", "mag_k"]].head(15), use_container_width=True)

# ----------------- TAB 4: MISSION INFO -----------------
with tab_info:
    st.subheader("ℹ️ Giới thiệu Sứ mệnh NASA SPHEREx")
    st.markdown("""
    **SPHEREx** (*Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer*) 
    là sứ mệnh không gian lớp Trung bình (MIDEX) của NASA.
    
    - **Phương pháp quan sát**: Khảo sát toàn thiên cầu bằng bộ lọc biến thiên tuyến tính (*Linear Variable Filters - LVF*).
    - **Dải bước sóng**: $0.75 - 5.0\\ \\mu m$ chia thành 102 dải phổ liên tục.
    - **3 Trụ cột Khoa học cốt lõi**:
        1. **Vũ trụ học & Thời kỳ Lạm phát**: Tìm kiếm dấu vết trường lượng tử nguyên sơ qua phân bố cụm thiên hà 3D quy mô lớn ($f_{NL}$).
        2. **Lịch sử hình thành thiên hà & EBL**: Đo bức xạ nền hồng ngoại ngoại thiên hà để tìm hiểu về những ngôi sao và thiên hà đầu tiên thời kỳ Tái ion hóa.
        3. **Nguồn gốc của Nước và Phân tử sinh học**: Khảo sát sự phân bố của băng nước ($H_2O$), khí carbonic ($CO_2$), và carbon monoxide ($CO$) trong dải Ngân Hà.
    """)
