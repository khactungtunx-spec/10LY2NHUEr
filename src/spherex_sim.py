"""
Module mô phỏng dữ liệu quang phổ SPHEREx (NASA)
Dải bước sóng: 0.75 - 5.00 micromet qua 102 kênh phổ (Linear Variable Filter - LVF).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def generate_spherex_bands():
    """
    Tạo mảng 102 bước sóng trung tâm của SPHEREx từ 0.75 đến 5.00 micrometers.
    Chia làm 6 dải detector LVF (mỗi dải 17 kênh phổ độc lập):
    - Band 1: 0.75 - 1.11 µm
    - Band 2: 1.11 - 1.64 µm
    - Band 3: 1.64 - 2.42 µm
    - Band 4: 2.42 - 3.82 µm
    - Band 5: 3.82 - 4.42 µm
    - Band 6: 4.42 - 5.00 µm
    """
    bands_edges = [
        (0.75, 1.11),
        (1.12, 1.64),
        (1.65, 2.42),
        (2.43, 3.82),
        (3.83, 4.42),
        (4.43, 5.00)
    ]
    all_bands = []
    for low, high in bands_edges:
        all_bands.append(np.linspace(low, high, 17))
    return np.concatenate(all_bands)


def simulate_stellar_ice_spectrum(wavelengths, teff=4500, tau_h2o=0.8, tau_co2=0.5, tau_co=0.2, snr=50):
    """
    Giả lập phổ hấp thụ băng đá liên sao (Interstellar Ices) của SPHEREx:
    - H2O ice: tâm ~3.05 µm, độ rộng FWHM ~0.4 µm
    - CO2 ice: tâm ~4.27 µm, độ rộng FWHM ~0.1 µm
    - CO ice: tâm ~4.67 µm, độ rộng FWHM ~0.08 µm
    """
    # Mô hình bức xạ vật đen xấp xỉ (Planck law đơn giản hóa cho continuum)
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    wl_m = wavelengths * 1e-6
    # Phổ liên tục
    continuum = (2 * h * c**2 / (wl_m**5)) / (np.exp((h * c) / (wl_m * k * teff)) - 1)
    continuum = continuum / np.max(continuum)  # chuẩn hóa

    # Các vạch hấp thụ Gauss cho băng đá
    def gaussian_tau(wl, center, fwhm, strength):
        sigma = fwhm / 2.355
        return strength * np.exp(-0.5 * ((wl - center) / sigma) ** 2)

    tau_total = (
        gaussian_tau(wavelengths, 3.05, 0.38, tau_h2o) +
        gaussian_tau(wavelengths, 4.27, 0.12, tau_co2) +
        gaussian_tau(wavelengths, 4.67, 0.08, tau_co)
    )

    transmitted = continuum * np.exp(-tau_total)

    # Thêm nhiễu Gaussian theo SNR
    if snr > 0:
        noise = np.random.normal(0, 1.0 / snr, size=len(wavelengths))
        transmitted = np.clip(transmitted + noise, 0, None)

    return continuum, transmitted, tau_total


def simulate_galaxy_spectrum(wavelengths, redshift=1.5, snr=30):
    """
    Giả lập phổ thiên hà hồng ngoại với độ dịch chuyển đỏ z:
    - Vạch H-alpha (nghỉ: 0.6563 µm) -> quan sát: 0.6563 * (1 + z)
    - Vạch Pa-alpha (nghỉ: 1.8756 µm) -> quan sát: 1.8756 * (1 + z)
    - Vạch [OIII] (nghỉ: 0.5007 µm) -> quan sát: 0.5007 * (1 + z)
    """
    # Phổ nền thiên hà dạng lũy thừa
    continuum = 0.5 * (wavelengths / 2.0) ** (-0.7)

    emission = np.zeros_like(wavelengths)

    lines = [
        {"name": "[OIII]", "rest_wl": 0.5007, "flux": 0.4, "fwhm": 0.03},
        {"name": "H-alpha", "rest_wl": 0.6563, "flux": 0.8, "fwhm": 0.04},
        {"name": "Pa-alpha", "rest_wl": 1.8756, "flux": 0.3, "fwhm": 0.05},
    ]

    detected_lines = []
    for line in lines:
        obs_wl = line["rest_wl"] * (1 + redshift)
        if wavelengths.min() <= obs_wl <= wavelengths.max():
            sigma = (line["fwhm"] * (1 + redshift)) / 2.355
            emission += line["flux"] * np.exp(-0.5 * ((wavelengths - obs_wl) / sigma) ** 2)
            detected_lines.append((line["name"], obs_wl))

    total_flux = continuum + emission

    if snr > 0:
        noise = np.random.normal(0, np.mean(continuum) / snr, size=len(wavelengths))
        total_flux = np.clip(total_flux + noise, 0, None)

    return total_flux, detected_lines


def generate_mock_targets(n=50):
    """
    Tạo danh sách tọa độ mục tiêu thiên văn mẫu (RA, Dec) và chuyển đổi sang Galactic coords.
    """
    np.random.seed(42)
    ra_deg = np.random.uniform(0, 360, n)
    dec_deg = np.random.uniform(-90, 90, n)

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    galactic = coords.galactic

    targets = []
    for i in range(n):
        targets.append({
            "target_id": f"SPH-{i+1:04d}",
            "ra": round(float(ra_deg[i]), 3),
            "dec": round(float(dec_deg[i]), 3),
            "l": round(float(galactic.l.deg[i]), 3),
            "b": round(float(galactic.b.deg[i]), 3),
            "type": "YSO / Dense Core" if abs(galactic.b.deg[i]) < 10 else "Extragalactic Galaxy",
            "mag_k": round(float(np.random.uniform(9.0, 16.5)), 2)
        })
    return targets
