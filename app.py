# -*- coding: utf-8 -*-
"""
SPHEREx Sky Tracker - MVP giao diện Streamlit (NASA Space Apps)
Cài:   pip install streamlit numpy pandas matplotlib astropy reproject scipy scikit-image astroquery
Chạy:  streamlit run app.py --theme.base dark
Dữ liệu demo (tùy chọn): data/spherex_t1_demo.fits, data/spherex_t2_demo.fits
"""
import io
import re
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# 1. CẤU HÌNH TRANG & THEME
# =============================================================================
st.set_page_config(page_title="SPHEREx Sky Tracker", page_icon="🌌", layout="wide")

# Bảng màu dùng chung cho cả CSS lẫn matplotlib
BG = "#0b1020"        # nền chính
PANEL = "#141b33"     # nền thẻ / sidebar
BORDER = "#263154"
TEXT = "#e6ebff"
MUTED = "#8f9bc4"
C_T1 = "#4cc9f0"      # xanh  = vị trí cũ (T1)
C_T2 = "#ff9f1c"      # cam   = vị trí mới (T2)
C_HIT = "#7CFC00"     # xanh lá = ứng viên di chuyển
C_VAR = "#ffd166"     # vàng  = nguồn chỉ đổi độ sáng

PIXEL_SCALE_DEFAULT = 6.2   # arcsec/pixel của SPHEREx, dùng khi header thiếu WCS
DT_FALLBACK_H = 43.2        # giờ (= 1.8 ngày): Δt giả định khi header thiếu mốc thời gian
MOCK_DT_H = 182.5 * 24      # giờ: dữ liệu giả lập mô phỏng hai lần chụp cách nhau 6 tháng

# Đường dẫn 2 file demo SPHEREx thật (đã cắt 512x512)
DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_T1 = DATA_DIR / "spherex_t1_demo.fits"
DEMO_T2 = DATA_DIR / "spherex_t2_demo.fits"

SRC_DEMO = "Dữ liệu SPHEREx thật (Có sẵn)"
SRC_MOCK = "Giả lập (Mock)"
SRC_UPLOAD = "Dữ liệu SPHEREx thật (Upload)"

BANDS = ["Band 1 (0.75 – 1.09 µm)", "Band 2 (1.10 – 1.62 µm)", "Band 3 (1.63 – 2.41 µm)",
         "Band 4 (2.42 – 3.82 µm)", "Band 5 (3.83 – 4.41 µm)", "Band 6 (4.42 – 5.00 µm)"]
BAND_EDGES = [(0.75, 1.095), (1.095, 1.625), (1.625, 2.415), (2.415, 3.825), (3.825, 4.415), (4.415, 5.0)]

LOAI_MOVE = "Di chuyển"
LOAI_NEW = "Nguồn mới / Sáng lên"
LOAI_GONE = "Mờ đi / Biến mất"
LOAI_TRAIL = "Vệt tuyến tính (nghi vệ tinh)"

# Phát hiện & ghép cặp
MIN_SHIFT_PX = 3.0    # dịch chuyển tối thiểu; nhỏ hơn = lưỡng cực của sao đứng yên
MAX_DFLUX_PCT = 35.0  # |Δ độ sáng| tối đa (%) để một cặp được coi là "Di chuyển"
PEAK_WIN = 7          # cửa sổ lọc cực đại (pixel) để khử các đỉnh kề nhau
MAX_PEAKS = 400       # giới hạn số đỉnh mỗi dấu
TOP_N = 12            # số ứng viên tối đa được vẽ trên biểu đồ
R_DEFAULT_REAL = 45   # bán kính ghép cặp mặc định cho dữ liệu thật (pixel)
R_DEFAULT_MOCK = 60   # ... và cho dữ liệu giả lập (vật thể mock dịch tới 51 px)

# Bộ lọc hình thái học trên từng đỉnh
PSF_MATCH_SIGMA = 0.35   # làm mượt T1 (pixel) để bù độ nhòe của T2 sau nội suy
PATCH_HALF = 5           # hộp 11x11 quanh đỉnh
BLOB_FRAC = 0.6          # cụm liên thông: pixel vượt 0.6 x ngưỡng phát hiện
MIN_AREA = 3             # diện tích tối thiểu (pixel) - loại gai 1 pixel
MAX_AXIS_RATIO = 2.4     # tỷ số trục tối đa của một nguồn điểm
MAX_MAJOR_AXIS = 9.0     # độ dài trục lớn tối đa (pixel)

# Khử chuỗi ứng viên thẳng hàng (vệt vệ tinh / persistence)
TRAIL_MIN_N = 4          # số ứng viên tối thiểu trên một đường thẳng
TRAIL_SPAN = 60.0        # độ trải dài tối thiểu đầu-cuối (pixel)
TRAIL_DIST = 2.5         # khoảng cách vuông góc tối đa tới đường thẳng (pixel)

# Kiểm tra bảo toàn quang thông khẩu độ cho các đỉnh đơn lẻ
AP_HALF = 2              # hộp khẩu độ 5x5
AP_K = 5.0               # nguồn "tồn tại rõ" khi F > 5σ ở cả hai thời điểm
AP_TOL = 0.20            # coi là sao tĩnh nếu |F2 - F1| / max(|F1|, |F2|) < 20%

# Đối chiếu danh mục SIMBAD
SIMBAD_TOP = 15          # số ứng viên tối đa được truy vấn
SIMBAD_RADIUS = 30.0     # bán kính tìm kiếm (arcsec)
SIMBAD_TIMEOUT = 5       # giây
SIMBAD_NONE = "Vật thể lạ / Chưa định danh (New Candidate)"
SIMBAD_OFFLINE = "Lỗi kết nối SIMBAD (Offline)"
SIMBAD_SKIPPED = "Chưa truy vấn (Ngoài top 15)"
SIMBAD_OFF = "Chưa truy vấn (đã tắt SIMBAD)"
SIMBAD_NO_LIB = "Chưa cài astroquery"
SIMBAD_NA = "Không áp dụng (tọa độ không phải bầu trời thật)"
OTYPE_LABELS = {
    "*": "Sao", "Star": "Sao", "**": "Sao đôi", "V*": "Sao biến quang", "PM*": "Sao chuyển động riêng lớn",
    "HPM*": "Sao chuyển động riêng lớn", "LP*": "Sao biến quang chu kỳ dài", "Mi*": "Sao biến quang kiểu Mira",
    "RR*": "Sao biến quang RR Lyrae", "Ce*": "Sao biến quang Cepheid", "EB*": "Sao đôi che khuất",
    "SB*": "Sao đôi quang phổ", "RG*": "Sao khổng lồ đỏ", "AB*": "Sao nhánh tiệm cận", "C*": "Sao carbon",
    "WD*": "Sao lùn trắng", "BD*": "Sao lùn nâu", "LM*": "Sao khối lượng thấp", "Y*O": "Thiên thể sao trẻ",
    "TT*": "Sao T Tauri", "Em*": "Sao vạch phát xạ", "G": "Thiên hà", "GiG": "Thiên hà trong nhóm",
    "GiC": "Thiên hà trong cụm", "GiP": "Thiên hà trong cặp", "LSB": "Thiên hà độ sáng bề mặt thấp",
    "AGN": "Nhân thiên hà hoạt động", "QSO": "Nhân thiên hà hoạt động", "Sy1": "Nhân thiên hà hoạt động",
    "Sy2": "Nhân thiên hà hoạt động", "SyG": "Nhân thiên hà hoạt động", "BLL": "Nhân thiên hà hoạt động",
    "IR": "Nguồn hồng ngoại", "NIR": "Nguồn hồng ngoại", "MIR": "Nguồn hồng ngoại", "FIR": "Nguồn hồng ngoại",
    "Rad": "Nguồn vô tuyến", "X": "Nguồn tia X", "UV": "Nguồn tử ngoại", "GlC": "Cụm sao cầu",
    "OpC": "Cụm sao mở", "Cl*": "Cụm sao", "PN": "Tinh vân hành tinh", "HII": "Vùng HII", "SNR": "Tàn dư siêu tân tinh",
}

# Căn chỉnh dưới pixel (tương quan pha)
REG_CROP = 0.10       # bỏ 10% viền mỗi cạnh khi đo
REG_UPSAMPLE = 20     # độ chính xác 1/20 = 0.05 pixel
REG_MIN, REG_MAX = 0.03, 15.0   # chỉ dịch khi độ lệch nằm trong khoảng này (pixel)

# Cân bằng quang thông & mặt nạ hình thái học (chỉ áp dụng cho dữ liệu SPHEREx thật)
BRIGHT_LO, BRIGHT_HI = 15.0, 200.0   # sao sáng chung chưa bão hòa: 15σ - 200σ trên nền
MIN_BRIGHT_PIX = 10                  # cần tối thiểu ngần này pixel sao sáng mới chỉnh tỷ lệ
SCALE_CLIP = (0.7, 1.3)              # giới hạn hệ số tỷ lệ quang thông T2/T1
EDGE_PX = 15                         # gọt biên vùng phủ sau khi chiếu lưới (pixel)
SAT_K = 1000.0                       # tâm sao bão hòa: > nền + 1000σ ...
SAT_PCTL = 99.85                     # ... và không thấp hơn bách phân vị 99.85
HALO_R = 12                          # bán kính che hào quang (pixel)
HALO_MAX_PCT = 10.0                  # diện tích che hào quang tối đa (% khung hình)

# Các cờ FLAGS được coi là pixel hỏng (theo danh sách bad_flags trong tutorial của IRSA).
# KHÔNG gồm SOURCE (bit 21, nguồn đã biết) và các cờ GHOST (bit 22-24).
BAD_FLAG_NAMES = ("TRANSIENT", "OVERFLOW", "SUR_ERROR", "PHANTOM", "REFERENCE", "NONFUNC", "DICHROIC",
                  "MISSING_DATA", "HOT", "COLD", "FULLSAMPLE", "PHANMISS", "NONLINEAR", "PERSIST", "OUTLIER")
FALLBACK_BAD_MASK = (1 << 20) - 1  # bit 0-19: dùng khi header FLAGS không có từ khóa MSKNnnnn

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(1200px 600px at 15% -10%, #1b2550 0%, {BG} 55%) fixed;
        color: {TEXT};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {{ color: {TEXT}; }}
    .block-container {{ padding-top: 2rem; }}

    /* Tiêu đề */
    .hero h1 {{
        font-size: 2.1rem; margin: 0; letter-spacing: .3px;
        background: linear-gradient(90deg, {C_T1}, #b388ff 55%, {C_T2});
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
    }}
    .hero p {{ color: {MUTED} !important; margin: .2rem 0 1rem 0; }}
    .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
        border: 1px solid {BORDER}; color: {MUTED}; margin-left: 8px; vertical-align: middle;
    }}
    .badge.mock {{ border-color: {C_VAR}; color: {C_VAR}; }}
    .badge.real {{ border-color: {C_HIT}; color: {C_HIT}; }}

    /* Thẻ số liệu */
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 12px 16px; transition: transform .15s ease, border-color .15s ease;
    }}
    [data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: {C_T1}; }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 10px 10px 0 0; padding: 8px 18px;
    }}
    .stTabs [data-baseweb="tab"] p {{ color: {MUTED}; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ background: {PANEL}; }}
    .stTabs [aria-selected="true"] p {{ color: {TEXT}; }}

    /* Chú giải màu */
    .legend span {{ margin-right: 18px; color: {MUTED}; font-size: .85rem; }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1b. MÀN HÌNH HƯỚNG DẪN (ONBOARDING) - chỉ hiện 1 lần mỗi phiên
# =============================================================================
if "tutorial_done" not in st.session_state:
    st.session_state.tutorial_done = False

if not st.session_state.tutorial_done:
    st.title("🚀 Hướng dẫn nhanh: SPHEREx Sky Tracker")
    st.markdown(
        "Chào mừng bạn đến với hệ thống phân tích dữ liệu thiên văn của **10LY2NHUEr**. "
        "Để phát hiện vật thể di chuyển, hệ thống sẽ thực hiện 3 bước:"
    )
    st.markdown("<style>[data-testid='stAlert'] *, [data-testid='stAlertContainer'] *, div[role='alert'] * { color: #0f172a !important; }</style>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**1. Nạp Dữ Liệu**\n\n"
            "Truy xuất trực tiếp từ kho dữ liệu NASA qua tọa độ, hoặc tải lên ảnh .FITS thủ công để khảo sát."
        )
    with col2:
        st.warning(
            "**2. Đồng Bộ & Lọc Nhiễu**\n\n"
            "Hệ thống tự động căn chỉnh tọa độ WCS và trừ ảnh. "
            "Bạn có thể kéo thanh trượt Sigma để lọc nhiễu hạt."
        )
    with col3:
        st.success(
            "**3. Trích Xuất**\n\n"
            "Nhận diện quỹ đạo nội suy và tải xuống bảng tọa độ RA/Dec của các ứng viên."
        )

    st.write("")
    try:
        bat_dau = st.button("Bắt đầu 🌌", type="primary", width="stretch", key="btn_start")
    except TypeError:  # Streamlit bản cũ chưa có tham số width
        bat_dau = st.button("Bắt đầu 🌌", type="primary", use_container_width=True, key="btn_start")

    if bat_dau:
        st.session_state.tutorial_done = True
        st.rerun()

    st.stop()  # Chặn toàn bộ phần còn lại của app.py khi chưa bấm "Bắt đầu"

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": BORDER, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": MUTED, "font.size": 8,
})
# Bảng màu ảnh trừ: xanh (mất đi) -> đen (không đổi) -> cam (xuất hiện)
CMAP_DIFF = LinearSegmentedColormap.from_list("diff", [C_T1, "#16306b", "#05070f", "#7a3d00", C_T2])


# =============================================================================
# 2. DỮ LIỆU: GIẢ LẬP & ĐỌC FITS SPHEREx
# =============================================================================
def _add_source(img, y, x, peak, fwhm_sigma=1.3):
    """Cộng một nguồn điểm dạng Gauss (giống PSF) vào ảnh."""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img += peak * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * fwhm_sigma ** 2))


@st.cache_data(show_spinner=False)
def generate_mock_data(seed=42, n_movers=3, size=128):
    """Bầu trời giả lập: sao nền đứng yên + vật thể di chuyển + 1 sao biến quang.
    Nhiễu của T1 và T2 độc lập với nhau (như 2 lần chụp thật)."""
    rng = np.random.default_rng(seed)
    sky = np.zeros((size, size))

    # Sao nền đứng yên
    for _ in range(25):
        y, x = rng.uniform(6, size - 6, 2)
        _add_source(sky, y, x, rng.uniform(15, 120))

    img_t1, img_t2 = sky.copy(), sky.copy()

    # Vật thể di chuyển: (y1, x1) -> (y2, x2), độ sáng đỉnh. Cái thứ 3 cố ý mờ.
    movers = [((64, 38), (76, 88), 80.0), ((30, 95), (22, 70), 45.0), ((100, 60), (108, 48), 18.0)]
    for (p1, p2, peak) in movers[:n_movers]:
        _add_source(img_t1, *p1, peak)
        _add_source(img_t2, *p2, peak)

    # Một sao biến quang: không di chuyển, chỉ sáng lên ở T2 (để kiểm tra phân loại)
    _add_source(img_t1, 100, 105, 60.0)
    _add_source(img_t2, 100, 105, 90.0)

    img_t1 += rng.normal(10, 2, sky.shape)
    img_t2 += rng.normal(10, 2, sky.shape)
    return img_t1, img_t2


def clipped_stats(values):
    """Trung vị và độ lệch chuẩn sau khi cắt 3σ (5 vòng lặp).
    An toàn với mảng rỗng / toàn NaN: khi đó trả về (0.0, 1.0)."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    med, std = None, None
    try:
        from astropy.stats import sigma_clipped_stats
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)  # (mean, median, std)
        med, std = float(med), float(std)
    except ImportError:
        for _ in range(5):
            m, s = np.median(v), v.std()
            keep = np.abs(v - m) <= 3.0 * s
            if s == 0 or keep.all():
                break
            v = v[keep]
        med, std = float(np.median(v)), float(v.std())
    if not np.isfinite(med):
        med = 0.0
    if not np.isfinite(std) or std <= 0:
        std = 1.4826 * float(np.median(np.abs(v - np.median(v))))  # thử MAD
        if not np.isfinite(std) or std <= 0:
            std = 1.0
    return med, std


def _celestial_wcs(header_str):
    """Dựng WCS 2 chiều từ chuỗi header; trả None nếu không có hệ tọa độ thiên cầu hợp lệ."""
    if not header_str:
        return None
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(fits.Header.fromstring(header_str), naxis=2)
        return w if w.has_celestial else None
    except Exception:
        return None


def _obs_mjd(headers):
    """Mốc thời gian quan sát (MJD) từ header; None nếu không có."""
    for hdr in headers:
        for k in ("MJD-OBS", "MJD-AVG", "MJD-BEG"):
            try:
                v = float(hdr.get(k))
                if np.isfinite(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        for k in ("DATE-OBS", "DATE-AVG", "DATE-BEG"):
            v = hdr.get(k)
            if v:
                try:
                    from astropy.time import Time
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return float(Time(str(v).strip()).mjd)
                except Exception:
                    pass
    return None


def _band_from_headers(headers):
    """Chỉ số band (0-5) và mô tả nguồn, đọc từ FILTER / BAND / DETECTOR / WAVELEN."""
    for hdr in headers:
        for k in ("FILTER", "BAND", "DETECTOR"):
            v = hdr.get(k)
            if v is not None:
                m = re.search(r"[1-6]", str(v))
                if m:
                    return int(m.group()) - 1, f"{k} = {v}"
        v = hdr.get("WAVELEN")
        try:
            w = float(v)
            for i, (lo, hi) in enumerate(BAND_EDGES):
                if lo <= w <= hi:
                    return i, f"WAVELEN = {w:g} µm"
        except (TypeError, ValueError):
            pass
    return None, None


@st.cache_data(show_spinner=False)
def load_fits(raw):
    """Đọc 1 file FITS (bytes): tự dò HDU khoa học có WCS, các HDU phụ (ZODI, FLAGS) và metadata."""
    from astropy.io import fits  # import trễ: chỉ cần khi dùng dữ liệu thật

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(raw)) as hdul:
            def co_anh(h):
                return getattr(h, "data", None) is not None and np.ndim(h.data) >= 2

            by_name = {h.name.upper(): h for h in hdul}
            ten_phu = ("ZODI", "BACKGROUND", "FLAGS", "DQ", "VARIANCE", "PSF")
            # Ưu tiên HDU tên IMAGE/SCI; sau đó HDU đầu tiên có ảnh + WCS thiên cầu; cuối cùng ảnh bất kỳ
            sci = next((by_name[n] for n in ("IMAGE", "SCI") if n in by_name and co_anh(by_name[n])), None)
            if sci is None:
                sci = next((h for h in hdul if co_anh(h) and h.name.upper() not in ten_phu
                            and "CTYPE1" in h.header and "CRVAL1" in h.header), None)
            if sci is None:
                sci = next((h for h in hdul if co_anh(h) and h.name.upper() not in ten_phu), None)
            if sci is None:
                raise ValueError("Không tìm thấy ảnh (>= 2 chiều) trong file FITS.")

            data = np.array(sci.data)
            while data.ndim > 3:
                data = data[0]
            out = {"data": data, "n_planes": data.shape[0] if data.ndim == 3 else 1,
                   "zodi": None, "flags": None, "flag_bits": {}, "header": sci.header.tostring()}

            def phu(names):
                for n in names:
                    h = by_name.get(n)
                    if h is not None and h is not sci and co_anh(h) and np.shape(h.data)[-2:] == data.shape[-2:]:
                        return h
                return None

            hz, hf = phu(("ZODI", "BACKGROUND")), phu(("FLAGS", "DQ"))
            if hz is not None:
                out["zodi"] = np.array(hz.data)
            if hf is not None:
                out["flags"] = np.array(hf.data)
                # Header FLAGS của SPHEREx ghi tên cờ của bit n trong từ khóa MSKNnnnn
                for k, v in hf.header.items():
                    if k.startswith("MSKN") and k[4:].isdigit():
                        out["flag_bits"][str(v).strip().upper()] = int(k[4:])

            headers = [sci.header, hdul[0].header]
            out["mjd"] = _obs_mjd(headers)
            out["band_idx"], out["band_src"] = _band_from_headers(headers)

    # Độ phân giải góc đọc trực tiếp từ ma trận WCS
    out["pixel_scale"], out["pixel_scale_src"] = PIXEL_SCALE_DEFAULT, "mặc định"
    w = _celestial_wcs(out["header"])
    if w is not None:
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            ps = float(np.mean(proj_plane_pixel_scales(w)) * 3600.0)
            if np.isfinite(ps) and 0.01 < ps < 3600:
                out["pixel_scale"], out["pixel_scale_src"] = ps, "WCS"
        except Exception:
            pass
    return out


def _plane(arr, idx):
    """Lấy lát cắt 2D từ ảnh 2D hoặc khối 3D (n_bands, H, W)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[min(max(int(idx), 0), arr.shape[0] - 1)]
    return arr


def bad_pixel_mask(flags, flag_bits):
    """Mặt nạ pixel hỏng theo phép bitwise AND, bỏ qua bit SOURCE.
    Trả về (mảng bool, mô tả nguồn định nghĩa bit)."""
    bits = [flag_bits[n] for n in BAD_FLAG_NAMES if n in flag_bits]
    if bits:
        mask_value, nguon = sum(1 << b for b in set(bits)), "theo header MSKN"
    else:
        mask_value, nguon = FALLBACK_BAD_MASK, "bit 0–19 mặc định"
    f = np.nan_to_num(np.asarray(flags, dtype=float)).astype(np.int64)
    return (f & mask_value) != 0, nguon


def _corr(a, b, m):
    """Hệ số tương quan của a và b trên mặt nạ m (dùng để kiểm tra bước tinh chỉnh có cải thiện không)."""
    if m.sum() < 50:
        return -1.0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else -1.0


@st.cache_data(show_spinner="Đang tiền xử lý và căn chỉnh WCS 2 bước…")
def prepare_real(raw1, raw2, tru_zodi, loc_flags, cube_index=0):
    """Giai đoạn 1: tiền xử lý Level 2 -> chiếu lưới WCS (2A) -> tinh chỉnh dưới pixel (2B)
    -> cân bằng quang thông -> gọt biên -> che hào quang sao bão hòa.
    Trả về: t1 - nền, t2 đã căn chỉnh - nền, valid_mask, info, header T1."""
    a, b = load_fits(raw1), load_fits(raw2)
    info = {"zodi": False, "flags_pct": None, "flags_src": None, "wcs": "no_wcs",
            "reg": "none", "dx": 0.0, "dy": 0.0, "shift_arcsec": 0.0,
            "scale": 1.0, "n_bright": 0, "edge": False, "n_sat": 0, "halo_pct": 0.0,
            "pixel_scale": a["pixel_scale"], "pixel_scale_src": a["pixel_scale_src"]}

    # Khoảng thời gian giữa hai lần chụp
    if a["mjd"] is not None and b["mjd"] is not None and abs(b["mjd"] - a["mjd"]) > 0:
        info["dt_hours"], info["dt_src"] = abs(b["mjd"] - a["mjd"]) * 24.0, "header"
    else:
        info["dt_hours"], info["dt_src"] = DT_FALLBACK_H, "fallback"

    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ---- 1. Tiền xử lý Level 2: ZODI và FLAGS, TRƯỚC khi chiếu lưới ----
        imgs, pcts = [], []
        for d in (a, b):
            img = np.array(_plane(d["data"], cube_index), dtype=float)
            img[~np.isfinite(img)] = np.nan
            if tru_zodi and d["zodi"] is not None:
                img = img - np.asarray(_plane(d["zodi"], cube_index), dtype=float)
            if loc_flags and d["flags"] is not None:
                bad, info["flags_src"] = bad_pixel_mask(_plane(d["flags"], cube_index), d["flag_bits"])
                img[bad] = np.nan      # che pixel hỏng / tia vũ trụ để không bị nội suy loang ra xung quanh
                pcts.append(float(bad.mean() * 100))
            imgs.append(img)
        t1, t2 = imgs
        info["zodi"] = bool(tru_zodi and a["zodi"] is not None and b["zodi"] is not None)
        if len(pcts) == 2:
            info["flags_pct"] = tuple(pcts)

        # ---- 2A. Chiếu lưới thiên cầu: T2 -> hệ quy chiếu (RA, Dec) của T1 ----
        wcs1, wcs2 = _celestial_wcs(a["header"]), _celestial_wcs(b["header"])
        footprint = phu = None
        if wcs1 is not None and wcs2 is not None:
            try:
                from reproject import reproject_interp
                t2_goc = t2
                t2, footprint = reproject_interp((t2_goc, wcs2), wcs1, shape_out=t1.shape, order="bilinear")
                # Vùng phủ hình học của T2 trên lưới T1 (không bị khoét lỗ bởi các pixel NaN)
                _, phu = reproject_interp((np.ones(t2_goc.shape), wcs2), wcs1, shape_out=t1.shape,
                                          order="bilinear")
                info["wcs"] = "ok"
            except ImportError:
                info["wcs"] = "no_reproject"
        if footprint is None:
            if t1.shape != t2.shape:
                raise ValueError(f"Không đồng bộ được WCS và kích thước hai ảnh khác nhau: {t1.shape} và {t2.shape}.")
            footprint = phu = np.ones(t1.shape)

        phu = np.nan_to_num(phu) > 0.5
        ok1 = np.isfinite(t1)
        ok2 = (np.nan_to_num(footprint) > 0.5) & np.isfinite(t2)   # pixel hợp lệ của riêng T2
        valid = ok1 & ok2
        if valid.sum() < 100:
            raise ValueError("Gần như không còn pixel hợp lệ chung giữa T1 và T2 "
                             "(hai ảnh không chồng lấn, hoặc mặt nạ FLAGS che gần hết - thử tắt lọc FLAGS).")

        med1, s1 = clipped_stats(t1[valid])
        med2, s2 = clipped_stats(t2[valid])

        # ---- 2B. Tinh chỉnh lệch tâm dưới mức pixel bằng tương quan pha ----
        H, W = t1.shape
        loi = (slice(int(REG_CROP * H), int((1 - REG_CROP) * H)), slice(int(REG_CROP * W), int((1 - REG_CROP) * W)))

        def nen_dai_dong(img, med, sig):
            """Điền NaN bằng nền rồi nén dải động để sao quá sáng không lấn át phổ FFT."""
            return np.arcsinh((np.where(np.isfinite(img), img, med) - med) / max(sig, 1e-8))

        if ndimage is None:
            info["reg"] = "no_scipy"
        else:
            try:
                from skimage.registration import phase_cross_correlation
                ref_norm, mov_norm = nen_dai_dong(t1, med1, s1), nen_dai_dong(t2, med2, s2)
                try:
                    shift = phase_cross_correlation(ref_norm[loi], mov_norm[loi], upsample_factor=REG_UPSAMPLE,
                                                    normalization=None)[0]
                except TypeError:  # scikit-image cũ chưa có tham số normalization
                    shift = phase_cross_correlation(ref_norm[loi], mov_norm[loi], upsample_factor=REG_UPSAMPLE)[0]
                shift_y, shift_x = float(shift[0]), float(shift[1])
                do_lech = float(np.hypot(shift_y, shift_x))
                info["dx"], info["dy"] = shift_x, shift_y
                info["shift_arcsec"] = do_lech * info["pixel_scale"]
                if not np.isfinite(do_lech):
                    info["reg"], info["dx"], info["dy"], info["shift_arcsec"] = "failed", 0.0, 0.0, 0.0
                elif do_lech < REG_MIN:
                    info["reg"] = "small"
                elif do_lech > REG_MAX:
                    info["reg"] = "too_big"
                else:
                    # Dịch ảnh (bậc 3) sau khi điền nền vào ô NaN; dịch riêng mặt nạ bằng bậc 0
                    t2_dich = ndimage.shift(np.where(np.isfinite(t2), t2, med2), shift=(shift_y, shift_x),
                                            order=3, mode="constant", cval=med2)
                    ok2_dich = ndimage.shift(ok2.astype(float), shift=(shift_y, shift_x),
                                             order=0, mode="constant", cval=0.0) > 0.5
                    phu_dich = ndimage.shift(phu.astype(float), shift=(shift_y, shift_x),
                                             order=0, mode="constant", cval=0.0) > 0.5
                    t2_dich[~ok2_dich] = np.nan
                    # Chỉ nhận phép dịch nếu nó thật sự làm hai ảnh khớp nhau hơn
                    m_truoc = np.zeros_like(valid); m_truoc[loi] = valid[loi]
                    m_sau = np.zeros_like(valid); m_sau[loi] = (ok1 & ok2_dich)[loi]
                    c_truoc = _corr(ref_norm, mov_norm, m_truoc)
                    c_sau = _corr(ref_norm, nen_dai_dong(t2_dich, med2, s2), m_sau)
                    if c_sau >= c_truoc:
                        t2, ok2, phu = t2_dich, ok2_dich, phu & phu_dich
                        valid = ok1 & ok2
                        med2, s2 = clipped_stats(t2[valid])
                        info["reg"] = "applied"
                    else:
                        info["reg"] = "rejected"
            except ImportError:
                info["reg"] = "no_skimage"

        if valid.sum() < 100:
            raise ValueError("Gần như không còn pixel hợp lệ sau bước căn chỉnh.")

        # ---- 3a. Cân bằng tỷ lệ quang thông trên các sao sáng chung, chưa bão hòa ----
        d1, d2 = t1 - med1, t2 - med2
        bright_mask = (valid & (d1 > BRIGHT_LO * s1) & (d1 < BRIGHT_HI * s1)
                       & (d2 > BRIGHT_LO * s2) & (d2 < BRIGHT_HI * s2))
        ratio = np.nan
        if bright_mask.sum() >= MIN_BRIGHT_PIX:
            if ndimage is not None:
                # Tỷ số theo TỔNG thông lượng từng sao (vùng sáng nở thêm 2 px), rồi lấy trung vị giữa các sao.
                # Tỷ số từng pixel bị lệch thấp vì phép nội suy làm nhòe đỉnh sao ở T2.
                nhan, n_sao = ndimage.label(ndimage.binary_dilation(bright_mask, iterations=2))
                idx = np.arange(1, n_sao + 1)
                f1 = np.atleast_1d(ndimage.sum(np.where(valid, d1, 0.0), nhan, idx))
                f2 = np.atleast_1d(ndimage.sum(np.where(valid, d2, 0.0), nhan, idx))
                ok = f1 > 0
                info["n_bright"] = int(ok.sum())
                if ok.any():
                    ratio = float(np.median(f2[ok] / f1[ok]))
            else:
                info["n_bright"] = int(bright_mask.sum())
                ratio = float(np.nanmedian(d2[bright_mask] / d1[bright_mask]))
        if np.isfinite(ratio) and ratio > 0:
            info["scale"] = float(np.clip(ratio, *SCALE_CLIP))
            info["scale_raw"] = ratio
            t2 = med2 + (t2 - med2) / info["scale"]
            s2 = s2 / info["scale"]

        if ndimage is None:
            info["edge"] = None  # thiếu scipy: bỏ qua gọt biên và che hào quang
        else:
            # ---- 3b. Gọt biên nội suy: co VÙNG PHỦ vào trong EDGE_PX pixel (tính cả mép ảnh T1). ----
            # Không co trực tiếp valid_mask: mỗi pixel bị cờ FLAGS là một "lỗ", phép co sẽ khoét rộng
            # từng lỗ thêm 15 px và xóa gần hết ảnh.
            valid = valid & ndimage.binary_erosion(phu, iterations=EDGE_PX, border_value=0)
            info["edge"] = True

            # ---- 3c. Che hào quang quanh tâm sao cực sáng / bão hòa ----
            if valid.sum() >= 100:
                sat_1 = max(med1 + SAT_K * s1, float(np.nanpercentile(t1[valid], SAT_PCTL)))
                sat_2 = max(med2 + SAT_K * s2, float(np.nanpercentile(t2[valid], SAT_PCTL)))
                yy, xx = np.ogrid[-HALO_R:HALO_R + 1, -HALO_R:HALO_R + 1]
                disk = (xx ** 2 + yy ** 2) <= HALO_R ** 2
                he_so = 1.0
                for _ in range(8):  # nâng ngưỡng cho tới khi vùng che nhỏ hơn HALO_MAX_PCT % khung hình
                    tam = valid & ((t1 > med1 + he_so * (sat_1 - med1)) | (t2 > med2 + he_so * (sat_2 - med2)))
                    halo = ndimage.binary_dilation(tam, structure=disk) if tam.any() else np.zeros_like(valid)
                    info["halo_pct"] = float(halo.mean() * 100)
                    if info["halo_pct"] < HALO_MAX_PCT:
                        break
                    he_so *= 1.5
                info["n_sat"] = int(ndimage.label(tam, structure=np.ones((3, 3)))[1]) if tam.any() else 0
                valid = valid & ~halo

        if valid.sum() < 100:
            raise ValueError("Gần như không còn pixel hợp lệ sau khi gọt biên và che hào quang sao sáng.")
        info["valid_pct"] = float(valid.mean() * 100)

        # ---- 3d. Khớp PSF cực nhẹ cho T1: bù độ suy giảm đỉnh của T2 sau khi nội suy ----
        info["psf_match"] = False
        if ndimage is not None and PSF_MATCH_SIGMA > 0 and info["wcs"] == "ok":
            t1_filled = np.where(ok1, t1, med1)                 # điền nền để NaN không loang khi lọc
            t1 = ndimage.gaussian_filter(t1_filled, sigma=PSF_MATCH_SIGMA)
            t1[~ok1] = np.nan                                   # trả lại NaN cho mọi pixel vốn không hợp lệ
            info["psf_match"] = True
        return t1 - med1, t2 - med2, valid, info, (a["header"] if wcs1 is not None else None)


# =============================================================================
# 3. THUẬT TOÁN PHÁT HIỆN (chạy thật trên ảnh trừ, không hard-code)
# =============================================================================
def _max_filter(v, size):
    """Lọc cực đại cửa sổ size x size (scipy nếu có, nếu không thì numpy thuần)."""
    try:
        from scipy.ndimage import maximum_filter
        return maximum_filter(v, size=size, mode="constant", cval=-np.inf)
    except ImportError:
        r = size // 2
        p = np.pad(v, r, mode="constant", constant_values=-np.inf)
        return np.lib.stride_tricks.sliding_window_view(p, (size, size)).max(axis=(2, 3))


def _cum_lien_thong(mask, cy, cx):
    """Cụm pixel liên thông (8 hướng) chứa pixel (cy, cx) trong một hộp nhỏ."""
    blob = np.zeros_like(mask)
    if not mask[cy, cx]:
        return blob
    h, w = mask.shape
    stack = [(cy, cx)]
    blob[cy, cx] = True
    while stack:
        y, x = stack.pop()
        for ny in range(max(y - 1, 0), min(y + 2, h)):
            for nx in range(max(x - 1, 0), min(x + 2, w)):
                if mask[ny, nx] and not blob[ny, nx]:
                    blob[ny, nx] = True
                    stack.append((ny, nx))
    return blob


def _find_peaks(v, nguong):
    """Cực trị cục bộ của v vượt ngưỡng (cửa sổ PEAK_WIN để khử đỉnh kề nhau), lọc theo hình thái:
    diện tích cụm >= MIN_AREA và dạng tròn như PSF. Tâm lấy theo trọng lượng quang thông hộp 3x3.
    Trả về: danh sách đỉnh, số gai bị loại (diện tích nhỏ), số đốm dẹt / kéo dài bị loại."""
    la_dinh = (v == _max_filter(v, PEAK_WIN)) & (v > nguong)
    if la_dinh.sum() > 50_000:
        raise ValueError("Quá nhiều đỉnh vượt ngưỡng - hai ảnh có thể chưa được căn chỉnh, hãy tăng ngưỡng σ.")
    h, w = v.shape
    peaks, n_gai, n_det = [], 0, 0
    for y0, x0 in np.argwhere(la_dinh):
        # Hộp 11x11 quanh đỉnh; chỉ xét phần cùng dấu với đỉnh (v đã được đổi dấu cho đỉnh âm)
        pa, pb = max(y0 - PATCH_HALF, 0), min(y0 + PATCH_HALF + 1, h)
        qa, qb = max(x0 - PATCH_HALF, 0), min(x0 + PATCH_HALF + 1, w)
        patch = v[pa:pb, qa:qb]
        blob = _cum_lien_thong(patch > BLOB_FRAC * nguong, y0 - pa, x0 - qa)
        if blob.sum() < MIN_AREA:
            n_gai += 1
            continue
        by, bx = np.nonzero(blob)
        wgt = np.maximum(patch[blob], 0.0)
        sw = float(wgt.sum())
        if sw <= 0:
            n_gai += 1
            continue
        xc, yc = float((wgt * bx).sum() / sw), float((wgt * by).sum() / sw)
        # Mô-men bậc 2 theo trọng số cường độ; cộng 1/12 là mô-men của chính một pixel vuông
        mu20 = float((wgt * (bx - xc) ** 2).sum() / sw) + 1.0 / 12.0
        mu02 = float((wgt * (by - yc) ** 2).sum() / sw) + 1.0 / 12.0
        mu11 = float((wgt * (bx - xc) * (by - yc)).sum() / sw)
        common = float(np.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0)))
        lambda_max = max((mu20 + mu02) / 2.0 + common, 1e-6)
        lambda_min = max((mu20 + mu02) / 2.0 - common, 1e-6)
        axis_ratio = float(np.sqrt(lambda_max / lambda_min))
        major_axis = 4.0 * float(np.sqrt(lambda_max))
        if axis_ratio > MAX_AXIS_RATIO or major_axis > MAX_MAJOR_AXIS:
            n_det += 1  # vệt vệ tinh hoặc gai nhiễu xạ
            continue

        ya, yb, xa, xb = max(y0 - 1, 0), min(y0 + 2, h), max(x0 - 1, 0), min(x0 + 2, w)
        hop = np.clip(v[ya:yb, xa:xb], 0.0, None)
        tong = float(hop.sum())
        if tong <= 0:
            continue
        ys, xs = np.mgrid[ya:yb, xa:xb]
        peaks.append({"y": float((ys * hop).sum() / tong), "x": float((xs * hop).sum() / tong),
                      "flux": tong, "peak": float(v[y0, x0]), "iy": int(y0), "ix": int(x0)})
    peaks.sort(key=lambda p: -p["peak"])
    return peaks[:MAX_PEAKS], n_gai, n_det


def find_trails(points):
    """Đánh dấu các điểm nằm trên một chuỗi thẳng hàng: >= TRAIL_MIN_N điểm cách đường thẳng
    < TRAIL_DIST px và trải dài > TRAIL_SPAN px. Đường thẳng khớp bằng SVD nên đúng với mọi góc nghiêng."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(P)
    flagged = np.zeros(n, dtype=bool)
    if n < TRAIL_MIN_N:
        return flagged
    n_seed = min(n, 200)  # giới hạn số cặp điểm dùng làm giả thuyết đường thẳng
    for i in range(n_seed - 1):
        for j in range(i + 1, n_seed):
            dvec = P[j] - P[i]
            dai = float(np.hypot(dvec[0], dvec[1]))
            if dai < 1e-6:
                continue
            phap_tuyen = np.array([-dvec[1], dvec[0]]) / dai
            gan = np.abs((P - P[i]) @ phap_tuyen) < TRAIL_DIST
            if gan.sum() < TRAIL_MIN_N:
                continue
            tam = P[gan].mean(axis=0)
            _, _, vt = np.linalg.svd(P[gan] - tam, full_matrices=False)
            huong, phap_tuyen = vt[0], vt[-1]       # vector kỳ dị nhỏ nhất = pháp tuyến đường thẳng
            thuoc = np.abs((P - tam) @ phap_tuyen) < TRAIL_DIST
            if thuoc.sum() < TRAIL_MIN_N:
                continue
            chieu = (P[thuoc] - tam) @ huong
            if float(chieu.max() - chieu.min()) > TRAIL_SPAN:
                flagged |= thuoc
    return flagged


@st.cache_data(show_spinner=False)
def detect(img_t1, img_t2, k_sigma, r_max, valid=None, min_shift_px=MIN_SHIFT_PX):
    """Trừ ảnh (chỉ trên vùng hợp lệ) -> đo nhiễu (cắt 3σ) -> dò cực trị cục bộ + lọc hình thái
    -> loại lưỡng cực sao tĩnh -> ghép cặp đỉnh âm (T1) với đỉnh dương (T2) -> khử chuỗi thẳng hàng
    -> kiểm tra bảo toàn quang thông khẩu độ cho các đỉnh đơn lẻ.
    Trả về: ảnh trừ, sigma, danh sách ứng viên, thống kê các bộ lọc."""
    stats = {"dipole": 0, "spike": 0, "elong": 0, "trail": 0, "veto": 0}
    t1 = np.asarray(img_t1, dtype=float)
    t2 = np.asarray(img_t2, dtype=float)
    with np.errstate(all="ignore"):
        diff = t2 - t1
    if valid is None:
        valid = np.isfinite(diff)
    valid = valid & np.isfinite(diff)
    diff = np.where(valid, diff, np.nan)
    if not valid.any():
        return diff, 1.0, [], stats

    diff_med, diff_sigma = clipped_stats(diff[valid])
    d = np.where(valid, diff - diff_med, 0.0)  # pixel không hợp lệ = 0 nên không bao giờ vượt ngưỡng

    neg, gai_n, det_n = _find_peaks(-d, k_sigma * diff_sigma)  # vật thể có ở T1, mất ở T2
    pos, gai_p, det_p = _find_peaks(d, k_sigma * diff_sigma)   # vật thể mới xuất hiện ở T2
    stats["spike"], stats["elong"] = gai_n + gai_p, det_n + det_p

    # Mọi cặp âm-dương trong bán kính ghép cặp, sắp theo khoảng cách tăng dần (gần nhất ghép trước)
    cap = sorted(
        (float(np.hypot(p["x"] - n["x"], p["y"] - n["y"])), i, j)
        for i, n in enumerate(neg) for j, p in enumerate(pos)
        if abs(p["x"] - n["x"]) <= r_max and abs(p["y"] - n["y"]) <= r_max
    )
    used_n, used_p, rows = set(), set(), []

    # Bước 1 - Lưỡng cực: đỉnh âm và đỉnh dương cách nhau dưới min_shift_px là dư ảnh của một
    # ngôi sao đứng yên (sai số căn chỉnh dưới pixel) -> bỏ cả hai, không coi là ứng viên.
    for dist, i, j in cap:
        if dist >= min_shift_px:
            break
        if i not in used_n and j not in used_p:
            used_n.add(i); used_p.add(j)
            stats["dipole"] += 1

    # Bước 2 - Ghép mỗi đỉnh âm với đỉnh dương gần nhất chưa ghép, thỏa điều kiện bảo toàn độ sáng
    for dist, i, j in cap:
        if dist < min_shift_px or dist > r_max or i in used_n or j in used_p:
            continue
        flux1, flux2 = neg[i]["flux"], pos[j]["flux"]
        delta_flux_pct = (flux2 - abs(flux1)) / max(abs(flux1), 1e-8) * 100.0
        if abs(delta_flux_pct) > MAX_DFLUX_PCT:
            continue
        used_n.add(i); used_p.add(j)
        n, p = neg[i], pos[j]
        rows.append({"loai": LOAI_MOVE, "x1": n["x"], "y1": n["y"], "x2": p["x"], "y2": p["y"],
                     "shift_px": dist, "dflux": delta_flux_pct,
                     "snr": min(n["peak"], p["peak"]) / diff_sigma})

    # Bước 3 - Chuỗi thẳng hàng: >= 4 cặp "Di chuyển" có trung điểm cùng nằm trên một đường thẳng
    # là vệt nhiễu tuyến tính (vệ tinh / persistence), không phải nhiều thiên thể độc lập.
    if len(rows) >= TRAIL_MIN_N:
        rows.sort(key=lambda r: -r["snr"])
        trung_diem = [((r["x1"] + r["x2"]) / 2.0, (r["y1"] + r["y2"]) / 2.0) for r in rows]
        for r, la_vet in zip(rows, find_trails(trung_diem)):
            if la_vet:
                r["loai"] = LOAI_TRAIL
                stats["trail"] += 1

    # Bước 4 - Đỉnh đơn lẻ: bỏ nếu nguồn có mặt rõ ở cả T1 và T2 với tổng quang thông khẩu độ 5x5
    # gần như không đổi (sao tĩnh chỉ dao động nhẹ về hình dạng PSF).
    med1, s1 = clipped_stats(t1[valid])
    med2, s2 = clipped_stats(t2[valid])
    h, w = d.shape

    def la_sao_tinh(pk):
        ya, yb = max(pk["iy"] - AP_HALF, 0), min(pk["iy"] + AP_HALF + 1, h)
        xa, xb = max(pk["ix"] - AP_HALF, 0), min(pk["ix"] + AP_HALF + 1, w)
        b1, b2 = t1[ya:yb, xa:xb] - med1, t2[ya:yb, xa:xb] - med2
        m = np.isfinite(b1) & np.isfinite(b2)
        if m.sum() < 9:
            return False
        f1, f2 = float(b1[m].sum()), float(b2[m].sum())
        return (f1 > AP_K * s1 and f2 > AP_K * s2
                and abs(f2 - f1) / max(abs(f1), abs(f2), 1e-8) < AP_TOL)

    for j, p in enumerate(pos):
        if j not in used_p:
            if la_sao_tinh(p):
                stats["veto"] += 1
                continue
            rows.append({"loai": LOAI_NEW, "x1": np.nan, "y1": np.nan, "x2": p["x"], "y2": p["y"],
                         "shift_px": 0.0, "dflux": np.nan, "snr": p["peak"] / diff_sigma})
    for i, n in enumerate(neg):
        if i not in used_n:
            if la_sao_tinh(n):
                stats["veto"] += 1
                continue
            rows.append({"loai": LOAI_GONE, "x1": n["x"], "y1": n["y"], "x2": np.nan, "y2": np.nan,
                         "shift_px": 0.0, "dflux": np.nan, "snr": n["peak"] / diff_sigma})
    rows.sort(key=lambda r: -r["snr"])
    return diff, float(diff_sigma), rows, stats


# =============================================================================
# 3b. ĐỐI CHIẾU DANH MỤC SIMBAD
# =============================================================================
def _txt(val):
    """Giá trị ô bảng -> chuỗi sạch (giải mã bytes, bỏ giá trị bị che)."""
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    val = str(val).strip()
    return "" if val in ("--", "None", "nan", "N/A") else val


@st.cache_data(ttl=3600, show_spinner=False)
def _simbad_lookup(ra_deg, dec_deg, radius_arcsec):
    """Truy vấn SIMBAD quanh một tọa độ. Lỗi mạng được NÉM RA (không trả chuỗi) để Streamlit
    không lưu cache một lần mất mạng suốt 1 giờ."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad

    custom_simbad = Simbad()
    custom_simbad.server = "simbad.harvard.edu"
    custom_simbad.TIMEOUT = SIMBAD_TIMEOUT
    try:
        custom_simbad.add_votable_fields("otype")
    except Exception:
        pass  # một số phiên bản astroquery đặt tên trường khác; khi đó chỉ lấy tên thiên thể
    toa_do = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bản cũ phát cảnh báo khi không tìm thấy gì
        result = custom_simbad.query_region(toa_do, radius=radius_arcsec * u.arcsec)
    if result is None or len(result) == 0:
        return SIMBAD_NONE

    cols = {c.lower(): c for c in result.colnames}  # bản mới: 'main_id'; bản cũ: 'MAIN_ID'
    k = 0
    try:  # nếu bảng có ra/dec dạng số (độ) thì chọn thiên thể gần tâm tìm kiếm nhất
        ra = np.asarray(result[cols["ra"]], dtype=float)
        dec = np.asarray(result[cols["dec"]], dtype=float)
        lech = np.hypot((ra - ra_deg) * np.cos(np.radians(dec_deg)), dec - dec_deg)
        if np.isfinite(lech).any():
            k = int(np.nanargmin(lech))
    except Exception:
        k = 0
    main_id = _txt(result[cols["main_id"]][k]) if "main_id" in cols else ""
    otype = _txt(result[cols["otype"]][k]) if "otype" in cols else ""
    nhan = OTYPE_LABELS.get(otype, otype if otype else "chưa rõ loại")
    return f'{main_id or "Không rõ tên"} ({nhan})'


def query_simbad_catalog(ra_deg: float, dec_deg: float, radius_arcsec: float = SIMBAD_RADIUS) -> str:
    """Tên + phân loại SIMBAD của thiên thể gần tọa độ nhất. Không bao giờ làm sập ứng dụng."""
    try:
        return _simbad_lookup(round(float(ra_deg), 3), round(float(dec_deg), 5), float(radius_arcsec))
    except ImportError:
        return SIMBAD_NO_LIB
    except Exception:
        return SIMBAD_OFFLINE


def simbad_ngan(text):
    """Nhãn rút gọn để ghi trên biểu đồ."""
    if text == SIMBAD_NONE:
        return "Chưa định danh"
    if text == SIMBAD_OFFLINE:
        return "SIMBAD offline"
    if text in (SIMBAD_SKIPPED, SIMBAD_OFF, SIMBAD_NO_LIB, SIMBAD_NA):
        return ""
    ten = text.split(" (")[0]
    return ten if len(ten) <= 18 else ten[:17] + "…"


def sep_pa(ra1, dec1, ra2, dec2):
    """Khoảng cách góc (arcsec) và góc vị trí PA (độ, tính từ hướng Bắc sang Đông) từ điểm 1 tới điểm 2."""
    a1, d1, a2, d2 = np.radians([ra1, dec1, ra2, dec2])
    da = a2 - a1
    x = np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(da)
    y = np.cos(d2) * np.sin(da)
    sep = np.arctan2(np.hypot(x, y), np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(da))
    return float(np.degrees(sep) * 3600.0), float(np.degrees(np.arctan2(y, x)) % 360.0)


def fmt_dt(hours):
    return f"{hours:.1f} giờ" if hours < 72 else f"{hours / 24:.1f} ngày"


# =============================================================================
# 4. SIDEBAR
# =============================================================================
co_demo = DEMO_T1.exists() and DEMO_T2.exists()

with st.sidebar:
    st.header("⚙️ Điều khiển đầu vào")
    che_do = st.radio("Chế độ dữ liệu", [SRC_DEMO, SRC_MOCK, SRC_UPLOAD])

    file_1 = file_2 = None
    n_movers, seed = 3, 42
    if che_do == SRC_DEMO:
        if co_demo:
            st.caption(f"Đã nạp `data/{DEMO_T1.name}` và `data/{DEMO_T2.name}`.")
        else:
            st.warning("Không tìm thấy 2 file demo trong thư mục `data/`. Đang dùng dữ liệu giả lập.")
    elif che_do == SRC_UPLOAD:
        file_1 = st.file_uploader("Ảnh T1 (.fits)", type=["fits", "fit", "fz"])
        file_2 = st.file_uploader("Ảnh T2 (.fits)", type=["fits", "fit", "fz"])
        st.caption("Nếu cả hai file có WCS, ảnh T2 sẽ tự động được chiếu về lưới của T1.")
    else:
        n_movers = st.slider("Số vật thể di chuyển", 1, 3, 3)
        seed = st.number_input("Seed bầu trời", 0, 9999, 42, help="Đổi seed để sinh bầu trời khác")

    # Đọc trước metadata của T1 (band, độ phân giải góc, cube) để đồng bộ các điều khiển bên dưới
    raw1 = raw2 = None
    if che_do == SRC_DEMO and co_demo:
        raw1, raw2 = DEMO_T1.read_bytes(), DEMO_T2.read_bytes()
    elif che_do == SRC_UPLOAD and file_1 and file_2:
        raw1, raw2 = file_1.getvalue(), file_2.getvalue()

    meta, loi_doc = None, None
    if raw1 is not None:
        try:
            meta = load_fits(raw1)
        except ImportError:
            loi_doc = "Chưa cài astropy. Chạy `pip install astropy reproject scipy scikit-image` để đọc file FITS."
        except Exception as e:
            loi_doc = f"Không đọc được file FITS T1: {e}"
    pixel_scale = meta["pixel_scale"] if meta else PIXEL_SCALE_DEFAULT
    la_cube = bool(meta and meta["n_planes"] > 1)

    st.divider()
    st.subheader("🧪 Tiền xử lý SPHEREx Level 2")
    tru_zodi = st.checkbox("Khử nền ánh sáng hoàng đạo (ZODI)", value=True,
                           help="Trừ ma trận HDU ZODI (hoặc BACKGROUND) khỏi ảnh khoa học")
    loc_flags = st.checkbox("Lọc mặt nạ tia vũ trụ (FLAGS)", value=True,
                            help="Gán NaN cho pixel dính cờ lỗi phần cứng / tia vũ trụ (bitwise AND) trước khi "
                                 "chiếu lưới. Bỏ qua cờ SOURCE nên các ngôi sao được giữ lại.")
    st.caption("Chỉ áp dụng cho dữ liệu SPHEREx thật.")

    st.divider()
    st.subheader("🔬 Tham số vật lý")
    # Dải bước sóng: khóa theo header khi dùng FITS thật 2D; mở khi Mock hoặc FITS cube 3D
    khoa_band = bool(meta) and not la_cube
    band_idx = meta["band_idx"] if (meta and meta["band_idx"] is not None) else 0
    dai_pho = st.selectbox("Dải bước sóng SPHEREx", BANDS, index=band_idx, disabled=khoa_band)
    if khoa_band:
        nguon_band = meta["band_src"] if meta["band_src"] else "header không có FILTER/BAND/WAVELEN, mặc định Band 1"
        st.caption(f"Tự động khóa theo thông số FITS header ({nguon_band}).")
    elif la_cube:
        st.caption(f'FITS cube {meta["n_planes"]} kênh: band được chọn quyết định lát cắt đem phân tích.')
    cube_index = BANDS.index(dai_pho) if la_cube else 0

    # Mặc định cho dữ liệu SPHEREx thật: 9σ; dữ liệu giả lập: 5σ
    mac_dinh_that = bool(meta)
    nguong_sigma = st.slider("Ngưỡng lọc nhiễu (σ)", 1.0, 10.0, 9.0 if mac_dinh_that else 5.0, 0.5,
                             help="Chỉ giữ các đỉnh trên ảnh trừ sáng hơn k lần độ nhiễu nền")
    st.caption("Ngưỡng thấp → bắt được vật thể mờ nhưng nhiều báo động giả. "
               "Mặc định 9σ cho dữ liệu thật, 5σ cho dữ liệu giả lập.")

    # Bán kính ghép cặp: 10-80 px; mặc định 45 px cho dữ liệu thật, 60 px cho mock (ảnh mock dịch tới 51 px)
    kieu = "that" if meta else "mock"
    if st.session_state.get("_kieu_ban_kinh") != kieu or "max_radius" not in st.session_state:
        st.session_state["_kieu_ban_kinh"] = kieu
        st.session_state["max_radius"] = R_DEFAULT_REAL if meta else R_DEFAULT_MOCK
    max_radius = int(st.session_state["max_radius"])
    st.session_state["max_radius"] = max_radius  # giữ giá trị khi nhãn thay đổi
    ban_kinh = st.slider(f'Bán kính ghép cặp tối đa: {max_radius} px (~{max_radius * pixel_scale:.0f}")',
                         10, 80, step=5, key="max_radius",
                         help="Một đỉnh ở T1 chỉ được ghép với đỉnh ở T2 nằm trong bán kính này "
                              f"(và cách tối thiểu {MIN_SHIFT_PX:g} px)")

    st.divider()
    st.subheader("🔭 Đối chiếu danh mục")
    dung_simbad = st.checkbox("Định danh tọa độ (RA, Dec) qua kho sao SIMBAD", value=False,
                              help=f"Tra {SIMBAD_TOP} ứng viên tin cậy nhất trong bán kính {SIMBAD_RADIUS:g}″. "
                                   "Cần kết nối Internet; chỉ áp dụng cho dữ liệu thật có WCS.")

# =============================================================================
# 5. NẠP DỮ LIỆU & CHẠY PHÁT HIỆN
# =============================================================================
dang_mock, wcs, valid, info = True, None, None, None

if loi_doc:
    st.error(f"{loi_doc} Đang hiển thị dữ liệu giả lập.")
elif meta is not None:
    try:
        img1, img2, valid, info, header1 = prepare_real(raw1, raw2, tru_zodi, loc_flags, cube_index)
        wcs = _celestial_wcs(header1)
        dang_mock = False
    except ImportError:
        st.error("Thiếu thư viện. Chạy `pip install astropy reproject scipy scikit-image`. "
                 "Đang hiển thị dữ liệu giả lập.")
    except Exception as e:
        st.error(f"Không xử lý được file FITS: {e} Đang hiển thị dữ liệu giả lập.")

if dang_mock:
    pixel_scale = PIXEL_SCALE_DEFAULT
    if che_do == SRC_MOCK:
        img1, img2 = generate_mock_data(int(seed), int(n_movers))
    else:
        img1, img2 = generate_mock_data()
        if che_do == SRC_UPLOAD and not (file_1 and file_2):
            st.info("Hãy upload đủ 2 file T1 và T2. Trong lúc chờ, giao diện hiển thị dữ liệu giả lập.")
    dt_hours, dt_la_gia_dinh = MOCK_DT_H, False
else:
    pixel_scale = info["pixel_scale"]
    dt_hours, dt_la_gia_dinh = info["dt_hours"], info["dt_src"] != "header"

try:
    diff, sigma, cands, loc = detect(img1, img2, float(nguong_sigma), float(ban_kinh), valid)
except ValueError as e:
    st.error(str(e))
    st.stop()


def pix2sky(x, y):
    """Pixel -> (RA, Dec) độ theo WCS của T1. Mock dùng WCS tuyến tính giả định quanh (180.25, 45.10)."""
    if wcs is not None:
        ra, dec = wcs.all_pix2world(x, y, 0)
        return float(ra), float(dec)
    cy, cx = (np.array(img1.shape) - 1) / 2
    dec = 45.10 + (y - cy) * pixel_scale / 3600
    ra = 180.25 - (x - cx) * pixel_scale / 3600 / np.cos(np.radians(45.10))
    return float(ra), float(dec)


# Tọa độ thiên cầu, khoảng cách góc, góc vị trí và vận tốc góc biểu kiến cho từng ứng viên
for i, c in enumerate(cands, 1):
    c["id"] = f"Cand-X{i}"
    c["ra1"] = c["dec1"] = c["ra2"] = c["dec2"] = c["sep"] = c["pa"] = c["vel"] = None
    if not np.isnan(c["x1"]):
        c["ra1"], c["dec1"] = pix2sky(c["x1"], c["y1"])
    if not np.isnan(c["x2"]):
        c["ra2"], c["dec2"] = pix2sky(c["x2"], c["y2"])
    if c["loai"] in (LOAI_MOVE, LOAI_TRAIL):
        c["sep"], c["pa"] = sep_pa(c["ra1"], c["dec1"], c["ra2"], c["dec2"])
        c["vel"] = c["sep"] / dt_hours if dt_hours > 0 else None

movers = [c for c in cands if c["loai"] == LOAI_MOVE]
don_le = [c for c in cands if c["loai"] in (LOAI_NEW, LOAI_GONE)]

# Top-N để vẽ: ưu tiên vật thể "Di chuyển", còn chỗ thì thêm đỉnh đơn lẻ; đều theo snr giảm dần
movers_ve = movers[:TOP_N]
cands_ve = movers_ve + don_le[:TOP_N - len(movers_ve)]

# Đối chiếu SIMBAD cho tối đa SIMBAD_TOP ứng viên: toàn bộ nhóm "Di chuyển" trước, rồi đỉnh đơn lẻ theo snr
co_toa_do_that = (not dang_mock) and wcs is not None
for c in cands:
    c["simbad"] = SIMBAD_SKIPPED if (dung_simbad and co_toa_do_that) else (SIMBAD_NA if dung_simbad else SIMBAD_OFF)
if dung_simbad and co_toa_do_that:
    can_tra = (movers + don_le)[:SIMBAD_TOP]
    if can_tra:
        from concurrent.futures import ThreadPoolExecutor

        def _tra_mot_diem(cand):
            ra, dec = (cand["ra2"], cand["dec2"]) if cand["ra2"] is not None else (cand["ra1"], cand["dec1"])
            return query_simbad_catalog(ra, dec, SIMBAD_RADIUS)

        st.toast(f"Đang đối chiếu song song {len(can_tra)} ứng viên với SIMBAD...", icon="🔭")
        with ThreadPoolExecutor(max_workers=8) as pool:
            ket_qua = list(pool.map(_tra_mot_diem, can_tra))
        for c, res in zip(can_tra, ket_qua):
            c["simbad"] = res

KHONG_KET_QUA = (SIMBAD_NONE, SIMBAD_OFFLINE, SIMBAD_SKIPPED, SIMBAD_OFF, SIMBAD_NO_LIB, SIMBAD_NA)
n_khop = sum(c["simbad"] not in KHONG_KET_QUA for c in cands)
n_la = sum(c["simbad"] == SIMBAD_NONE for c in cands)
n_offline = sum(c["simbad"] == SIMBAD_OFFLINE for c in cands)


# =============================================================================
# 6. TIÊU ĐỀ, THANH TRẠNG THÁI & THẺ SỐ LIỆU
# =============================================================================
badge = '<span class="badge mock">DỮ LIỆU GIẢ LẬP</span>' if dang_mock else '<span class="badge real">DỮ LIỆU THẬT</span>'
st.markdown(
    f'<div class="hero"><h1>🌌 SPHEREx Sky Tracker</h1>'
    f'<p>So sánh hai thời điểm quan sát để phát hiện thiên thể thay đổi vị trí '
    f'<span class="badge">Team 10LY2NHUEr</span>{badge}</p></div>',
    unsafe_allow_html=True,
)

# Thanh trạng thái tổng hợp (chỉ với dữ liệu thật)
if info is not None:
    zodi_txt = "đã trừ" if info["zodi"] else ("tắt" if not tru_zodi else "file thiếu HDU ZODI")
    if info["flags_pct"] is not None:
        flags_txt = (f'che {info["flags_pct"][0]:.1f}% (T1) / {info["flags_pct"][1]:.1f}% (T2) pixel, '
                     f'{info["flags_src"]}, giữ SOURCE')
    else:
        flags_txt = "tắt" if not loc_flags else "file thiếu HDU FLAGS"
    wcs_txt = {"ok": "đã chiếu T2 → lưới T1", "no_wcs": "không có WCS, so thẳng theo pixel",
               "no_reproject": "chưa cài reproject, so thẳng theo pixel"}[info["wcs"]]
    reg_txt = {"applied": "đã dịch", "small": "dưới 0.03 px, không cần dịch",
               "too_big": "vượt 15 px, KHÔNG dịch", "rejected": "không cải thiện độ khớp, KHÔNG dịch",
               "failed": "không đo được", "no_skimage": "bỏ qua (chưa cài scikit-image)",
               "no_scipy": "bỏ qua (chưa cài scipy)", "none": "bỏ qua"}[info["reg"]]
    st.caption(f'**ZODI:** {zodi_txt} · **FLAGS:** {flags_txt} · **WCS Macro:** {wcs_txt} · '
               f'**WCS Sub-pixel:** dx = {info["dx"]:+.2f} px, dy = {info["dy"]:+.2f} px '
               f'({info["shift_arcsec"]:.2f}″; {reg_txt})')
    if "scale_raw" in info:
        scale_txt = f'{info["scale"]:.3f} (trung vị trên {info["n_bright"]} sao sáng chung)'
    else:
        scale_txt = "không chỉnh (không đủ sao sáng chung chưa bão hòa)"
    if info["edge"] is None:
        mask_txt = "bỏ qua (chưa cài scipy)"
    else:
        mask_txt = f'{info["halo_pct"]:.1f}% khung hình ({info["n_sat"]} sao cực sáng), gọt biên {EDGE_PX} px'
    dt_txt = fmt_dt(info["dt_hours"]) + ("" if info["dt_src"] == "header" else " (GIẢ ĐỊNH, header thiếu mốc thời gian)")
    st.caption(f'**Quang thông T2/T1:** {scale_txt} · **Che hào quang:** {mask_txt} · '
               f'**Vùng hợp lệ:** {info["valid_pct"]:.1f}% · **Δt:** {dt_txt} · '
               f'**Tỷ lệ ảnh:** {info["pixel_scale"]:.2f}″/px ({info["pixel_scale_src"]})')

    if not loc_flags:
        st.warning("Đang tắt lọc FLAGS: các tia vũ trụ chưa bị che sẽ xuất hiện như những đỉnh sáng đơn lẻ "
                   "và có thể bị ghép nhầm thành vật thể di chuyển.")
    if info["reg"] == "no_skimage":
        st.warning("Chưa cài `scikit-image` nên CHƯA tinh chỉnh căn chỉnh dưới pixel. Chạy `pip install scikit-image`.")
    if info["edge"] is None:
        st.warning("Chưa cài `scipy` nên CHƯA tinh chỉnh dưới pixel, gọt biên và che hào quang. Chạy `pip install scipy`.")
    if info["reg"] in ("too_big", "rejected"):
        st.warning("Phép đo lệch dưới pixel không đáng tin (quá lớn hoặc không làm hai ảnh khớp hơn) nên không được áp dụng.")
    if "scale_raw" in info and not (SCALE_CLIP[0] < info["scale_raw"] < SCALE_CLIP[1]):
        st.warning(f'Tỷ lệ quang thông đo được là {info["scale_raw"]:.2f}, nằm ngoài khoảng '
                   f'{SCALE_CLIP[0]}–{SCALE_CLIP[1]} nên đã bị cắt về biên. Hai ảnh có thể chụp ở '
                   'bước sóng quá khác nhau để so sánh trực tiếp.')
    if info["halo_pct"] >= HALO_MAX_PCT:
        st.warning(f'Mặt nạ hào quang vẫn che {info["halo_pct"]:.0f}% khung hình dù đã nâng ngưỡng. '
                   'Nếu đang tắt lọc FLAGS, hãy bật lại.')
    if info["wcs"] == "no_reproject":
        st.warning("Chưa cài thư viện `reproject` nên hai ảnh CHƯA được chiếu về cùng lưới WCS. Chạy `pip install reproject`.")
    elif info["wcs"] == "no_wcs":
        st.warning("Ít nhất một file không có WCS hợp lệ nên hai ảnh CHƯA được chiếu về cùng lưới tọa độ.")
    if info["flags_pct"] is not None and max(info["flags_pct"]) > 10:
        st.warning("Mặt nạ FLAGS đang che hơn 10% ảnh dù đã bỏ qua cờ SOURCE. "
                   "Hãy kiểm tra định nghĩa bit trong header FLAGS của file.")

# Thống kê các bộ lọc nhiễu giả (mọi chế độ dữ liệu)
psf_txt = f" · **Khớp PSF T1:** σ = {PSF_MATCH_SIGMA:g} px" if (info is not None and info.get("psf_match")) else ""
st.caption(f'**Bộ lọc:** Đã lọc {loc["trail"]} nhiễu vệt vệ tinh (chuỗi thẳng hàng) · {loc["elong"]} đốm dẹt / kéo dài · '
           f'{loc["spike"]} gai dưới {MIN_AREA} pixel · {loc["veto"]} sao tĩnh (bảo toàn quang thông khẩu độ) · '
           f'{loc["dipole"]} lưỡng cực{psf_txt}')
if dung_simbad and co_toa_do_that:
    st.caption(f'**SIMBAD:** {n_khop} ứng viên khớp danh mục · {n_la} vật thể lạ chưa định danh'
               + (f' · {n_offline} lỗi kết nối' if n_offline else ''))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vật thể di chuyển", len(movers))
m2.metric("Nguồn chỉ đổi độ sáng", len(don_le))
m3.metric("Nhiễu ảnh trừ (1σ)", f"{sigma:.3g}")
m4.metric("Dịch chuyển lớn nhất", f'{max((c["sep"] for c in movers), default=0):.0f}″')


# =============================================================================
# 7. HÀM VẼ
# =============================================================================
def _khoang_hien_thi(img):
    """Khoảng giá trị hiển thị (5% - 99.7%), an toàn với ảnh toàn NaN."""
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(v, [5, 99.7])
    return float(vmin), float(vmax if vmax > vmin else vmin + 1.0)


def show_sky(ax, img, title):
    vmin, vmax = _khoang_hien_thi(img)
    ax.imshow(img, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")


def render(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # tránh rò bộ nhớ sau nhiều lần rerun


R = max(4.0, img1.shape[0] / 22)  # bán kính vòng khoanh theo cỡ ảnh
if dang_mock:
    tieu_de_t2 = "T2 · Sau 6 tháng"
else:
    tieu_de_t2 = "T2 · Đã căn chỉnh về lưới T1" if info["wcs"] == "ok" else "T2 · Ảnh sau"

# =============================================================================
# 8. KHU VỰC CHÍNH
# =============================================================================
tab_so_sanh, tab_time_lapse, tab_du_lieu = st.tabs(
    ["🔍 So sánh & Trừ ảnh", "⏳ Nội suy thời gian", "📋 Dữ liệu ứng viên"])

# ---- TAB 1 ------------------------------------------------------------------
with tab_so_sanh:
    st.subheader(f"Difference Viewer · {dai_pho}")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        show_sky(ax, img1, "T1 · Ảnh gốc")
        for c in movers_ve:
            ax.add_patch(plt.Circle((c["x1"], c["y1"]), R, color=C_T1, fill=False, lw=1.2, ls="--"))
        render(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        show_sky(ax, img2, tieu_de_t2)
        for c in movers_ve:
            ax.add_patch(plt.Circle((c["x2"], c["y2"]), R, color=C_T2, fill=False, lw=1.2))
        render(fig)

    with c3:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        lim = max(8 * sigma, float(nguong_sigma) * sigma * 1.5)
        nen_diff = float(np.nanmedian(diff)) if np.isfinite(diff).any() else 0.0
        ax.imshow(diff - nen_diff, cmap=CMAP_DIFF, origin="lower", vmin=-lim, vmax=lim)
        ax.set_title("T2 − T1 · Ứng viên", color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        for c in cands_ve:
            if c["loai"] == LOAI_MOVE:
                ax.add_patch(plt.Circle((c["x1"], c["y1"]), R, color=C_T1, fill=False, lw=1.2, ls="--"))
                ax.add_patch(plt.Circle((c["x2"], c["y2"]), R, color=C_HIT, fill=False, lw=2))
                ax.annotate("", xy=(c["x2"], c["y2"]), xytext=(c["x1"], c["y1"]),
                            arrowprops=dict(arrowstyle="->", color=C_HIT, lw=1, alpha=.8,
                                            shrinkA=R * 2.2, shrinkB=R * 2.2))
                ngan = simbad_ngan(c["simbad"])
                ax.text(c["x2"], c["y2"] + R * 1.4, f'{c["id"]} · {ngan}' if ngan else c["id"],
                        color=C_HIT, ha="center", fontsize=7)
            else:
                x = c["x2"] if not np.isnan(c["x2"]) else c["x1"]
                y = c["y2"] if not np.isnan(c["y2"]) else c["y1"]
                ax.add_patch(plt.Rectangle((x - R, y - R), 2 * R, 2 * R, color=C_VAR, fill=False, lw=1.2))
                ax.text(x, y + R * 1.4, c["id"], color=C_VAR, ha="center", fontsize=7)
        render(fig)

    st.markdown(
        f'<div class="legend">'
        f'<span><i class="dot" style="background:{C_T1}"></i>Vị trí ở T1 (đỉnh âm)</span>'
        f'<span><i class="dot" style="background:{C_HIT}"></i>Vị trí ở T2 – ứng viên di chuyển</span>'
        f'<span><i class="dot" style="background:{C_VAR}"></i>Chỉ đổi độ sáng, không di chuyển</span>'
        f'</div>', unsafe_allow_html=True)
    if len(cands) > len(cands_ve):
        st.caption(f"Biểu đồ chỉ vẽ {len(cands_ve)}/{len(cands)} ứng viên có độ tin cậy cao nhất "
                   "(ưu tiên vật thể di chuyển). Danh sách đầy đủ nằm ở tab Dữ liệu ứng viên.")

    n_bo = loc["dipole"] + loc["spike"] + loc["elong"] + loc["veto"] + loc["trail"]
    loc_txt = f" Các bộ lọc đã loại **{n_bo}** nhiễu giả (chi tiết ở dòng Bộ lọc phía trên)." if n_bo else ""
    if movers:
        st.success(f"Lọc ở ngưỡng {nguong_sigma}σ, dịch chuyển {MIN_SHIFT_PX:g}–{ban_kinh} px: "
                   f"tìm thấy **{len(movers)}** vật thể di chuyển (|Δ độ sáng| ≤ {MAX_DFLUX_PCT:g}%).{loc_txt}")
    else:
        st.warning(f"Không có vật thể di chuyển nào vượt ngưỡng {nguong_sigma}σ với dịch chuyển "
                   f"{MIN_SHIFT_PX:g}–{ban_kinh} px.{loc_txt} Thử hạ ngưỡng hoặc tăng bán kính ghép cặp.")

# ---- TAB 2 ------------------------------------------------------------------
with tab_time_lapse:
    st.subheader("Mô phỏng chuyển động giữa T1 và T2")
    st.warning(
        "⚠️ **Cảnh báo khoa học:** SPHEREx chỉ quan sát tại hai thời điểm T1 và T2. Mọi vị trí ở giữa "
        "là **nội suy tuyến tính** để minh họa, **không phải dữ liệu đo** và không phải quỹ đạo thật "
        "(quỹ đạo thật là đường cong, cần tối thiểu 3 lần quan sát để xác định).")

    cs, cb = st.columns([5, 1])
    tien_do = cs.slider("Thời gian nội suy (0% = T1, 100% = T2)", 0, 100, 50, format="%d%%")
    cb.write("")
    play = cb.button("▶ Phát", help="Chạy hoạt ảnh từ T1 đến T2")
    khung = st.empty()

    def draw_frame(p):
        t = p / 100
        fig, ax = plt.subplots(figsize=(7.5, 5))
        vmin, vmax = _khoang_hien_thi(img1)
        ax.imshow(img2 if p == 100 else img1,
                  cmap="gray", origin="lower", vmin=vmin, vmax=vmax, alpha=.45)
        for c in movers_ve:
            x = c["x1"] + (c["x2"] - c["x1"]) * t
            y = c["y1"] + (c["y2"] - c["y1"]) * t
            ax.plot([c["x1"], c["x2"]], [c["y1"], c["y2"]], ls="--", color=MUTED, lw=1, alpha=.6)
            ax.plot([c["x1"], x], [c["y1"], y], color=C_T2, lw=2, alpha=.9)       # đoạn đã đi
            ax.scatter([c["x1"]], [c["y1"]], s=40, facecolors="none", edgecolors=C_T1, lw=1.5)
            ax.scatter([c["x2"]], [c["y2"]], s=40, facecolors="none", edgecolors=C_T2, lw=1.5)
            ax.scatter([x], [y], s=260, color=C_VAR, alpha=.18, lw=0)             # quầng sáng
            ax.scatter([x], [y], s=60, color=C_VAR, edgecolors="white", lw=.8, zorder=5)
            ax.text(x, y + R * 1.2, c["id"], color=C_VAR, ha="center", fontsize=8)
        nhan = "Dữ liệu quan sát" if p in (0, 100) else "NỘI SUY – không phải ảnh thật"
        ax.set_title(f"{p}%  ·  {nhan}", color=C_HIT if p in (0, 100) else C_VAR, fontsize=10, pad=8)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        fig.tight_layout()
        return fig

    if not movers:
        st.info("Chưa có vật thể di chuyển nào để nội suy. Hạ ngưỡng σ hoặc tăng bán kính ghép cặp ở thanh bên.")
    else:
        if play:
            for p in range(0, 101, 4):
                f = draw_frame(p)
                khung.pyplot(f)
                plt.close(f)
                time.sleep(0.03)
        f = draw_frame(100 if play else tien_do)
        khung.pyplot(f)
        plt.close(f)
        if tien_do in (0, 100) and not play:
            st.info("✅ Đang hiển thị vị trí quan sát gốc.")

# ---- TAB 3 ------------------------------------------------------------------
with tab_du_lieu:
    st.subheader("Bảng ứng viên (RA/Dec)")
    if dang_mock:
        st.caption("RA/Dec tính từ một WCS giả định cho dữ liệu mock (không ứng với bầu trời thật); "
                   f"vận tốc góc tính với Δt = {fmt_dt(dt_hours)}.")
    else:
        nguon_toa_do = "WCS của ảnh T1" if wcs is not None else "WCS giả định (file thiếu WCS, chỉ tin cột pixel)"
        st.caption(f"RA/Dec tính từ {nguon_toa_do}; vận tốc góc = Δθ / Δt với Δt = {fmt_dt(dt_hours)}.")
        if dt_la_gia_dinh:
            st.warning(f"Header FITS không có mốc thời gian (MJD-OBS / DATE-OBS) nên Δt = {DT_FALLBACK_H} giờ là "
                       "giá trị GIẢ ĐỊNH. Cột vận tốc góc vì thế chỉ mang tính minh họa, không phải số đo.")

    if dung_simbad and co_toa_do_that:
        thong_ke = (f"**SIMBAD** (bán kính {SIMBAD_RADIUS:g}″, {min(len(movers) + len(don_le), SIMBAD_TOP)} ứng viên "
                    f"được tra): **{n_khop}** khớp danh mục · **{n_la}** vật thể lạ chưa định danh")
        if n_offline:
            st.warning(thong_ke + f" · **{n_offline}** không tra được do lỗi kết nối SIMBAD.")
        elif any(c["simbad"] == SIMBAD_NO_LIB for c in cands):
            st.warning("Chưa cài `astroquery` nên không đối chiếu được SIMBAD. Chạy `pip install astroquery`.")
        else:
            st.info(thong_ke)
        st.caption("SIMBAD không chứa tiểu hành tinh và sao chổi, và không đầy đủ với sao mờ. "
                   "\"Chưa định danh\" nghĩa là không có thiên thể SIMBAD trong bán kính tìm kiếm, "
                   "chưa phải bằng chứng về một vật thể mới.")
    elif dung_simbad:
        st.caption("Không đối chiếu SIMBAD: tọa độ của dữ liệu này không ứng với bầu trời thật.")

    def lam_tron(v, n):
        return None if v is None else round(v, n)

    rows = []
    for c in cands:
        x = c["x2"] if not np.isnan(c["x2"]) else c["x1"]
        y = c["y2"] if not np.isnan(c["y2"]) else c["y1"]
        rows.append({
            "Mã ứng viên": c["id"], "Định danh thiên văn (SIMBAD)": c["simbad"], "Phân loại": c["loai"],
            "RA1 (độ)": lam_tron(c["ra1"], 5), "Dec1 (độ)": lam_tron(c["dec1"], 5),
            "RA2 (độ)": lam_tron(c["ra2"], 5), "Dec2 (độ)": lam_tron(c["dec2"], 5),
            "x (px)": round(x, 2), "y (px)": round(y, 2),
            "Δθ (arcsec)": lam_tron(c["sep"], 1), "PA (độ)": lam_tron(c["pa"], 1),
            "Vận tốc góc (arcsec/giờ)": lam_tron(c["vel"], 3),
            "Δ Độ sáng (%)": None if np.isnan(c["dflux"]) else round(c["dflux"], 1),
            "Độ tin cậy (σ)": round(c["snr"], 1),
        })
    df = pd.DataFrame(rows)

    if df.empty:
        st.info("Không có ứng viên nào ở ngưỡng hiện tại.")
    else:
        try:
            st.dataframe(df, hide_index=True, width="stretch")
        except Exception:  # Streamlit bản cũ
            st.dataframe(df, hide_index=True, use_container_width=True)

    st.download_button("📥 Tải file CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="candidates.csv", mime="text/csv", disabled=df.empty)
st.divider()
st.caption("Dữ liệu khoa học: NASA / JPL-Caltech SPHEREx Mission & NASA/IPAC Infrared Science Archive (IRSA) · Đối chiếu danh mục: CDS SIMBAD.")

st.divider()
st.caption(
    """Ấn phẩm này sử dụng dữ liệu từ Kính viễn vọng Quang phổ Kế hoạch Khám phá Lịch sử Vũ trụ, Kỷ nguyên Tái ion hóa và Băng (SPHEREx), một dự án hợp tác giữa Phòng thí nghiệm Phản lực (Jet Propulsion Laboratory) và Viện Công nghệ California (Caltech), và được tài trợ bởi Cơ quan Hàng không và Vũ trụ Quốc gia (NASA).

This publication makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration.

Kho lưu trữ: NASA/IPAC Infrared Science Archive (IRSA)  Đối chiếu danh mục: CDS SIMBAD."""
)