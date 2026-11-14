# -*- coding: utf-8 -*-
"""
SPHEREx Sky Tracker - MVP giao diện Streamlit (NASA Space Apps)
Cài:   pip install streamlit numpy pandas matplotlib astropy reproject scipy
Chạy:  streamlit run app.py --theme.base dark
Dữ liệu demo (tùy chọn): data/spherex_t1_demo.fits, data/spherex_t2_demo.fits
"""
import io
import time
import warnings
from collections import deque
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

PIXEL_SCALE = 6.2     # arcsec/pixel của SPHEREx

# Đường dẫn 2 file demo SPHEREx thật (đã cắt 512x512)
DATA_DIR = Path(__file__).resolve().parent / "data"
DEMO_T1 = DATA_DIR / "spherex_t1_demo.fits"
DEMO_T2 = DATA_DIR / "spherex_t2_demo.fits"

SRC_DEMO = "Dữ liệu SPHEREx thật (Có sẵn)"
SRC_MOCK = "Giả lập (Mock)"
SRC_UPLOAD = "Dữ liệu SPHEREx thật (Upload)"

MIN_SHIFT_PX = 3.0    # dịch chuyển tối thiểu (>= 18.6 arcsec); nhỏ hơn = dư ảnh sao tĩnh
MAX_FLUX_RATIO = 1.6  # tỷ số thông lượng T2/T1 tối đa khi ghép cặp
TOP_N = 12            # số ứng viên tối đa được vẽ trên biểu đồ
MAX_DFLUX_PCT = 35.0  # |Δ độ sáng| tối đa (%) để một cặp được coi là "Di chuyển"

# Cân bằng quang thông & mặt nạ vật lý (chỉ áp dụng cho dữ liệu SPHEREx thật)
BRIGHT_K = 15.0           # pixel sao sáng chung: > 15σ trên cả T1 và T2
SCALE_CLIP = (0.7, 1.3)   # giới hạn hệ số tỷ lệ quang thông T2/T1
EDGE_PX = 15              # gọt biên vùng phủ sau reproject (pixel)
SAT_K = 250.0             # tâm sao cực sáng / bão hòa: > 250σ
HALO_R = 18               # bán kính che hào quang quanh sao cực sáng (pixel)
MIN_SCALE_STARS = 5       # cần tối thiểu ngần này sao sáng chung mới chỉnh tỷ lệ quang thông

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


def read_spherex(raw):
    """Đọc 1 file FITS SPHEREx Level 2 (bytes).
    Trả về dict: image, zodi, flags (None nếu thiếu HDU), header (chuỗi) của HDU ảnh."""
    from astropy.io import fits  # import trễ: chỉ cần khi dùng dữ liệu thật

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(io.BytesIO(raw)) as hdul:
            by_name = {h.name.upper(): h for h in hdul}
            hdu = by_name.get("IMAGE")
            if hdu is None or hdu.data is None or np.ndim(hdu.data) != 2:
                # File không theo chuẩn SPHEREx: lấy mảng 2D đầu tiên
                hdu = next((h for h in hdul if h.data is not None and np.ndim(h.data) == 2), None)
            if hdu is None:
                raise ValueError("Không tìm thấy ảnh 2D trong file FITS.")

            out = {"image": np.array(hdu.data, dtype=float), "zodi": None, "flags": None,
                   "flag_bits": {}, "header": hdu.header.tostring()}
            for key in ("ZODI", "FLAGS"):
                h = by_name.get(key)
                if h is not None and h is not hdu and h.data is not None and h.data.shape == out["image"].shape:
                    out[key.lower()] = np.array(h.data)
                    if key == "FLAGS":
                        # Header FLAGS của SPHEREx ghi tên cờ của bit n trong từ khóa MSKNnnnn
                        for k, v in h.header.items():
                            if k.startswith("MSKN") and k[4:].isdigit():
                                out["flag_bits"][str(v).strip().upper()] = int(k[4:])
    return out


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


def _celestial_wcs(header_str):
    """Dựng WCS từ chuỗi header; trả None nếu không có hệ tọa độ bầu trời hợp lệ."""
    if not header_str:
        return None
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = WCS(fits.Header.fromstring(header_str))
        return w if w.has_celestial else None
    except Exception:
        return None


@st.cache_data(show_spinner="Đang tiền xử lý và đồng bộ WCS…")
def prepare_real(raw1, raw2, tru_zodi, loc_flags):
    """Tiền xử lý Level 2 (ZODI, FLAGS) rồi chiếu T2 về lưới WCS của T1.
    Sau đó: cân bằng quang thông T2/T1, gọt biên và che hào quang sao cực sáng.
    Trả về: img1, img2 (đã chiếu và chuẩn hóa), valid_mask, info, header T1."""
    a, b = read_spherex(raw1), read_spherex(raw2)
    info = {"zodi": False, "flags_pct": None, "flags_src": None, "wcs": "no_wcs",
            "scale": 1.0, "n_bright": 0, "edge": False, "n_sat": 0, "halo_pct": 0.0}
    imgs, pcts = [], []
    for d in (a, b):
        img = d["image"].copy()
        img[~np.isfinite(img)] = np.nan
        if tru_zodi and d["zodi"] is not None:
            img -= d["zodi"]                 # khử nền ánh sáng hoàng đạo
        if loc_flags and d["flags"] is not None:
            bad, info["flags_src"] = bad_pixel_mask(d["flags"], d["flag_bits"])
            img[bad] = np.nan                # chỉ che pixel hỏng / tia vũ trụ, giữ nguyên sao (SOURCE)
            pcts.append(float(bad.mean() * 100))
        imgs.append(img)
    img1, img2 = imgs
    info["zodi"] = bool(tru_zodi and a["zodi"] is not None and b["zodi"] is not None)
    if len(pcts) == 2:
        info["flags_pct"] = tuple(pcts)

    wcs1, wcs2 = _celestial_wcs(a["header"]), _celestial_wcs(b["header"])
    footprint = None
    if wcs1 is not None and wcs2 is not None:
        try:
            from reproject import reproject_interp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img2_goc = img2
                img2, footprint = reproject_interp((img2_goc, wcs2), wcs1, shape_out=img1.shape)
                # Vùng phủ hình học của T2 trên lưới T1 (không bị khoét lỗ bởi các pixel NaN)
                _, phu_hinh_hoc = reproject_interp((np.ones(img2_goc.shape), wcs2), wcs1, shape_out=img1.shape)
            info["wcs"] = "ok"
        except ImportError:
            info["wcs"] = "no_reproject"

    if footprint is None:
        if img1.shape != img2.shape:
            raise ValueError(f"Không đồng bộ được WCS và kích thước hai ảnh khác nhau: {img1.shape} và {img2.shape}.")
        footprint = np.ones(img1.shape)
        phu_hinh_hoc = footprint

    phu = np.nan_to_num(phu_hinh_hoc) > 0                 # vùng T2 phủ lên lưới T1
    valid = np.isfinite(img1) & np.isfinite(img2) & (footprint > 0)
    if valid.sum() < 100:
        raise ValueError("Gần như không còn pixel hợp lệ chung giữa T1 và T2 "
                         "(hai ảnh không chồng lấn, hoặc mặt nạ FLAGS che gần hết - thử tắt lọc FLAGS).")

    # Nền và nhiễu bền vững (trung vị + MAD) của từng ảnh trên vùng hợp lệ
    def _bg_sigma(img):
        v = img[valid]
        bg = float(np.median(v))
        sg = 1.4826 * float(np.median(np.abs(v - bg)))
        return bg, (sg if sg > 0 else 1.0)

    bg1, sigma_1 = _bg_sigma(img1)
    bg2, sigma_2 = _bg_sigma(img2)
    s1, s2 = img1 - bg1, img2 - bg2                       # tín hiệu trên nền

    try:
        from scipy import ndimage
    except ImportError:
        ndimage = None

    # (1) Cân bằng tỷ lệ quang thông trên các sao sáng chung của T1 và T2
    with np.errstate(invalid="ignore", divide="ignore"):
        bright_mask = valid & (s1 > BRIGHT_K * sigma_1) & (s2 > BRIGHT_K * sigma_2)
        ratio = np.nan
        if ndimage is not None and bright_mask.any():
            # Tỷ số tính theo TỔNG thông lượng từng sao (khẩu độ = vùng sáng nở thêm 2 px), rồi lấy
            # trung vị giữa các sao. Tỷ số từng pixel bị lệch thấp vì reproject làm nhòe đỉnh sao ở T2.
            nhan, n_sao = ndimage.label(ndimage.binary_dilation(bright_mask, iterations=2))
            idx = np.arange(1, n_sao + 1)
            f1 = ndimage.sum(np.where(valid, s1, 0.0), nhan, idx)
            f2 = ndimage.sum(np.where(valid, s2, 0.0), nhan, idx)
            ok = f1 > 0
            info["n_bright"] = int(ok.sum())
            if info["n_bright"] >= MIN_SCALE_STARS:
                ratio = float(np.median(f2[ok] / f1[ok]))
        elif bright_mask.sum() >= 20:
            # Không có scipy: dùng trung vị tỷ số từng pixel
            info["n_bright"] = MIN_SCALE_STARS
            ratio = float(np.nanmedian(s2[bright_mask] / s1[bright_mask]))
        if np.isfinite(ratio) and ratio > 0:
            info["scale"] = float(np.clip(ratio, *SCALE_CLIP))
            info["scale_raw"] = ratio
        img2 = img2 / info["scale"]
        s2 = s2 / info["scale"]
        sigma_2 = sigma_2 / info["scale"]

        # (2a) Tâm sao cực sáng / bão hòa trên T1 hoặc T2 (xác định trước khi gọt biên)
        loi = valid & ((s1 > SAT_K * sigma_1) | (s2 > SAT_K * sigma_2))

    if ndimage is not None:
        # (2b) Gọt biên: co VÙNG PHỦ vào trong EDGE_PX pixel (tính cả mép ảnh T1).
        # Không co trực tiếp valid_mask, vì mỗi pixel bị cờ FLAGS là một "lỗ" và phép co
        # sẽ khoét rộng từng lỗ thêm 15 px, xóa gần hết ảnh.
        phu_trong = ndimage.binary_erosion(phu, iterations=EDGE_PX, border_value=0)
        valid &= phu_trong
        info["edge"] = True

        # (2c) Che hào quang: nở các tâm sao cực sáng bằng đĩa tròn bán kính HALO_R
        if loi.any():
            yy, xx = np.mgrid[-HALO_R:HALO_R + 1, -HALO_R:HALO_R + 1]
            dia = (yy ** 2 + xx ** 2) <= HALO_R ** 2
            halo = ndimage.binary_dilation(loi, structure=dia)
            info["n_sat"] = int(ndimage.label(loi, structure=np.ones((3, 3)))[1])
            info["halo_pct"] = float(halo.mean() * 100)
            valid &= ~halo
    else:
        info["edge"] = None  # thiếu scipy: bỏ qua gọt biên và che hào quang

    if valid.sum() < 100:
        raise ValueError("Gần như không còn pixel hợp lệ sau khi gọt biên và che hào quang sao sáng.")
    info["valid_pct"] = float(valid.mean() * 100)
    return img1, img2, valid, info, (a["header"] if wcs1 is not None else None)


# =============================================================================
# 3. THUẬT TOÁN PHÁT HIỆN (chạy thật trên ảnh trừ, không hard-code)
# =============================================================================
MAX_BLOBS = 400  # giới hạn số đốm mỗi dấu để ghép cặp không bị chậm


def _find_blobs(mask, values, min_area=3):
    """Gom các pixel vượt ngưỡng liền kề thành từng đốm (8 hướng)."""
    if mask.sum() > 200_000:
        raise ValueError("Quá nhiều pixel vượt ngưỡng - hai ảnh có thể chưa được căn chỉnh (align).")
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    blobs = []
    for y0, x0 in np.argwhere(mask):
        if seen[y0, x0]:
            continue
        queue, pix = deque([(y0, x0)]), []
        seen[y0, x0] = True
        while queue:
            y, x = queue.popleft()
            pix.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(pix) < min_area:
            continue  # loại pixel nhiễu đơn lẻ
        ys, xs = np.array(pix).T
        v = np.abs(values[ys, xs])
        blobs.append({"y": float((ys * v).sum() / v.sum()), "x": float((xs * v).sum() / v.sum()),
                      "flux": float(v.sum()), "peak": float(v.max())})
    blobs.sort(key=lambda b: -b["peak"])
    return blobs[:MAX_BLOBS]


@st.cache_data(show_spinner=False)
def detect(img_t1, img_t2, k_sigma, r_max, valid=None, min_shift_px=MIN_SHIFT_PX):
    """Trừ ảnh (chỉ trên vùng hợp lệ) -> ước lượng nhiễu -> lọc ngưỡng ->
    loại dư ảnh sao tĩnh (cặp âm-dương cách nhau < min_shift_px) ->
    ghép cặp đốm âm (T1) với đốm dương (T2) theo thông lượng VÀ khoảng cách.
    Trả về: ảnh trừ, sigma, danh sách ứng viên, số dư ảnh sao đã loại."""
    diff = img_t2 - img_t1
    if valid is None:
        valid = np.isfinite(diff)
    diff = np.where(valid, diff, np.nan)

    vals = diff[valid]
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))  # độ lệch chuẩn bền vững (MAD)
    sigma = sigma if sigma > 0 else 1.0
    d = np.where(valid, diff - med, 0.0)  # pixel không hợp lệ = 0 nên không bao giờ vượt ngưỡng

    neg = _find_blobs(d < -k_sigma * sigma, d)  # vật thể có ở T1, mất ở T2
    pos = _find_blobs(d > k_sigma * sigma, d)   # vật thể mới xuất hiện ở T2

    # Bước 1 - Lọc dipole: đốm âm và đốm dương cách nhau dưới min_shift_px là sai số nội suy
    # sub-pixel của một ngôi sao đứng yên sau khi reproject -> bỏ cả hai, không coi là ứng viên.
    dip = sorted(
        (float(np.hypot(p["x"] - n["x"], p["y"] - n["y"])), i, j)
        for i, n in enumerate(neg) for j, p in enumerate(pos)
        if abs(p["x"] - n["x"]) < min_shift_px and abs(p["y"] - n["y"]) < min_shift_px
    )
    used_n, used_p, n_dipole = set(), set(), 0
    for dist, i, j in dip:
        if dist < min_shift_px and i not in used_n and j not in used_p:
            used_n.add(i); used_p.add(j)
            n_dipole += 1

    # Bước 2 - Ghép cặp tham lam: thông lượng chênh tối đa MAX_FLUX_RATIO lần và
    # |Δ độ sáng| <= MAX_DFLUX_PCT %, khoảng cách từ min_shift_px đến r_max
    pairs = []
    for i, n in enumerate(neg):
        if i in used_n:
            continue
        for j, p in enumerate(pos):
            if j in used_p:
                continue
            dist = float(np.hypot(p["x"] - n["x"], p["y"] - n["y"]))
            if dist < min_shift_px or dist > r_max:
                continue
            flux_cost = abs(np.log(p["flux"] / n["flux"]))
            if flux_cost > np.log(MAX_FLUX_RATIO):
                continue
            delta_flux_pct = (p["flux"] / n["flux"] - 1) * 100
            if abs(delta_flux_pct) > MAX_DFLUX_PCT:
                continue  # vật thể di chuyển thật phải giữ độ sáng gần như không đổi
            pairs.append((flux_cost + dist / r_max, i, j, dist))
    pairs.sort()

    dip_n, dip_p, rows = set(used_n), set(used_p), []
    for _, i, j, dist in pairs:
        if i in used_n or j in used_p:
            continue
        used_n.add(i); used_p.add(j)
        n, p = neg[i], pos[j]
        rows.append({"loai": "Di chuyển", "x1": n["x"], "y1": n["y"], "x2": p["x"], "y2": p["y"],
                     "shift_px": dist,
                     "dflux": (p["flux"] / n["flux"] - 1) * 100,
                     "snr": min(n["peak"], p["peak"]) / sigma})
    for j, p in enumerate(pos):
        if j not in used_p and j not in dip_p:
            rows.append({"loai": "Sáng lên / mới", "x1": np.nan, "y1": np.nan, "x2": p["x"], "y2": p["y"],
                         "shift_px": 0.0, "dflux": np.nan, "snr": p["peak"] / sigma})
    for i, n in enumerate(neg):
        if i not in used_n and i not in dip_n:
            rows.append({"loai": "Mờ đi / mất", "x1": n["x"], "y1": n["y"], "x2": np.nan, "y2": np.nan,
                         "shift_px": 0.0, "dflux": np.nan, "snr": n["peak"] / sigma})
    rows.sort(key=lambda r: -r["snr"])
    return diff, float(sigma), rows, n_dipole


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

    st.divider()
    st.subheader("🧪 Tiền xử lý SPHEREx Level 2")
    tru_zodi = st.checkbox("Khử nền ánh sáng hoàng đạo (ZODI)", value=True,
                           help="Trừ ma trận HDU ZODI khỏi IMAGE")
    loc_flags = st.checkbox("Lọc mặt nạ tia vũ trụ (FLAGS)", value=True,
                            help="Gán NaN cho pixel dính cờ lỗi phần cứng / tia vũ trụ (bitwise AND). "
                                 "Bỏ qua cờ SOURCE nên các ngôi sao được giữ lại.")
    st.caption("Chỉ áp dụng cho dữ liệu SPHEREx thật.")

    st.divider()
    st.subheader("🔬 Tham số vật lý")
    dai_pho = st.selectbox("Dải bước sóng SPHEREx", [
        "Band 1 (0.75 – 1.09 µm)", "Band 2 (1.10 – 1.62 µm)", "Band 3 (1.63 – 2.41 µm)",
        "Band 4 (2.42 – 3.82 µm)", "Band 5 (3.83 – 4.41 µm)", "Band 6 (4.42 – 5.00 µm)"], index=0)
    # Mặc định riêng cho dữ liệu SPHEREx thật có sẵn: 8σ và 45 px; còn lại: 5σ và 100 px
    mac_dinh_that = che_do == SRC_DEMO and co_demo
    nguong_sigma = st.slider("Ngưỡng lọc nhiễu (σ)", 1.0, 10.0, 8.0 if mac_dinh_that else 5.0, 0.5,
                             help="Chỉ giữ các đốm trên ảnh trừ sáng hơn k lần độ nhiễu nền")
    st.caption("Ngưỡng thấp → bắt được vật thể mờ nhưng nhiều báo động giả. "
               "Mặc định 8σ cho dữ liệu thật, 5σ cho dữ liệu giả lập.")
    ban_kinh = st.slider("Bán kính ghép cặp tối đa (pixel)", 20, 300, 45 if mac_dinh_that else 100, 5,
                         help="Một đốm ở T1 chỉ được ghép với đốm ở T2 nằm trong bán kính này "
                              f"(và cách tối thiểu {MIN_SHIFT_PX:g} px)")
    st.caption(f"{ban_kinh} px ≈ {ban_kinh * PIXEL_SCALE / 60:.1f} phút góc.")

# =============================================================================
# 5. NẠP DỮ LIỆU & CHẠY PHÁT HIỆN
# =============================================================================
dang_mock, wcs, valid, info = True, None, None, None

raw1 = raw2 = None
if che_do == SRC_DEMO and co_demo:
    raw1, raw2 = DEMO_T1.read_bytes(), DEMO_T2.read_bytes()
elif che_do == SRC_UPLOAD and file_1 and file_2:
    raw1, raw2 = file_1.getvalue(), file_2.getvalue()

if raw1 is not None:
    try:
        img1, img2, valid, info, header1 = prepare_real(raw1, raw2, tru_zodi, loc_flags)
        wcs = _celestial_wcs(header1)
        dang_mock = False
    except ImportError:
        st.error("Chưa cài astropy. Chạy `pip install astropy reproject` để đọc file FITS. "
                 "Đang hiển thị dữ liệu giả lập.")
    except Exception as e:
        st.error(f"Không xử lý được file FITS: {e} Đang hiển thị dữ liệu giả lập.")

if dang_mock:
    if che_do == SRC_MOCK:
        img1, img2 = generate_mock_data(int(seed), int(n_movers))
    else:
        img1, img2 = generate_mock_data()
        if che_do == SRC_UPLOAD and not (file_1 and file_2):
            st.info("Hãy upload đủ 2 file T1 và T2. Trong lúc chờ, giao diện hiển thị dữ liệu giả lập.")

try:
    diff, sigma, cands, n_dipole = detect(img1, img2, float(nguong_sigma), float(ban_kinh), valid)
except ValueError as e:
    st.error(str(e))
    st.stop()

movers = [c for c in cands if c["loai"] == "Di chuyển"]
for i, c in enumerate(cands, 1):
    c["id"] = f"Cand-X{i}"

# Top-N để vẽ: ưu tiên vật thể "Di chuyển", còn chỗ thì thêm ứng viên khác; đều theo snr giảm dần
movers_ve = movers[:TOP_N]
cands_ve = movers_ve + [c for c in cands if c["loai"] != "Di chuyển"][:TOP_N - len(movers_ve)]


def pix2sky(x, y):
    """Pixel -> (RA, Dec) độ. Mock dùng WCS tuyến tính giả định quanh (180.25, 45.10)."""
    if wcs is not None:
        ra, dec = wcs.all_pix2world(x, y, 0)
        return float(ra), float(dec)
    cy, cx = (np.array(img1.shape) - 1) / 2
    dec = 45.10 + (y - cy) * PIXEL_SCALE / 3600
    ra = 180.25 - (x - cx) * PIXEL_SCALE / 3600 / np.cos(np.radians(45.10))
    return float(ra), float(dec)


# =============================================================================
# 6. TIÊU ĐỀ & THẺ SỐ LIỆU
# =============================================================================
badge = '<span class="badge mock">DỮ LIỆU GIẢ LẬP</span>' if dang_mock else '<span class="badge real">DỮ LIỆU THẬT</span>'
st.markdown(
    f'<div class="hero"><h1>🌌 SPHEREx Sky Tracker</h1>'
    f'<p>So sánh hai thời điểm quan sát để phát hiện thiên thể thay đổi vị trí '
    f'<span class="badge">Team 10LY2NHUEr</span>{badge}</p></div>',
    unsafe_allow_html=True,
)

# Trạng thái tiền xử lý (chỉ với dữ liệu thật)
if info is not None:
    zodi_txt = "đã trừ" if info["zodi"] else ("tắt" if not tru_zodi else "file thiếu HDU ZODI")
    if info["flags_pct"] is not None:
        flags_txt = (f'che {info["flags_pct"][0]:.1f}% (T1) / {info["flags_pct"][1]:.1f}% (T2) pixel, '
                     f'{info["flags_src"]}, giữ SOURCE')
    else:
        flags_txt = "tắt" if not loc_flags else "file thiếu HDU FLAGS"
    wcs_txt = {"ok": "đã chiếu T2 → lưới T1", "no_wcs": "không có WCS, trừ thẳng theo pixel",
               "no_reproject": "chưa cài reproject, trừ thẳng theo pixel"}[info["wcs"]]
    st.caption(f'**ZODI:** {zodi_txt} · **FLAGS:** {flags_txt} · **WCS:** {wcs_txt} · '
               f'**Vùng hợp lệ:** {info["valid_pct"]:.1f}% ảnh')
    if "scale_raw" in info:
        scale_txt = f'T2/T1 = {info["scale"]:.3f} (trung vị trên {info["n_bright"]} sao sáng chung)'
    else:
        scale_txt = f'không chỉnh (chỉ có {info["n_bright"]} sao sáng chung)'
    if info["edge"] is None:
        mask_txt = "bỏ qua (chưa cài scipy)"
    else:
        mask_txt = (f'gọt biên {EDGE_PX} px, che hào quang {info["n_sat"]} sao cực sáng '
                    f'({info["halo_pct"]:.1f}% ảnh)')
    st.caption(f'**Tỷ lệ quang thông:** {scale_txt} · **Mặt nạ:** {mask_txt}')
    if info["halo_pct"] > 25:
        st.warning(f'Mặt nạ hào quang đang che {info["halo_pct"]:.0f}% ảnh. Nếu đang tắt lọc FLAGS, các tia vũ trụ '
                   'chưa bị che sẽ bị coi nhầm là tâm sao cực sáng - hãy bật lại lọc FLAGS.')
    if info["edge"] is None:
        st.warning("Chưa cài `scipy` nên CHƯA gọt biên và che hào quang sao sáng. Chạy `pip install scipy`.")
    if "scale_raw" in info and not (SCALE_CLIP[0] < info["scale_raw"] < SCALE_CLIP[1]):
        st.warning(f'Tỷ lệ quang thông đo được là {info["scale_raw"]:.2f}, nằm ngoài khoảng '
                   f'{SCALE_CLIP[0]}–{SCALE_CLIP[1]} nên đã bị cắt về biên. Hai ảnh có thể chụp ở '
                   'bước sóng quá khác nhau để so sánh trực tiếp.')
    if info["wcs"] == "no_reproject":
        st.warning("Chưa cài thư viện `reproject` nên hai ảnh CHƯA được đồng bộ WCS. Chạy `pip install reproject`.")
    elif info["wcs"] == "no_wcs":
        st.warning("Ít nhất một file không có WCS hợp lệ nên hai ảnh CHƯA được đồng bộ tọa độ.")
    if info["flags_pct"] is not None and max(info["flags_pct"]) > 10:
        st.warning("Mặt nạ FLAGS đang che hơn 10% ảnh dù đã bỏ qua cờ SOURCE. "
                   "Hãy kiểm tra định nghĩa bit trong header FLAGS của file.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vật thể di chuyển", len(movers))
m2.metric("Nguồn chỉ đổi độ sáng", len(cands) - len(movers))
m3.metric("Nhiễu ảnh trừ (1σ)", f"{sigma:.3g}")
m4.metric("Dịch chuyển lớn nhất",
          f'{max((c["shift_px"] for c in movers), default=0) * PIXEL_SCALE:.0f}″')


# =============================================================================
# 7. HÀM VẼ
# =============================================================================
def show_sky(ax, img, title):
    vmin, vmax = np.nanpercentile(img, [5, 99.7])
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
    tieu_de_t2 = "T2 · Đã chiếu về lưới T1" if info["wcs"] == "ok" else "T2 · Ảnh sau"

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
        ax.imshow(diff - np.nanmedian(diff), cmap=CMAP_DIFF, origin="lower", vmin=-lim, vmax=lim)
        ax.set_title("T2 − T1 · Ứng viên", color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        for c in cands_ve:
            if c["loai"] == "Di chuyển":
                ax.add_patch(plt.Circle((c["x1"], c["y1"]), R, color=C_T1, fill=False, lw=1.2, ls="--"))
                ax.add_patch(plt.Circle((c["x2"], c["y2"]), R, color=C_HIT, fill=False, lw=2))
                ax.annotate("", xy=(c["x2"], c["y2"]), xytext=(c["x1"], c["y1"]),
                            arrowprops=dict(arrowstyle="->", color=C_HIT, lw=1, alpha=.8,
                                            shrinkA=R * 2.2, shrinkB=R * 2.2))
                ax.text(c["x2"], c["y2"] + R * 1.4, c["id"], color=C_HIT, ha="center", fontsize=7)
            else:
                x = c["x2"] if not np.isnan(c["x2"]) else c["x1"]
                y = c["y2"] if not np.isnan(c["y2"]) else c["y1"]
                ax.add_patch(plt.Rectangle((x - R, y - R), 2 * R, 2 * R, color=C_VAR, fill=False, lw=1.2))
                ax.text(x, y + R * 1.4, c["id"], color=C_VAR, ha="center", fontsize=7)
        render(fig)

    st.markdown(
        f'<div class="legend">'
        f'<span><i class="dot" style="background:{C_T1}"></i>Vị trí ở T1 (đốm âm)</span>'
        f'<span><i class="dot" style="background:{C_HIT}"></i>Vị trí ở T2 – ứng viên di chuyển</span>'
        f'<span><i class="dot" style="background:{C_VAR}"></i>Chỉ đổi độ sáng, không di chuyển</span>'
        f'</div>', unsafe_allow_html=True)
    if len(cands) > len(cands_ve):
        st.caption(f"Biểu đồ chỉ vẽ {len(cands_ve)}/{len(cands)} ứng viên có độ tin cậy cao nhất "
                   "(ưu tiên vật thể di chuyển). Danh sách đầy đủ nằm ở tab Dữ liệu ứng viên.")

    loc_txt = f" Đã loại **{n_dipole}** dư ảnh sao tĩnh (dịch chuyển < {MIN_SHIFT_PX:g} px)." if n_dipole else ""
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
        vmin, vmax = np.nanpercentile(img1, [5, 99.7])
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
        st.caption("Tọa độ RA/Dec dưới đây tính từ một WCS giả định cho dữ liệu mock, không ứng với bầu trời thật.")
    elif wcs is None:
        st.caption("File FITS không có WCS hợp lệ nên RA/Dec dùng WCS giả định – chỉ tin cột pixel.")
    else:
        st.caption("RA/Dec tính từ WCS của ảnh T1.")

    rows = []
    for c in cands:
        x = c["x2"] if not np.isnan(c["x2"]) else c["x1"]
        y = c["y2"] if not np.isnan(c["y2"]) else c["y1"]
        ra, dec = pix2sky(x, y)
        rows.append({
            "Mã ứng viên": c["id"], "Phân loại": c["loai"],
            "RA (độ)": round(ra, 5), "Dec (độ)": round(dec, 5),
            "x (px)": round(x, 1), "y (px)": round(y, 1),
            "Độ dịch chuyển (arcsec)": round(c["shift_px"] * PIXEL_SCALE, 1),
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
