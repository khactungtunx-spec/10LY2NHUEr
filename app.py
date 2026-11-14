# -*- coding: utf-8 -*-
"""
SPHEREx Sky Tracker - MVP giao diện Streamlit (NASA Space Apps)
Chạy:  streamlit run app.py --theme.base dark
Cần:   streamlit, numpy, pandas, matplotlib   (astropy: tùy chọn, để đọc file FITS thật)
"""
import time
from collections import deque

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
# 2. DỮ LIỆU: GIẢ LẬP & ĐỌC FITS
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


def load_fits(uploaded):
    """Đọc mảng 2D đầu tiên trong file FITS. Trả về (ảnh, wcs hoặc None)."""
    from astropy.io import fits  # import trễ: chỉ cần khi dùng dữ liệu thật

    with fits.open(uploaded) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.ndim(hdu.data) == 2:
                data = np.asarray(hdu.data, dtype=float)
                data = np.where(np.isfinite(data), data, np.nanmedian(data))
                try:
                    from astropy.wcs import WCS
                    wcs = WCS(hdu.header)
                    wcs = wcs if wcs.has_celestial else None
                except Exception:
                    wcs = None
                return data, wcs
    raise ValueError("Không tìm thấy ảnh 2D trong file FITS.")


# =============================================================================
# 3. THUẬT TOÁN PHÁT HIỆN (chạy thật trên ảnh trừ, không hard-code)
# =============================================================================
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
    return blobs


@st.cache_data(show_spinner=False)
def detect(img_t1, img_t2, k_sigma):
    """Trừ ảnh -> ước lượng nhiễu -> lọc ngưỡng -> ghép cặp đốm âm (T1) với đốm dương (T2)."""
    diff = img_t2 - img_t1
    med = np.median(diff)
    sigma = 1.4826 * np.median(np.abs(diff - med))  # độ lệch chuẩn bền vững (MAD)
    sigma = sigma if sigma > 0 else 1.0
    d = diff - med

    neg = _find_blobs(d < -k_sigma * sigma, d)  # vật thể có ở T1, mất ở T2
    pos = _find_blobs(d > k_sigma * sigma, d)   # vật thể mới xuất hiện ở T2

    # Ghép cặp tham lam theo độ giống nhau về thông lượng (chênh tối đa 2 lần)
    pairs = sorted(
        (abs(np.log(p["flux"] / n["flux"])), i, j)
        for i, n in enumerate(neg) for j, p in enumerate(pos)
    )
    used_n, used_p, rows = set(), set(), []
    for cost, i, j in pairs:
        if cost > np.log(2) or i in used_n or j in used_p:
            continue
        used_n.add(i); used_p.add(j)
        n, p = neg[i], pos[j]
        rows.append({"loai": "Di chuyển", "x1": n["x"], "y1": n["y"], "x2": p["x"], "y2": p["y"],
                     "shift_px": float(np.hypot(p["x"] - n["x"], p["y"] - n["y"])),
                     "dflux": (p["flux"] / n["flux"] - 1) * 100,
                     "snr": min(n["peak"], p["peak"]) / sigma})
    for j, p in enumerate(pos):
        if j not in used_p:
            rows.append({"loai": "Sáng lên / mới", "x1": np.nan, "y1": np.nan, "x2": p["x"], "y2": p["y"],
                         "shift_px": 0.0, "dflux": np.nan, "snr": p["peak"] / sigma})
    for i, n in enumerate(neg):
        if i not in used_n:
            rows.append({"loai": "Mờ đi / mất", "x1": n["x"], "y1": n["y"], "x2": np.nan, "y2": np.nan,
                         "shift_px": 0.0, "dflux": np.nan, "snr": n["peak"] / sigma})
    rows.sort(key=lambda r: -r["snr"])
    return diff, float(sigma), rows


# =============================================================================
# 4. SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Điều khiển đầu vào")
    che_do = st.radio("Chế độ dữ liệu", ["Giả lập (Mock)", "Dữ liệu SPHEREx thật"])

    file_1 = file_2 = None
    if che_do == "Dữ liệu SPHEREx thật":
        file_1 = st.file_uploader("Ảnh T1 (.fits)", type=["fits", "fit", "fz"])
        file_2 = st.file_uploader("Ảnh T2 (.fits)", type=["fits", "fit", "fz"])
        st.caption("Hai ảnh phải cùng kích thước và đã căn chỉnh (align) về cùng một lưới pixel.")
    else:
        n_movers = st.slider("Số vật thể di chuyển", 1, 3, 3)
        seed = st.number_input("Seed bầu trời", 0, 9999, 42, help="Đổi seed để sinh bầu trời khác")

    st.divider()
    st.subheader("🔬 Tham số vật lý")
    dai_pho = st.selectbox("Dải bước sóng SPHEREx", [
        "Band 1 (0.75 – 1.09 µm)", "Band 2 (1.10 – 1.62 µm)", "Band 3 (1.63 – 2.41 µm)",
        "Band 4 (2.42 – 3.82 µm)", "Band 5 (3.83 – 4.41 µm)", "Band 6 (4.42 – 5.00 µm)"])
    nguong_sigma = st.slider("Ngưỡng lọc nhiễu (σ)", 1.0, 10.0, 5.0, 0.5,
                             help="Chỉ giữ các đốm trên ảnh trừ sáng hơn k lần độ nhiễu nền")
    st.caption("Ngưỡng thấp → bắt được vật thể mờ nhưng nhiều báo động giả. 5σ là mức thông dụng.")

# =============================================================================
# 5. NẠP DỮ LIỆU & CHẠY PHÁT HIỆN
# =============================================================================
dang_mock, wcs = True, None
if che_do == "Dữ liệu SPHEREx thật" and file_1 and file_2:
    try:
        img1, wcs = load_fits(file_1)
        img2, _ = load_fits(file_2)
        if img1.shape != img2.shape:
            raise ValueError(f"Kích thước hai ảnh khác nhau: {img1.shape} và {img2.shape}.")
        dang_mock = False
    except ImportError:
        st.error("Chưa cài astropy. Chạy `pip install astropy` để đọc file FITS. Đang hiển thị dữ liệu giả lập.")
    except Exception as e:
        st.error(f"Không đọc được file FITS: {e} Đang hiển thị dữ liệu giả lập.")

if dang_mock:
    if che_do == "Dữ liệu SPHEREx thật":
        img1, img2 = generate_mock_data()
        if not (file_1 and file_2):
            st.info("Hãy upload đủ 2 file T1 và T2. Trong lúc chờ, giao diện hiển thị dữ liệu giả lập.")
    else:
        img1, img2 = generate_mock_data(int(seed), int(n_movers))

try:
    diff, sigma, cands = detect(img1, img2, float(nguong_sigma))
except ValueError as e:
    st.error(str(e))
    st.stop()

movers = [c for c in cands if c["loai"] == "Di chuyển"]
for i, c in enumerate(cands, 1):
    c["id"] = f"Cand-X{i}"


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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Vật thể di chuyển", len(movers))
m2.metric("Nguồn chỉ đổi độ sáng", len(cands) - len(movers))
m3.metric("Nhiễu ảnh trừ (1σ)", f"{sigma:.2f}")
m4.metric("Dịch chuyển lớn nhất",
          f'{max((c["shift_px"] for c in movers), default=0) * PIXEL_SCALE:.0f}″')


# =============================================================================
# 7. HÀM VẼ
# =============================================================================
def show_sky(ax, img, title):
    vmin, vmax = np.percentile(img, [5, 99.7])
    ax.imshow(img, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")


def render(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # tránh rò bộ nhớ sau nhiều lần rerun


R = max(4.0, img1.shape[0] / 22)  # bán kính vòng khoanh theo cỡ ảnh

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
        for c in movers:
            ax.add_patch(plt.Circle((c["x1"], c["y1"]), R, color=C_T1, fill=False, lw=1.2, ls="--"))
        render(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        show_sky(ax, img2, "T2 · Sau 6 tháng")
        for c in movers:
            ax.add_patch(plt.Circle((c["x2"], c["y2"]), R, color=C_T2, fill=False, lw=1.2))
        render(fig)

    with c3:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        lim = max(8 * sigma, float(nguong_sigma) * sigma * 1.5)
        ax.imshow(diff - np.median(diff), cmap=CMAP_DIFF, origin="lower", vmin=-lim, vmax=lim)
        ax.set_title("T2 − T1 · Ứng viên", color=TEXT, fontsize=10, pad=8)
        ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
        for c in cands:
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

    if movers:
        st.success(f"Lọc ở ngưỡng {nguong_sigma}σ: tìm thấy **{len(movers)}** vật thể di chuyển. "
                   "Các sao nền đứng yên đã triệt tiêu trên ảnh trừ.")
    else:
        st.warning(f"Không có vật thể di chuyển nào vượt ngưỡng {nguong_sigma}σ. Thử hạ ngưỡng xuống.")

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
        vmin, vmax = np.percentile(img1, [5, 99.7])
        ax.imshow(img1 * (1 - t) + img2 * t if p in (0, 100) else img1,
                  cmap="gray", origin="lower", vmin=vmin, vmax=vmax, alpha=.45)
        for c in movers:
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
        st.info("Chưa có vật thể di chuyển nào để nội suy. Hạ ngưỡng σ ở thanh bên.")
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
