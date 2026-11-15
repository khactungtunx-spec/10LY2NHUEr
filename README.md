# Dự án NASA SPHEREx - Team 10LY2NHUEr
# SPHEREx Sky Tracker (Team 10LY2NHUEr)

Pipeline phân tích dữ liệu hồng ngoại **SPHEREx Level 2** tự động phát hiện vật thể di chuyển (tiểu hành tinh) và nguồn biến quang giữa hai thời điểm quan sát ($T_1, T_2$).

## Kiến trúc xử lý chính
1. **Tiền xử lý SPHEREx Level 2:** Khử nền ánh sáng hoàng đạo (`ZODI`) và lọc mặt nạ tia vũ trụ (`FLAGS` bit 0–19).
2. **Căn chỉnh hệ tọa độ (2-Step WCS Alignment):** Chiếu lại ảnh $T_2$ sang lưới WCS của $T_1$ bằng `reproject`, kết hợp tinh chỉnh dịch chuyển dưới mức pixel (sub-pixel) và cân bằng quang thông trung vị.
3. **Bộ lọc hình thái học 4 lớp (Mặc định $9.0\sigma$):** Loại bỏ vệt vệ tinh (chuỗi thẳng hàng), đốm dẹt/kéo dài, gai nhiễu dưới 3 pixel, lưỡng cực dư và khớp PSF thực tế.
4. **Đối chiếu danh mục thiên văn (Parallel SIMBAD):** Truy vấn song song qua `astroquery.simbad` (`simbad.harvard.edu`, bán kính $30''$) để tách sao cố định/nguồn bức xạ đã biết khỏi các ứng viên di chuyển chưa định danh.

## Cài đặt & Chạy cục bộ
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nguồn dữ liệu & Ghi công (Data Attribution)
- **SPHEREx (NASA / JPL-Caltech):**
  > "Ấn phẩm này sử dụng dữ liệu từ Kính viễn vọng Quang phổ Kế hoạch Khám phá Lịch sử Vũ trụ, Kỷ nguyên Tái ion hóa và Băng (SPHEREx), một dự án hợp tác giữa Phòng thí nghiệm Phản lực (Jet Propulsion Laboratory) và Viện Công nghệ California (Caltech), và được tài trợ bởi Cơ quan Hàng không và Vũ trụ Quốc gia (NASA)."
  >
  > *"This publication makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration."*
- **Kho lưu trữ & Danh mục đối chiếu:** Dữ liệu Level 2 khai thác qua **NASA/IPAC Infrared Science Archive (IRSA)**; đối chiếu tọa độ thiên thể qua cơ sở dữ liệu **SIMBAD** (CDS, Strasbourg, Pháp).





# SPHEREx Sky Tracker (Team 10LY2NHUEr)

An automated infrared image-differencing pipeline for **SPHEREx Level 2** data products, built to detect moving celestial objects (asteroids) and variable sources across multi-epoch observations ($T_1, T_2$).

## Key Processing Architecture
1. **SPHEREx Level 2 Pre-processing:** Subtracts zodiacal background emission (`ZODI`) and masks cosmic-ray hits and detector artifacts using quality bitmasks (`FLAGS` bits 0–19).
2. **Two-Step Sub-Pixel WCS Alignment:** Reprojects epoch $T_2$ onto the $T_1$ World Coordinate System (WCS) grid via `reproject`, paired with sub-pixel registration refinement and median aperture flux normalization.
3. **Four-Layer Morphological & PSF Filtering (Default $9.0\sigma$):** Suppresses linear satellite trails, flat/elongated artifacts, sub-3-pixel noise spikes, and subtraction dipoles while validating candidates against the empirical Point Spread Function (PSF).
4. **Parallel Catalog Cross-Matching (SIMBAD):** Queries `simbad.harvard.edu` concurrently via `astroquery.simbad` ($30''$ search radius) to separate known static stars and cataloged sources from uncataloged moving candidates.

## Local Installation & Usage
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Attribution & Acknowledgements
- **SPHEREx Mission (NASA / JPL-Caltech):**
  > "This publication makes use of data products from the Spectro-Photometer for the History of the Universe, Epoch of Reionization and Ices Explorer (SPHEREx), which is a joint project of the Jet Propulsion Laboratory and the California Institute of Technology, and is funded by the National Aeronautics and Space Administration."
- **Archive & Astronomical Catalogs:** Level 2 FITS data products accessed via the **NASA/IPAC Infrared Science Archive (IRSA)**; astrometric cross-matching powered by the **SIMBAD** database, operated at CDS, Strasbourg, France.


*## AI Usage Disclosure
Generative AI assistants (Google Gemini, Anthropic Claude, Antigravity IDE) were used solely to assist with Python/Streamlit debugging, UI/CSS styling, and documentation refinement. All astrophysics algorithms, data processing logic, and candidate validations were developed and verified by Team 10LY2NHUEr.
