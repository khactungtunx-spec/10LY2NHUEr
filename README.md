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
