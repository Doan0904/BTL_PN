# Phiếu cảnh báo đầu ca - 2015-03-11 (Wednesday)

- Mô hình: Random Forest (kịch bản A, siêu tham số chọn trên train), huấn luyện trên 58 ngày (1173 dòng) trước ngày này.
- Xếp hạng: top-k theo điểm rủi ro của mô hình (p chưa hiệu chỉnh); lăng kính (b) giao hàng (sản phẩm thiếu); k = 3 tổ trong tổng 24 tổ-bộ phận hôm nay.

| Hạng | Tổ | Bộ phận | Điểm rủi ro (dùng để xếp hạng) | Xác suất trượt ước tính (đã hiệu chỉnh) | Thiệt hại kỳ vọng (sản phẩm) | Lý do chính | Kết quả thực tế (chỉ để kiểm tra) |
|---|---|---|---|---|---|---|---|
| 1 | 8 | finishing | 0.85 | 78% | 541 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 3.9, trung vị train 15.3; chênh lệch TB 5 ngày trước = -0.131, trung vị train 0 | TRƯỢT (0.63 so với định mức 0.70) |
| 2 | 10 | finishing | 0.85 | 78% | 775 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 2.9, trung vị train 15.3; tỷ lệ trượt 5 ngày trước = 1, trung vị train 0.2 | TRƯỢT (0.63 so với định mức 0.75) |
| 3 | 6 | finishing | 0.84 | 77% | 533 | số công nhân = 6, trung vị train 34; SMV (thời gian chuẩn) = 2.9, trung vị train 15.3; tỷ lệ trượt 5 ngày trước = 0.8, trung vị train 0.2 | TRƯỢT (0.39 so với định mức 0.70) |

Kiểm tra (không dùng khi chấm điểm): 3/3 tổ được cảnh báo thực sự trượt; cả ngày có 6/24 tổ-bộ phận trượt.

Ghi chú: điểm rủi ro là đầu ra thô của mô hình, chỉ dùng để xếp hạng; nó cao hơn xác suất thật vì mô hình được huấn luyện với trọng số cân bằng lớp. Xác suất trượt ước tính là điểm này sau hiệu chỉnh sigmoid (fit trên điểm out-of-fold của các ngày < D), đọc được theo nghĩa đen. Thiệt hại kỳ vọng = xác suất trượt ước tính x quy mô mục tiêu của tổ (targeted x số công nhân x 480 / smv); không phải ước tính tiền. Khi trượt, năng suất thường thấp hơn định mức khoảng 26% (trung bình lịch sử), nên thiệt hại thực thấp hơn quy mô mục tiêu. Lý do chính tính bằng p(x) - p(x với đặc trưng đặt về trung vị train); chỉ liệt kê yếu tố làm tăng rủi ro.
