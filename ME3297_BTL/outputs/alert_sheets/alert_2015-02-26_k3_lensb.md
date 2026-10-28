# Phiếu cảnh báo đầu ca - 2015-02-26 (Thursday)

- Mô hình: Random Forest (kịch bản A, siêu tham số chọn trên train), huấn luyện trên 47 ngày (943 dòng) trước ngày này.
- Xếp hạng: top-k theo điểm rủi ro của mô hình (p chưa hiệu chỉnh); lăng kính (b) giao hàng (sản phẩm thiếu); k = 3 tổ trong tổng 21 tổ-bộ phận hôm nay.

| Hạng | Tổ | Bộ phận | Điểm rủi ro (dùng để xếp hạng) | Xác suất trượt ước tính (đã hiệu chỉnh) | Thiệt hại kỳ vọng (sản phẩm) | Lý do chính | Kết quả thực tế (chỉ để kiểm tra) |
|---|---|---|---|---|---|---|---|
| 1 | 6 | finishing | 0.83 | 75% | 747 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 2.9, trung vị train 15.3; tỷ lệ trượt 5 ngày trước = 0.8, trung vị train 0.2 | TRƯỢT (0.27 so với định mức 0.75) |
| 2 | 9 | finishing | 0.83 | 75% | 744 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 2.9, trung vị train 15.3; tỷ lệ trượt 5 ngày trước = 0.8, trung vị train 0.2 | TRƯỢT (0.41 so với định mức 0.75) |
| 3 | 2 | finishing | 0.81 | 71% | 490 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 3.9, trung vị train 15.3; tỷ lệ trượt 5 ngày trước = 0.8, trung vị train 0.2 | TRƯỢT (0.56 so với định mức 0.70) |

Kiểm tra (không dùng khi chấm điểm): 3/3 tổ được cảnh báo thực sự trượt; cả ngày có 8/21 tổ-bộ phận trượt.

Ghi chú: điểm rủi ro là đầu ra thô của mô hình, chỉ dùng để xếp hạng; nó cao hơn xác suất thật vì mô hình được huấn luyện với trọng số cân bằng lớp. Xác suất trượt ước tính là điểm này sau hiệu chỉnh sigmoid (fit trên điểm out-of-fold của các ngày < D), đọc được theo nghĩa đen. Thiệt hại kỳ vọng = xác suất trượt ước tính x quy mô mục tiêu của tổ (targeted x số công nhân x 480 / smv); không phải ước tính tiền. Khi trượt, năng suất thường thấp hơn định mức khoảng 25% (trung bình lịch sử), nên thiệt hại thực thấp hơn quy mô mục tiêu. Lý do chính tính bằng p(x) - p(x với đặc trưng đặt về trung vị train); chỉ liệt kê yếu tố làm tăng rủi ro.
