# Tóm tắt kết quả (sinh tự động bởi `src/run_all.py` từ `outputs/tables/*.csv`)

Mọi con số dưới đây được đọc trực tiếp từ các bảng CSV; chạy lại `python src/run_all.py` sẽ ghi đè file này. Test chỉ được dùng để đánh giá; mọi lựa chọn (mô hình chính, siêu tham số, chính sách thiệt hại kỳ vọng, hiệu chỉnh) được thực hiện trên train/OOF.

## 1. Dữ liệu và cách chia

| Tập | Từ ngày | Đến ngày | Số ngày | Số dòng | Tỷ lệ miss |
|---|---|---|---|---|---|
| train | 2015-01-01 | 2015-02-25 | 47 | 943 | 0.269 |
| test | 2015-02-26 | 2015-03-11 | 12 | 254 | 0.268 |

Rolling-origin theo ngày trên train (5 fold):

| Fold | Huấn luyện | Validate | Dòng train | Dòng validate |
|---|---|---|---|---|
| 1 | 2015-01-01 đến 2015-01-10 | 2015-01-11 đến 2015-01-19 | 170 | 167 |
| 2 | 2015-01-01 đến 2015-01-19 | 2015-01-20 đến 2015-01-28 | 337 | 161 |
| 3 | 2015-01-01 đến 2015-01-28 | 2015-01-29 đến 2015-02-07 | 498 | 157 |
| 4 | 2015-01-01 đến 2015-02-07 | 2015-02-08 đến 2015-02-16 | 655 | 148 |
| 5 | 2015-01-01 đến 2015-02-16 | 2015-02-17 đến 2015-02-25 | 803 | 140 |

## 2. Kịch bản A (chính): CV rolling-origin (trung bình ± độ lệch chuẩn qua 5 fold) và test

| Mô hình | PR-AUC (CV) | ROC-AUC (CV) | F2 (CV) | Recall (CV) | Precision (CV) | PR-AUC (test) | ROC-AUC (test) | F2 (test) | Recall (test) | Precision (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 (lớp đa số) | 0.286 ± 0.083 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.268 | 0.500 | 0.000 | 0.000 | 0.000 |
| B1 (hôm qua trượt) | 0.443 ± 0.100 | 0.656 ± 0.055 | 0.488 ± 0.090 | 0.486 ± 0.092 | 0.496 ± 0.082 | 0.491 | 0.638 | 0.538 | 0.544 | 0.514 |
| B2 (mọi tổ finishing) | 0.455 ± 0.085 | 0.713 ± 0.055 | 0.528 ± 0.098 | 0.585 ± 0.128 | 0.401 ± 0.115 | 0.592 | 0.786 | 0.685 | 0.779 | 0.461 |
| Cây quyết định | 0.503 ± 0.090 | 0.712 ± 0.067 | 0.635 ± 0.106 | 0.700 ± 0.173 | 0.494 ± 0.070 | 0.609 | 0.799 | 0.652 | 0.809 | 0.367 |
| SVM (RBF) | 0.544 ± 0.103 | 0.756 ± 0.051 | 0.587 ± 0.098 | 0.636 ± 0.119 | 0.453 ± 0.061 | 0.546 | 0.773 | 0.641 | 0.721 | 0.445 |
| Mạng nơ-ron (MLP) | 0.485 ± 0.069 | 0.718 ± 0.041 | 0.558 ± 0.097 | 0.633 ± 0.225 | 0.458 ± 0.104 | 0.569 | 0.730 | 0.539 | 0.529 | 0.581 |
| Hồi quy logistic | 0.493 ± 0.100 | 0.727 ± 0.025 | 0.558 ± 0.072 | 0.588 ± 0.094 | 0.473 ± 0.050 | 0.583 | 0.805 | 0.714 | 0.794 | 0.509 |
| Random Forest | 0.574 ± 0.123 | 0.766 ± 0.045 | 0.580 ± 0.132 | 0.623 ± 0.170 | 0.473 ± 0.075 | 0.690 | 0.822 | 0.702 | 0.809 | 0.458 |

F2/Recall/Precision tính tại ngưỡng mặc định của `predict()` (MLP không có class_weight nên recall thấp).

## 3. Kịch bản B (độ nhạy): CV rolling-origin và test

| Mô hình | PR-AUC (CV) | ROC-AUC (CV) | F2 (CV) | Recall (CV) | Precision (CV) | PR-AUC (test) | ROC-AUC (test) | F2 (test) | Recall (test) | Precision (test) |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 (lớp đa số) | 0.286 ± 0.083 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.268 | 0.500 | 0.000 | 0.000 | 0.000 |
| B1 (hôm qua trượt) | 0.443 ± 0.100 | 0.656 ± 0.055 | 0.488 ± 0.090 | 0.486 ± 0.092 | 0.496 ± 0.082 | 0.491 | 0.638 | 0.538 | 0.544 | 0.514 |
| B2 (mọi tổ finishing) | 0.455 ± 0.085 | 0.713 ± 0.055 | 0.528 ± 0.098 | 0.585 ± 0.128 | 0.401 ± 0.115 | 0.592 | 0.786 | 0.685 | 0.779 | 0.461 |
| Cây quyết định | 0.486 ± 0.100 | 0.737 ± 0.032 | 0.611 ± 0.121 | 0.675 ± 0.165 | 0.459 ± 0.063 | 0.535 | 0.790 | 0.702 | 0.838 | 0.425 |
| SVM (RBF) | 0.566 ± 0.101 | 0.776 ± 0.028 | 0.612 ± 0.088 | 0.664 ± 0.140 | 0.488 ± 0.056 | 0.546 | 0.804 | 0.691 | 0.765 | 0.500 |
| Mạng nơ-ron (MLP) | 0.495 ± 0.101 | 0.719 ± 0.058 | 0.245 ± 0.079 | 0.217 ± 0.071 | 0.530 ± 0.155 | 0.572 | 0.776 | 0.489 | 0.471 | 0.582 |
| Hồi quy logistic | 0.579 ± 0.085 | 0.770 ± 0.042 | 0.598 ± 0.126 | 0.635 ± 0.158 | 0.497 ± 0.062 | 0.597 | 0.797 | 0.737 | 0.809 | 0.545 |
| Random Forest | 0.590 ± 0.127 | 0.793 ± 0.030 | 0.621 ± 0.120 | 0.658 ± 0.148 | 0.525 ± 0.101 | 0.611 | 0.824 | 0.748 | 0.838 | 0.523 |

F2/Recall/Precision tính tại ngưỡng mặc định của `predict()` (MLP không có class_weight nên recall thấp).

## 4. Thí nghiệm rò rỉ: k-fold ngẫu nhiên 5x3 so với rolling-origin (cùng mô hình, cùng siêu tham số)


Kịch bản A:

| Mô hình | PR-AUC ngẫu nhiên | PR-AUC rolling | Chênh | ROC-AUC ngẫu nhiên | ROC-AUC rolling | Chênh |
|---|---|---|---|---|---|---|
| Cây quyết định | 0.531 ± 0.052 | 0.503 ± 0.090 | +0.028 | 0.758 ± 0.030 | 0.712 ± 0.067 | +0.046 |
| SVM (RBF) | 0.571 ± 0.075 | 0.544 ± 0.103 | +0.028 | 0.790 ± 0.039 | 0.756 ± 0.051 | +0.033 |
| Mạng nơ-ron (MLP) | 0.512 ± 0.077 | 0.485 ± 0.069 | +0.026 | 0.743 ± 0.045 | 0.718 ± 0.041 | +0.025 |
| Hồi quy logistic | 0.557 ± 0.058 | 0.493 ± 0.100 | +0.064 | 0.789 ± 0.028 | 0.727 ± 0.025 | +0.062 |
| Random Forest | 0.619 ± 0.059 | 0.574 ± 0.123 | +0.044 | 0.801 ± 0.040 | 0.766 ± 0.045 | +0.035 |

Kịch bản B:

| Mô hình | PR-AUC ngẫu nhiên | PR-AUC rolling | Chênh | ROC-AUC ngẫu nhiên | ROC-AUC rolling | Chênh |
|---|---|---|---|---|---|---|
| Cây quyết định | 0.577 ± 0.053 | 0.486 ± 0.100 | +0.091 | 0.812 ± 0.034 | 0.737 ± 0.032 | +0.075 |
| SVM (RBF) | 0.624 ± 0.070 | 0.566 ± 0.101 | +0.058 | 0.835 ± 0.035 | 0.776 ± 0.028 | +0.059 |
| Mạng nơ-ron (MLP) | 0.540 ± 0.092 | 0.495 ± 0.101 | +0.045 | 0.774 ± 0.058 | 0.719 ± 0.058 | +0.055 |
| Hồi quy logistic | 0.615 ± 0.071 | 0.579 ± 0.085 | +0.036 | 0.832 ± 0.028 | 0.770 ± 0.042 | +0.061 |
| Random Forest | 0.681 ± 0.050 | 0.590 ± 0.127 | +0.092 | 0.850 ± 0.029 | 0.793 ± 0.030 | +0.057 |

## 5. Kịch bản A so với B (PR-AUC)

| Mô hình | CV A | CV B | B - A | Test A | Test B | B - A |
|---|---|---|---|---|---|---|
| Cây quyết định | 0.503 | 0.486 | -0.017 | 0.609 | 0.535 | -0.075 |
| SVM (RBF) | 0.544 | 0.566 | +0.023 | 0.546 | 0.546 | +0.001 |
| Mạng nơ-ron (MLP) | 0.485 | 0.495 | +0.009 | 0.569 | 0.572 | +0.003 |
| Hồi quy logistic | 0.493 | 0.579 | +0.086 | 0.583 | 0.597 | +0.014 |
| Random Forest | 0.574 | 0.590 | +0.015 | 0.690 | 0.611 | -0.079 |

## 6. Mô hình chính và siêu tham số

Mô hình chính = PR-AUC CV (rolling-origin, kịch bản A) trung bình cao nhất; k cố định = 3.

| Hạng | Mô hình | PR-AUC CV | Mô hình chính | Mô hình cho thiệt hại kỳ vọng | Siêu tham số chọn trên train |
|---|---|---|---|---|---|
| 1 | Random Forest | 0.574 ± 0.123 | x | x | {'clf__max_depth': 4, 'clf__min_samples_leaf': 3} |
| 2 | SVM (RBF) | 0.544 ± 0.103 |  |  | {'clf__C': 0.1, 'clf__gamma': 'scale'} |
| 3 | Cây quyết định | 0.503 ± 0.090 |  |  | {'clf__max_depth': 4, 'clf__min_samples_leaf': 10} |
| 4 | Hồi quy logistic | 0.493 ± 0.100 |  |  | {'clf__C': 100.0} |
| 5 | Mạng nơ-ron (MLP) | 0.485 ± 0.069 |  |  | {'clf__alpha': 0.1, 'clf__hidden_layer_sizes': (8,)} |

Siêu tham số nằm ở biên lưới:

| Kịch bản | Mô hình | Tham số | Lưới | Giá trị chọn | Biên |
|---|---|---|---|---|---|
| A | SVM (RBF) | C | [0.1, 1.0, 10.0] | 0.1 | dưới |
| A | Mạng nơ-ron (MLP) | hidden_layer_sizes | [(8,), (16,), (32,), (64,), (16, 8)] | (8,) | dưới |
| A | Hồi quy logistic | C | [0.01, 0.1, 1.0, 10.0, 100.0] | 100.0 | trên |
| B | Mạng nơ-ron (MLP) | hidden_layer_sizes | [(8,), (16,), (32,), (64,), (16, 8)] | (64,) | trên |
| B | Mạng nơ-ron (MLP) | alpha | [0.001, 0.01, 0.1, 1.0] | 1.0 | trên |
| B | Random Forest | min_samples_leaf | [2, 3, 5, 10, 20] | 2 | dưới |

Thăm dò một nấc nữa vượt biên (chỉ để ghi chú, không đổi lựa chọn chính thức):

| Kịch bản | Mô hình | Tham số | Giá trị ở biên | PR-AUC CV | Giá trị vượt biên | PR-AUC CV | Chênh |
|---|---|---|---|---|---|---|---|
| A | SVM (RBF) | C | 0.1 | 0.544 | 0.01 | 0.550 | +0.007 |
| A | Mạng nơ-ron (MLP) | hidden_layer_sizes | (8,) | 0.485 | (4,) | 0.505 | +0.019 |
| A | Hồi quy logistic | C | 100.0 | 0.493 | 1000.0 | 0.494 | +0.001 |
| B | Mạng nơ-ron (MLP) | hidden_layer_sizes | (64,) | 0.495 | (128,) | 0.519 | +0.025 |
| B | Mạng nơ-ron (MLP) | alpha | 1.0 | 0.495 | 10.0 | 0.471 | -0.024 |
| B | Random Forest | min_samples_leaf | 2 | 0.590 | 1 | 0.588 | -0.002 |

Độ nhạy của kết quả test với việc phá hoà siêu tham số (Random Forest, hai cấu hình đứng đầu trên CV; chỉ để báo cáo):

| Hạng CV | Siêu tham số | PR-AUC CV | PR-AUC test | ROC-AUC test |
|---|---|---|---|---|
| 1 | {'clf__max_depth': 4, 'clf__min_samples_leaf': 3} | 0.5745 | 0.690 | 0.822 |
| 2 | {'clf__max_depth': 8, 'clf__min_samples_leaf': 2} | 0.5740 | 0.594 | 0.813 |

## 7. Độ tin cậy của xác suất và hiệu chỉnh sigmoid (kịch bản A)

Brier càng thấp càng tốt; Brier-hằng là Brier của dự báo "luôn bằng tỷ lệ trượt" của chính tập đó. Hiệu chỉnh sigmoid không rò rỉ: bộ hiệu chỉnh của khối validate j chỉ fit trên điểm OOF của các khối < j (nên chỉ có từ khối 2); test dùng bộ fit trên toàn bộ OOF. "Lệch rõ" = trung bình p lệch hơn 0,05 so với tỷ lệ trượt thực hoặc Brier tệ hơn dự báo hằng (xét trên OOF).

| Tập | Mô hình | Phiên bản | Tỷ lệ trượt thực | TB p | Brier | Brier-hằng | ECE | Độ dốc hiệu chỉnh | Lệch rõ |
|---|---|---|---|---|---|---|---|---|---|
| OOF khối 1-5 | Cây quyết định | chưa hiệu chỉnh | 0.282 | 0.386 | 0.2016 | 0.2025 | 0.156 | 0.29 | có |
| OOF khối 1-5 | Mạng nơ-ron (MLP) | chưa hiệu chỉnh | 0.282 | 0.485 | 0.2366 | 0.2025 | 0.203 | 0.64 | có |
| OOF khối 1-5 | Hồi quy logistic | chưa hiệu chỉnh | 0.282 | 0.370 | 0.2055 | 0.2025 | 0.122 | 0.36 | có |
| OOF khối 1-5 | Random Forest | chưa hiệu chỉnh | 0.282 | 0.425 | 0.1855 | 0.2025 | 0.143 | 1.17 | có |
| OOF khối 2-5 | Cây quyết định | chưa hiệu chỉnh | 0.289 | 0.385 | 0.2060 | 0.2054 | 0.137 | 0.27 | có |
| OOF khối 2-5 | Cây quyết định | sigmoid | 0.289 | 0.267 | 0.1848 | 0.2054 | 0.090 | 0.82 |  |
| OOF khối 2-5 | Mạng nơ-ron (MLP) | chưa hiệu chỉnh | 0.289 | 0.425 | 0.2000 | 0.2054 | 0.136 | 1.09 | có |
| OOF khối 2-5 | Mạng nơ-ron (MLP) | sigmoid | 0.289 | 0.155 | 0.2041 | 0.2054 | 0.134 | 0.81 |  |
| OOF khối 2-5 | Hồi quy logistic | chưa hiệu chỉnh | 0.289 | 0.406 | 0.2064 | 0.2054 | 0.127 | 0.49 | có |
| OOF khối 2-5 | Hồi quy logistic | sigmoid | 0.289 | 0.297 | 0.1817 | 0.2054 | 0.049 | 1.66 |  |
| OOF khối 2-5 | Random Forest | chưa hiệu chỉnh | 0.289 | 0.425 | 0.1875 | 0.2054 | 0.136 | 1.30 | có |
| OOF khối 2-5 | Random Forest | sigmoid | 0.289 | 0.247 | 0.1724 | 0.2054 | 0.063 | 1.18 |  |
| Test | Cây quyết định | chưa hiệu chỉnh | 0.268 | 0.497 | 0.2135 | 0.1960 | 0.237 | 0.71 | có |
| Test | Cây quyết định | sigmoid | 0.268 | 0.328 | 0.1633 | 0.1960 | 0.160 | 2.40 |  |
| Test | Mạng nơ-ron (MLP) | chưa hiệu chỉnh | 0.268 | 0.392 | 0.1801 | 0.1960 | 0.124 | 1.25 | có |
| Test | Mạng nơ-ron (MLP) | sigmoid | 0.268 | 0.229 | 0.1736 | 0.1960 | 0.078 | 1.96 |  |
| Test | Hồi quy logistic | chưa hiệu chỉnh | 0.268 | 0.464 | 0.1931 | 0.1960 | 0.196 | 0.86 | có |
| Test | Hồi quy logistic | sigmoid | 0.268 | 0.331 | 0.1639 | 0.1960 | 0.121 | 2.38 |  |
| Test | Random Forest | chưa hiệu chỉnh | 0.268 | 0.480 | 0.1930 | 0.1960 | 0.212 | 1.65 | có |
| Test | Random Forest | sigmoid | 0.268 | 0.332 | 0.1478 | 0.1960 | 0.066 | 1.42 |  |

Đồ thị: `outputs/figures/fig20_do_tin_cay_xac_suat.png`; các nhóm: `calibration_bins.csv`.

## 8. Chi phí, chính sách k = 3 và lợi ích ròng (hai lăng kính)

| Lăng kính | r cơ sở | C_int cơ sở | Dải độ nhạy C_int |
|---|---|---|---|
| (a) chi phí lao động (phút chuẩn) | 0.3 | 240.00 phút chuẩn/lần | 120.00 đến 480.00 |
| (b) giao hàng (sản phẩm thiếu) | 0.3 | 15.73 sản phẩm/lần | 7.86 đến 31.45 |

Lăng kính (b) là lăng kính chính cho khuyến nghị; C_int(b) = 240 / trung vị smv của train = 240 / 15.26. Lợi ích ròng/ngày = r x thiệt hại của các tổ trượt được cảnh báo đúng - C_int x số cảnh báo. Quy đổi BDT chỉ có ở lăng kính (a).

**Chọn chính sách trên OOF** cho Random Forest (k = 3, tiêu chí: lợi ích ròng/ngày ở lăng kính b):

| Lăng kính | Xếp hạng | Nguồn | Lợi ích ròng/ngày | Recall@3 | Precision@3 | Được chọn (theo lăng kính b) |
|---|---|---|---|---|---|---|
| a | p | oof_b2_5 | 119.1 | 0.291 | 0.548 | x |
| a | EV | oof_b2_5 | 1132.8 | 0.200 | 0.376 |  |
| a | EVcal | oof_b2_5 | 1289.4 | 0.217 | 0.409 |  |
| b | p | oof_b2_5 | 77.4 | 0.291 | 0.548 | x |
| b | EV | oof_b2_5 | 51.0 | 0.217 | 0.409 |  |
| b | EVcal | oof_b2_5 | 61.9 | 0.251 | 0.473 |  |

Chính sách được chọn: **Random Forest - top-k theo p**. (Lưu ý: ở lăng kính (a) thứ tự có thể khác; xem bảng.)


### Lăng kính (b) - sản phẩm thiếu (CHÍNH)

| Tập | Chính sách | Cảnh báo/ngày | Tổ trượt bắt được/ngày | Recall@3 | Precision@3 | Lợi ích ròng/ngày (sản phẩm) | KTC 95% (bootstrap theo ngày) | Lợi ích ròng/tháng (BDT, 26 ngày) |
|---|---|---|---|---|---|---|---|---|
| OOF khối 2-5 | Random Forest - top-k theo p | 3.00 | 1.65 | 0.291 | 0.548 | 77.4 | [44.5; 113.4] | - |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | 3.00 | 1.23 | 0.217 | 0.409 | 51.0 | [17.7; 89.8] | - |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | 3.00 | 1.42 | 0.251 | 0.473 | 61.9 | [27.0; 98.5] | - |
| OOF khối 2-5 | B1 (hôm qua trượt) | 3.00 | 1.58 | 0.280 | 0.527 | 66.8 | [34.0; 102.8] | - |
| OOF khối 2-5 | B2 (mọi tổ finishing) | 3.00 | 1.42 | 0.251 | 0.473 | 40.4 | [11.3; 73.7] | - |
| OOF khối 2-5 | B0 (lớp đa số) | 0.00 | 0.00 | 0.000 | - | 0.0 | [0.0; 0.0] | - |
| Test | Random Forest - top-k theo p | 3.00 | 2.08 | 0.368 | 0.694 | 131.4 | [70.1; 189.3] | - |
| Test | Random Forest - thiệt hại kỳ vọng | 3.00 | 1.42 | 0.250 | 0.472 | 100.3 | [42.0; 161.1] | - |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | 3.00 | 1.75 | 0.309 | 0.583 | 123.9 | [62.8; 193.0] | - |
| Test | B1 (hôm qua trượt) | 3.00 | 1.67 | 0.294 | 0.556 | 104.2 | [31.3; 179.8] | - |
| Test | B2 (mọi tổ finishing) | 3.00 | 1.92 | 0.338 | 0.639 | 129.0 | [66.2; 194.6] | - |
| Test | B0 (lớp đa số) | 0.00 | 0.00 | 0.000 | - | 0.0 | [0.0; 0.0] | - |

Hiệu lợi ích ròng/ngày (sản phẩm) so với B1 và B2 tại k = 3 (bootstrap theo ngày, 2000 lần):

| Tập | Chính sách | So với | Chênh trung bình | KTC 95% | P(chênh > 0) |
|---|---|---|---|---|---|
| OOF khối 2-5 | Random Forest - top-k theo p | B1 | +10.5 | [-27.1; 48.9] | 0.70 |
| OOF khối 2-5 | Random Forest - top-k theo p | B2 | +36.9 | [8.7; 66.4] | 0.99 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | B1 | -15.8 | [-47.9; 16.3] | 0.17 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | B2 | +10.6 | [-20.7; 44.8] | 0.72 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B1 | -5.0 | [-39.8; 31.0] | 0.40 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B2 | +21.4 | [-2.1; 45.0] | 0.96 |
| Test | Random Forest - top-k theo p | B1 | +27.2 | [-15.3; 71.0] | 0.90 |
| Test | Random Forest - top-k theo p | B2 | +2.5 | [-30.9; 36.1] | 0.56 |
| Test | Random Forest - thiệt hại kỳ vọng | B1 | -3.9 | [-41.8; 34.2] | 0.42 |
| Test | Random Forest - thiệt hại kỳ vọng | B2 | -28.6 | [-67.5; 9.7] | 0.08 |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B1 | +19.8 | [-18.9; 58.9] | 0.85 |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B2 | -5.0 | [-43.1; 28.8] | 0.39 |

### Lăng kính (a) - phút chuẩn

| Tập | Chính sách | Cảnh báo/ngày | Tổ trượt bắt được/ngày | Recall@3 | Precision@3 | Lợi ích ròng/ngày (phút chuẩn) | KTC 95% (bootstrap theo ngày) | Lợi ích ròng/tháng (BDT, 26 ngày) |
|---|---|---|---|---|---|---|---|---|
| OOF khối 2-5 | Random Forest - top-k theo p | 3.00 | 1.65 | 0.291 | 0.548 | 119.1 | [-199.8; 462.1] | 3097 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | 3.00 | 1.13 | 0.200 | 0.376 | 1132.8 | [559.9; 1758.2] | 29452 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | 3.00 | 1.23 | 0.217 | 0.409 | 1289.4 | [668.1; 1941.1] | 33524 |
| OOF khối 2-5 | B1 (hôm qua trượt) | 3.00 | 1.58 | 0.280 | 0.527 | 719.4 | [144.7; 1374.9] | 18705 |
| OOF khối 2-5 | B2 (mọi tổ finishing) | 3.00 | 1.42 | 0.251 | 0.473 | -402.0 | [-506.1; -280.8] | -10451 |
| OOF khối 2-5 | B0 (lớp đa số) | 0.00 | 0.00 | 0.000 | - | 0.0 | [0.0; 0.0] | 0 |
| Test | Random Forest - top-k theo p | 3.00 | 2.08 | 0.368 | 0.694 | -141.1 | [-330.4; 43.0] | -3669 |
| Test | Random Forest - thiệt hại kỳ vọng | 3.00 | 0.67 | 0.118 | 0.222 | 447.3 | [-313.4; 1273.5] | 11629 |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | 3.00 | 0.42 | 0.074 | 0.139 | -221.7 | [-720.0; 531.5] | -5763 |
| Test | B1 (hôm qua trượt) | 3.00 | 1.67 | 0.294 | 0.556 | -189.0 | [-406.9; 36.6] | -4913 |
| Test | B2 (mọi tổ finishing) | 3.00 | 1.92 | 0.338 | 0.639 | -140.9 | [-341.3; 65.8] | -3664 |
| Test | B0 (lớp đa số) | 0.00 | 0.00 | 0.000 | - | 0.0 | [0.0; 0.0] | 0 |

Hiệu lợi ích ròng/ngày (phút chuẩn) so với B1 và B2 tại k = 3 (bootstrap theo ngày, 2000 lần):

| Tập | Chính sách | So với | Chênh trung bình | KTC 95% | P(chênh > 0) |
|---|---|---|---|---|---|
| OOF khối 2-5 | Random Forest - top-k theo p | B1 | -600.3 | [-1177.1; -80.2] | 0.01 |
| OOF khối 2-5 | Random Forest - top-k theo p | B2 | +521.1 | [210.1; 851.6] | 1.00 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | B1 | +413.3 | [-31.7; 885.5] | 0.97 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | B2 | +1534.7 | [939.0; 2183.6] | 1.00 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B1 | +569.9 | [69.9; 1082.5] | 0.99 |
| OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B2 | +1691.3 | [1064.7; 2373.2] | 1.00 |
| Test | Random Forest - top-k theo p | B1 | +47.9 | [-101.2; 204.8] | 0.72 |
| Test | Random Forest - top-k theo p | B2 | -0.2 | [-134.7; 142.0] | 0.49 |
| Test | Random Forest - thiệt hại kỳ vọng | B1 | +636.2 | [-185.9; 1525.5] | 0.92 |
| Test | Random Forest - thiệt hại kỳ vọng | B2 | +588.2 | [-273.2; 1494.8] | 0.90 |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B1 | -32.7 | [-614.7; 737.6] | 0.43 |
| Test | Random Forest - thiệt hại kỳ vọng (p hiệu chỉnh) | B2 | -80.7 | [-676.0; 717.7] | 0.39 |

Lợi ích ròng/ngày theo fold rolling-origin tại k = 3 (trung bình ± độ lệch chuẩn qua 5 fold, OOF khối 1-5):

| Lăng kính | Chính sách | Lợi ích ròng/ngày |
|---|---|---|
| b | B0 (lớp đa số) | 0.0 ± 0.0 |
| b | B1 (hôm qua trượt) | 57.5 ± 32.5 |
| b | B2 (mọi tổ finishing) | 41.8 ± 20.2 |
| b | Random Forest - top-k theo p | 78.0 ± 29.6 |
| b | Random Forest - thiệt hại kỳ vọng | 51.8 ± 26.5 |
| a | B0 (lớp đa số) | 0.0 ± 0.0 |
| a | B1 (hôm qua trượt) | 477.4 ± 976.6 |
| a | B2 (mọi tổ finishing) | -406.9 ± 86.8 |
| a | Random Forest - top-k theo p | 67.1 ± 361.8 |
| a | Random Forest - thiệt hại kỳ vọng | 985.1 ± 807.1 |

Tác động dưới 3 kịch bản giả định (thận trọng r=0,10 và C_int x2; cơ sở r=0,30 và x1; lạc quan r=0,50 và x0,5; c_min 0,64/1,0/2,0 chỉ ở lăng kính a), k = 3:

| Lăng kính | Tập | Chính sách | Kịch bản | C_int | Cảnh báo/ngày | Tổ trượt bắt được/ngày | Thiệt hại cứu được/tháng | Lợi ích ròng/tháng | Lợi ích ròng/tháng (BDT) |
|---|---|---|---|---|---|---|---|---|---|
| b | OOF khối 2-5 | Random Forest - top-k theo p | Thận trọng | 31.5 | 3.00 | 1.65 | 1080 | -1374 | - |
| b | OOF khối 2-5 | Random Forest - top-k theo p | Cơ sở | 15.7 | 3.00 | 1.65 | 3239 | 2012 | - |
| b | OOF khối 2-5 | Random Forest - top-k theo p | Lạc quan | 7.9 | 3.00 | 1.65 | 5398 | 4784 | - |
| b | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Thận trọng | 31.5 | 3.00 | 1.23 | 851 | -1603 | - |
| b | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Cơ sở | 15.7 | 3.00 | 1.23 | 2553 | 1326 | - |
| b | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Lạc quan | 7.9 | 3.00 | 1.23 | 4255 | 3641 | - |
| b | OOF khối 2-5 | B1 (hôm qua trượt) | Thận trọng | 31.5 | 3.00 | 1.58 | 988 | -1465 | - |
| b | OOF khối 2-5 | B1 (hôm qua trượt) | Cơ sở | 15.7 | 3.00 | 1.58 | 2964 | 1738 | - |
| b | OOF khối 2-5 | B1 (hôm qua trượt) | Lạc quan | 7.9 | 3.00 | 1.58 | 4941 | 4327 | - |
| b | OOF khối 2-5 | B2 (mọi tổ finishing) | Thận trọng | 31.5 | 3.00 | 1.42 | 759 | -1694 | - |
| b | OOF khối 2-5 | B2 (mọi tổ finishing) | Cơ sở | 15.7 | 3.00 | 1.42 | 2278 | 1051 | - |
| b | OOF khối 2-5 | B2 (mọi tổ finishing) | Lạc quan | 7.9 | 3.00 | 1.42 | 3797 | 3183 | - |
| b | Test | Random Forest - top-k theo p | Thận trọng | 31.5 | 3.00 | 2.08 | 1548 | -906 | - |
| b | Test | Random Forest - top-k theo p | Cơ sở | 15.7 | 3.00 | 2.08 | 4643 | 3417 | - |
| b | Test | Random Forest - top-k theo p | Lạc quan | 7.9 | 3.00 | 2.08 | 7739 | 7126 | - |
| b | Test | Random Forest - thiệt hại kỳ vọng | Thận trọng | 31.5 | 3.00 | 1.42 | 1278 | -1175 | - |
| b | Test | Random Forest - thiệt hại kỳ vọng | Cơ sở | 15.7 | 3.00 | 1.42 | 3835 | 2608 | - |
| b | Test | Random Forest - thiệt hại kỳ vọng | Lạc quan | 7.9 | 3.00 | 1.42 | 6391 | 5778 | - |
| b | Test | B1 (hôm qua trượt) | Thận trọng | 31.5 | 3.00 | 1.67 | 1312 | -1142 | - |
| b | Test | B1 (hôm qua trượt) | Cơ sở | 15.7 | 3.00 | 1.67 | 3935 | 2708 | - |
| b | Test | B1 (hôm qua trượt) | Lạc quan | 7.9 | 3.00 | 1.67 | 6558 | 5945 | - |
| b | Test | B2 (mọi tổ finishing) | Thận trọng | 31.5 | 3.00 | 1.92 | 1527 | -927 | - |
| b | Test | B2 (mọi tổ finishing) | Cơ sở | 15.7 | 3.00 | 1.92 | 4580 | 3353 | - |
| b | Test | B2 (mọi tổ finishing) | Lạc quan | 7.9 | 3.00 | 1.92 | 7633 | 7019 | - |
| a | OOF khối 2-5 | Random Forest - top-k theo p | Thận trọng | 480.0 | 3.00 | 1.65 | 7272 | -30168 | -19307 |
| a | OOF khối 2-5 | Random Forest - top-k theo p | Cơ sở | 240.0 | 3.00 | 1.65 | 21817 | 3097 | 3097 |
| a | OOF khối 2-5 | Random Forest - top-k theo p | Lạc quan | 120.0 | 3.00 | 1.65 | 36362 | 27002 | 54005 |
| a | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Thận trọng | 480.0 | 3.00 | 1.13 | 16057 | -21383 | -13685 |
| a | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Cơ sở | 240.0 | 3.00 | 1.13 | 48172 | 29452 | 29452 |
| a | OOF khối 2-5 | Random Forest - thiệt hại kỳ vọng | Lạc quan | 120.0 | 3.00 | 1.13 | 80287 | 70927 | 141854 |
| a | OOF khối 2-5 | B1 (hôm qua trượt) | Thận trọng | 480.0 | 3.00 | 1.58 | 12475 | -24965 | -15978 |
| a | OOF khối 2-5 | B1 (hôm qua trượt) | Cơ sở | 240.0 | 3.00 | 1.58 | 37425 | 18705 | 18705 |
| a | OOF khối 2-5 | B1 (hôm qua trượt) | Lạc quan | 120.0 | 3.00 | 1.58 | 62376 | 53016 | 106031 |
| a | OOF khối 2-5 | B2 (mọi tổ finishing) | Thận trọng | 480.0 | 3.00 | 1.42 | 2756 | -34684 | -22198 |
| a | OOF khối 2-5 | B2 (mọi tổ finishing) | Cơ sở | 240.0 | 3.00 | 1.42 | 8269 | -10451 | -10451 |
| a | OOF khối 2-5 | B2 (mọi tổ finishing) | Lạc quan | 120.0 | 3.00 | 1.42 | 13782 | 4422 | 8843 |
| a | Test | Random Forest - top-k theo p | Thận trọng | 480.0 | 3.00 | 2.08 | 5017 | -32423 | -20751 |
| a | Test | Random Forest - top-k theo p | Cơ sở | 240.0 | 3.00 | 2.08 | 15051 | -3669 | -3669 |
| a | Test | Random Forest - top-k theo p | Lạc quan | 120.0 | 3.00 | 2.08 | 25085 | 15725 | 31449 |
| a | Test | Random Forest - thiệt hại kỳ vọng | Thận trọng | 480.0 | 3.00 | 0.67 | 10116 | -27324 | -17487 |
| a | Test | Random Forest - thiệt hại kỳ vọng | Cơ sở | 240.0 | 3.00 | 0.67 | 30349 | 11629 | 11629 |
| a | Test | Random Forest - thiệt hại kỳ vọng | Lạc quan | 120.0 | 3.00 | 0.67 | 50581 | 41221 | 82442 |
| a | Test | B1 (hôm qua trượt) | Thận trọng | 480.0 | 3.00 | 1.67 | 4602 | -32838 | -21016 |
| a | Test | B1 (hôm qua trượt) | Cơ sở | 240.0 | 3.00 | 1.67 | 13807 | -4913 | -4913 |
| a | Test | B1 (hôm qua trượt) | Lạc quan | 120.0 | 3.00 | 1.67 | 23011 | 13651 | 27302 |
| a | Test | B2 (mọi tổ finishing) | Thận trọng | 480.0 | 3.00 | 1.92 | 5019 | -32421 | -20750 |
| a | Test | B2 (mọi tổ finishing) | Cơ sở | 240.0 | 3.00 | 1.92 | 15056 | -3664 | -3664 |
| a | Test | B2 (mọi tổ finishing) | Lạc quan | 120.0 | 3.00 | 1.92 | 25093 | 15733 | 31466 |

Ngưỡng hoà vốn của Random Forest - top-k theo p (r* tại C_int cơ sở; C_int* tại r = 0,30; NaN = không có giao điểm dương):

| Lăng kính | Tập | So với | Chênh thiệt hại bắt được/ngày | Chênh cảnh báo/ngày | r* | C_int* |
|---|---|---|---|---|---|---|
| a | OOF khối 2-5 | B0 (không làm gì) | 2797.1 | 3.0 | 0.257 | 279.7 |
| a | OOF khối 2-5 | B1 | -2001.0 | 0.0 | - | - |
| a | OOF khối 2-5 | B2 | 1737.0 | 0.0 | - | - |
| b | OOF khối 2-5 | B0 (không làm gì) | 415.2 | 3.0 | 0.114 | 41.5 |
| b | OOF khối 2-5 | B1 | 35.1 | 0.0 | - | - |
| b | OOF khối 2-5 | B2 | 123.1 | 0.0 | - | - |
| a | Test | B0 (không làm gì) | 1929.6 | 3.0 | 0.373 | 193.0 |
| a | Test | B1 | 159.5 | 0.0 | - | - |
| a | Test | B2 | -0.7 | 0.0 | - | - |
| b | Test | B0 (không làm gì) | 595.3 | 3.0 | 0.079 | 59.5 |
| b | Test | B1 | 90.8 | 0.0 | - | - |
| b | Test | B2 | 8.2 | 0.0 | - | - |

Đường cong theo k (độ nhạy, không dùng để chọn k): `cost_curve.csv`, hình fig10/fig11.

## 9. Learning curve theo thời gian (kịch bản A, siêu tham số cố định)


Khối validate 5 (140 dòng, tỷ lệ trượt 0.407); B1 = 0.527 (không học):

| m (ngày huấn luyện) | Random Forest | Hồi quy logistic |
|---|---|---|
| 5 | 0.633 | 0.464 |
| 10 | 0.622 | 0.581 |
| 15 | 0.599 | 0.555 |
| 20 | 0.588 | 0.542 |
| 25 | 0.678 | 0.587 |
| 30 | 0.644 | 0.603 |
| 35 | 0.647 | 0.590 |
| 40 | 0.675 | 0.596 |

Khối validate 4 (148 dòng, tỷ lệ trượt 0.331); B1 = 0.465 (không học):

| m (ngày huấn luyện) | Random Forest | Hồi quy logistic |
|---|---|---|
| 5 | 0.470 | 0.569 |
| 10 | 0.414 | 0.560 |
| 15 | 0.472 | 0.516 |
| 20 | 0.531 | 0.547 |
| 25 | 0.548 | 0.546 |
| 30 | 0.528 | 0.528 |
| 32 | 0.545 | 0.531 |

## 10. Diễn giải (kịch bản A, mô hình chính)

Permutation importance trên test (giảm PR-AUC, 30 lần xáo trộn), top 6 cho Random Forest:

| Đặc trưng | Giảm PR-AUC (TB ± std) |
|---|---|
| no_of_workers | 0.1201 ± 0.0318 |
| smv | 0.0679 ± 0.0215 |
| roll5_miss_rate | 0.0311 ± 0.0177 |
| wip | 0.0226 ± 0.0118 |
| lag1_gap | 0.0144 ± 0.0171 |
| department | 0.0103 ± 0.0089 |

Hạng đặc trưng theo permutation importance của từng mô hình (1 = quan trọng nhất):

| Đặc trưng | Cây quyết định | SVM (RBF) | Mạng nơ-ron (MLP) | Hồi quy logistic | Random Forest |
|---|---|---|---|---|---|
| wip | 5 | 1 | 1 | 1 | 4 |
| no_of_workers | 1 | 4 | 10 | 2 | 1 |
| roll5_miss_rate | 3 | 2 | 12 | 3 | 3 |
| lag1_gap | 4 | 11 | 2 | 5 | 5 |
| department | 11 | 8 | 5 | 4 | 6 |
| smv | 2 | 12 | 13 | 6 | 2 |
| targeted_productivity | 7 | 3 | 6 | 12 | 9 |
| team | 6 | 9 | 8 | 7 | 10 |
| week_of_month | 8 | 6 | 4 | 11 | 11 |
| lag1_miss | 9 | 10 | 7 | 8 | 8 |
| roll5_gap_mean | 13 | 13 | 3 | 9 | 7 |
| is_first_obs | 10 | 5 | 11 | 10 | 12 |
| day | 12 | 7 | 9 | 13 | 13 |

Hệ số logistic lớn nhất (đặc trưng số đã chuẩn hoá; one-hot và is_first_obs theo thang 0/1):

| Đặc trưng | Hệ số |
|---|---|
| log1p(wip) | -2.969 |
| department_finishing | -2.451 |
| department_sewing | 1.993 |
| smv | 1.250 |
| no_of_workers | -0.962 |
| roll5_miss_rate | 0.628 |
| is_first_obs | 0.619 |
| team_12 | -0.541 |

Các luật của cây độ sâu 3 (sắp theo xác suất trượt giảm dần):

| Luật | Số dòng train | Tỷ lệ trượt thực | Xác suất (trọng số cân bằng) |
|---|---|---|---|
| roll5_miss_rate > 0.1 AND no_of_workers <= 8.5 AND smv <= 3.92 | 61 | 0.852 | 0.940 |
| roll5_miss_rate <= 0.1 AND smv > 30.2 | 14 | 0.786 | 0.909 |
| roll5_miss_rate > 0.1 AND no_of_workers <= 8.5 AND smv > 3.92 | 91 | 0.516 | 0.743 |
| roll5_miss_rate > 0.1 AND no_of_workers > 8.5 AND lag1_gap <= 0.0376 | 293 | 0.338 | 0.581 |
| roll5_miss_rate <= 0.1 AND smv <= 30.2 AND wip <= 714 | 133 | 0.195 | 0.397 |
| roll5_miss_rate > 0.1 AND no_of_workers > 8.5 AND lag1_gap > 0.0376 | 104 | 0.144 | 0.314 |
| roll5_miss_rate <= 0.1 AND smv <= 30.2 AND wip > 714 | 247 | 0.016 | 0.043 |

## 11. Độ nhạy: loại các dòng targeted_productivity < 0,5 (kịch bản A, k = 3, top-k theo p)

| Mô hình | PR-AUC CV | Chênh so với bản chính | PR-AUC test | Chênh so với bản chính | Ròng OOF/ngày (b) | Ròng test/ngày (b) | Ròng OOF/ngày (a) | Ròng test/ngày (a) |
|---|---|---|---|---|---|---|---|---|
| B0 (lớp đa số) | 0.293 ± 0.084 | +0.007 | 0.266 | -0.002 | 0.0 | 0.0 | 0 | 0 |
| B1 (hôm qua trượt) | 0.461 ± 0.099 | +0.018 | 0.485 | -0.006 | 58.0 | 112.6 | 492 | 78 |
| B2 (mọi tổ finishing) | 0.464 ± 0.084 | +0.009 | 0.599 | +0.008 | 41.7 | 129.0 | -408 | -141 |
| Cây quyết định | 0.522 ± 0.102 | +0.019 | 0.555 | -0.054 | 66.9 | 67.4 | 326 | -280 |
| SVM (RBF) | 0.557 ± 0.104 | +0.014 | 0.538 | -0.007 | 72.0 | 70.4 | 50 | -291 |
| Mạng nơ-ron (MLP) | 0.490 ± 0.103 | +0.004 | 0.543 | -0.026 | 62.0 | 145.3 | 266 | -96 |
| Hồi quy logistic | 0.532 ± 0.096 | +0.039 | 0.552 | -0.031 | 63.1 | 133.2 | 756 | 220 |
| Random Forest | 0.587 ± 0.126 | +0.013 | 0.580 | -0.110 | 73.4 | 85.2 | 487 | -249 |

Số dòng sau khi loại: train 919, test 248.

## 12. Phiếu cảnh báo mẫu

Tạo bằng `python src/alert_sheet.py --date 2015-03-05 --k 3 --lens b`. Các phiếu mẫu: `outputs/alert_sheets/alert_2015-02-26_k3_lensb.md`, `outputs/alert_sheets/alert_2015-03-05_k3_lensb.md`, `outputs/alert_sheets/alert_2015-03-11_k3_lensb.md`.

#### Phiếu cảnh báo đầu ca - 2015-03-05 (Thursday)

- Mô hình: Random Forest (kịch bản A, siêu tham số chọn trên train), huấn luyện trên 53 ngày (1068 dòng) trước ngày này.
- Xếp hạng: top-k theo điểm rủi ro của mô hình (p chưa hiệu chỉnh); lăng kính (b) giao hàng (sản phẩm thiếu); k = 3 tổ trong tổng 19 tổ-bộ phận hôm nay.

| Hạng | Tổ | Bộ phận | Điểm rủi ro (dùng để xếp hạng) | Xác suất trượt ước tính (đã hiệu chỉnh) | Thiệt hại kỳ vọng (sản phẩm) | Lý do chính | Kết quả thực tế (chỉ để kiểm tra) |
|---|---|---|---|---|---|---|---|
| 1 | 9 | finishing | 0.76 | 65% | 642 | số công nhân = 8, trung vị train 34; SMV (thời gian chuẩn) = 2.9, trung vị train 15.3; chênh lệch TB 5 ngày trước = -0.171, trung vị train 0 | TRƯỢT (0.45 so với định mức 0.75) |
| 2 | 12 | finishing | 0.67 | 51% | 387 | số công nhân = 9, trung vị train 34; chênh lệch TB 5 ngày trước = -0.138, trung vị train 0; tỷ lệ trượt 5 ngày trước = 0.8, trung vị train 0.2 | đạt (0.87 so với định mức 0.80) |
| 3 | 2 | finishing | 0.65 | 49% | 290 | số công nhân = 8, trung vị train 34; bộ phận = finishing; SMV (thời gian chuẩn) = 3.9, trung vị train 15.3 | đạt (0.95 so với định mức 0.60) |

Kiểm tra (không dùng khi chấm điểm): 1/3 tổ được cảnh báo thực sự trượt; cả ngày có 3/19 tổ-bộ phận trượt.

Ghi chú: điểm rủi ro là đầu ra thô của mô hình, chỉ dùng để xếp hạng; nó cao hơn xác suất thật vì mô hình được huấn luyện với trọng số cân bằng lớp. Xác suất trượt ước tính là điểm này sau hiệu chỉnh sigmoid (fit trên điểm out-of-fold của các ngày < D), đọc được theo nghĩa đen. Thiệt hại kỳ vọng = xác suất trượt ước tính x quy mô mục tiêu của tổ (targeted x số công nhân x 480 / smv); không phải ước tính tiền. Khi trượt, năng suất thường thấp hơn định mức khoảng 26% (trung bình lịch sử), nên thiệt hại thực thấp hơn quy mô mục tiêu. Lý do chính tính bằng p(x) - p(x với đặc trưng đặt về trung vị train); chỉ liệt kê yếu tố làm tăng rủi ro.


## 13. Các kiểm tra chống rò rỉ đã chạy

| Kiểm tra | Kết quả |
|---|---|
| Đặc trưng trễ ngày đầu của mỗi cặp (team, department) là NaN | đạt |
| Đặc trưng trễ nhân quả (kiểm tra nhiễu tương lai) | đạt |
| Không ngày nào xuất hiện ở cả train và test | đạt |
| Rolling-origin: mỗi fold validate đứng sau train, không chia ngày | đạt |
| Kịch bản A không chứa incentive/over_time/no_of_style_change/idle_*; không kịch bản nào chứa cột nhãn | đạt |
| Hiệu chỉnh sigmoid không rò rỉ: bộ hiệu chỉnh của khối j chỉ fit trên điểm OOF của các khối < j | đạt |
| Chính sách thiệt hại kỳ vọng được chọn chỉ dựa trên OOF (không dùng test) | đạt |
| Learning curve: mọi dòng huấn luyện đứng trước khối validate cố định | đạt |
| Pipeline fit chỉ trên train (DecisionTree, kịch bản A) | đạt |
| Pipeline fit chỉ trên train (SVM_RBF, kịch bản A) | đạt |
| Pipeline fit chỉ trên train (MLP, kịch bản A) | đạt |
| Pipeline fit chỉ trên train (Logistic, kịch bản A) | đạt |
| Pipeline fit chỉ trên train (RandomForest, kịch bản A) | đạt |
| Pipeline fit chỉ trên train (DecisionTree, kịch bản B) | đạt |
| Pipeline fit chỉ trên train (SVM_RBF, kịch bản B) | đạt |
| Pipeline fit chỉ trên train (MLP, kịch bản B) | đạt |
| Pipeline fit chỉ trên train (Logistic, kịch bản B) | đạt |
| Pipeline fit chỉ trên train (RandomForest, kịch bản B) | đạt |
| Không có cờ thiếu trùng lặp (wip_missing/lag_missing); đúng một cờ is_first_obs | đạt |
| Cây vẽ (không chuẩn hoá) trùng dự báo với cây cùng tham số trên dữ liệu chuẩn hoá | đạt |
| Phiếu cảnh báo: chỉ huấn luyện trên các ngày < D | đạt |
