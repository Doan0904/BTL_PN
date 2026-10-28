# ME3297 · Bài tập lớn · Dự báo tổ may không đạt định mức năng suất

Mã nguồn cho bài tập lớn môn ME3297 (Ứng dụng AI trong Quản lý chuỗi cung ứng, ĐH Bách khoa TP.HCM).
Bộ dữ liệu: *Productivity Prediction of Garment Employees* (UCI ML Repository, ID 597, CC BY 4.0).

**Quyết định được hỗ trợ.** Đầu mỗi ca, trưởng bộ phận sản xuất chọn 3 tổ (chuyền) có nguy cơ không đạt định mức năng suất trong ngày để cử người hỗ trợ.
Nhãn: `miss = 1` nếu `actual_productivity < targeted_productivity`.

## 1. Môi trường

- Python **3.11 đến 3.13**.
- `requirements.txt`: dải phiên bản tương thích. `requirements-lock.txt`: phiên bản ghim chính xác (Python 3.13), dùng để tái lập từng chữ số.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt    # hoặc: pip install -r requirements.txt
pip install notebook               # nếu muốn mở notebook trong trình duyệt (hoặc dùng VS Code)
```

Với `requirements.txt`, kết quả có thể lệch ở chữ số thập phân cuối; với `requirements-lock.txt`, kết quả trùng với các bảng trong `outputs/`.

## 2. Lấy dữ liệu (không nộp kèm dữ liệu thô)

Chạy từ thư mục gốc của mã nguồn:

```bash
python src/get_data.py                       # tải file zip chính thức của UCI ID 597
python src/get_data.py --method ucimlrepo    # cách khác: gói ucimlrepo (id=597)
python src/get_data.py --from-file duong/dan/garments_worker_productivity.csv   # khi không có mạng
```

Script kiểm tra file có đúng **1.197 dòng × 15 cột** và đúng tên cột, rồi ghi vào `data/garments_worker_productivity.csv`.
Nếu không tải được, tải tay từ <https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees> rồi dùng `--from-file`.

## 3. Chạy

**Cách 1: một lệnh (khoảng 3–4 phút)**

```bash
python src/run_all.py
```

Lệnh này ghi mọi bảng vào `outputs/tables/`, mọi hình vào `outputs/figures/`, các phiếu cảnh báo mẫu vào `outputs/alert_sheets/`, và sinh `RESULTS_SUMMARY.md` (các con số then chốt) từ các bảng. Mọi hạt giống ngẫu nhiên là 42, nên chạy lại cho đúng các con số trong báo cáo.

**Cách 2: lần lượt các notebook** trong `notebooks/`. Các notebook đã được chạy sẵn và lưu kèm output; mỗi notebook tự tính lại phần mình cần bằng hàm trong `src/`, có giải thích lý do và nhận xét cho từng kết quả.

| Notebook | Nội dung | Thời gian |
|---|---|---|
| `01_eda.ipynb` | Mô tả dữ liệu: từ điển, giá trị thiếu, ngoại lai, phân bố nhãn, các hình EDA | vài giây |
| `02_prep.ipynb` | Làm sạch, đặc trưng trễ và kiểm tra không nhìn vào tương lai, tách train/test, hai kịch bản đặc trưng A/B, `ColumnTransformer` | vài giây |
| `03_validation.ipynb` | Rolling-origin theo ngày, dò siêu tham số, thí nghiệm rò rỉ (chia ngẫu nhiên so với chia theo thời gian) | 1–2 phút |
| `04_models.ipynb` | 3 baseline, 5 mô hình, chọn mô hình chính, so từng cặp, kịch bản A/B, đánh giá test | 1–2 phút |
| `05_cost.ipynb` | Mô hình chi phí hai lăng kính, chọn chính sách cảnh báo 3 tổ/ngày, so với baseline, độ nhạy, hoà vốn | khoảng 1 phút |
| `06_interpret.ipynb` | Permutation importance, PDP/ICE, cây độ sâu 3, hệ số logistic, đánh giá tính hợp lý vận hành | 1–2 phút |
| `07_extensions.ipynb` | Độ tin cậy xác suất và hiệu chỉnh, learning curve, phiếu cảnh báo đầu ca | 1–2 phút |

Notebook chỉ hiển thị kết quả, không ghi đè `outputs/`.

**Phiếu cảnh báo đầu ca**, sau khi đã chạy `run_all.py`:

```bash
python src/alert_sheet.py --date 2015-03-05 --k 3 [--lens b] [--ranking auto|p|EV|EVcal] [--no-write]
```

Huấn luyện mô hình chính trên mọi ngày trước ngày D, rồi in k tổ cần hỗ trợ kèm điểm rủi ro (dùng để xếp hạng), xác suất trượt đã hiệu chỉnh, thiệt hại kỳ vọng và 2–3 lý do chính. Cột kết quả thực tế (nếu ngày D có trong dữ liệu) chỉ để kiểm tra. Kết quả ghi thêm ra `.md` và `.csv` trong `outputs/alert_sheets/`. `--lens a` tính thiệt hại theo phút chuẩn, `--lens b` (mặc định) theo số sản phẩm thiếu.

## 4. Cấu trúc

```
src/config.py          đường dẫn, hạt giống ngẫu nhiên
src/get_data.py        tải và kiểm tra dữ liệu
src/features.py        làm sạch, nhãn, đặc trưng trễ, kiểm tra không rò rỉ cho đặc trưng trễ
src/splits.py          tách test theo ngày, bộ chia rolling-origin theo ngày (tương thích scikit-learn)
src/pipeline.py        ColumnTransformer, kịch bản A/B, mô hình và lưới siêu tham số, baseline B0/B1/B2
src/evaluate.py        GridSearchCV, dự báo out-of-fold, đánh giá test, thí nghiệm rò rỉ
src/cost.py            hai lăng kính thiệt hại, cảnh báo top-k mỗi ngày, lợi ích ròng, độ nhạy, hoà vốn, bootstrap
src/calibration.py     độ tin cậy xác suất, Brier, hiệu chỉnh sigmoid không rò rỉ theo thời gian
src/learning_curve.py  learning curve theo thời gian
src/alert_sheet.py     phiếu cảnh báo đầu ca (dòng lệnh)
src/eda.py             các bảng EDA
src/plots.py           các hàm vẽ hình
src/interpret.py       permutation importance, PDP/ICE, cây, hệ số logistic
src/summary.py         sinh RESULTS_SUMMARY.md từ outputs/tables
src/run_all.py         chạy toàn bộ quy trình
notebooks/             01 đến 07 (đã chạy, có output)
outputs/tables/        mọi bảng CSV
outputs/figures/       mọi hình PNG
outputs/alert_sheets/  phiếu cảnh báo mẫu (md, csv)
RESULTS_SUMMARY.md     các con số then chốt (sinh tự động)
```

## 5. Chống rò rỉ dữ liệu

Các nguyên tắc dưới đây được kiểm tra tự động bằng `assert` khi chạy; `outputs/tables/leakage_checks.csv` liệt kê các kiểm tra đã qua.

- Đặc trưng trễ chỉ dùng các ngày trước: dòng đầu của mỗi cặp (tổ, bộ phận) là NaN, và thay đổi dữ liệu từ ngày d trở đi không làm đổi đặc trưng trễ của các ngày ≤ d.
- Không ngày nào nằm ở cả train và test; ở mỗi fold rolling-origin, các ngày validate đứng sau mọi ngày huấn luyện.
- Mọi bước học tham số (điền khuyết, chuẩn hoá, one-hot) nằm trong `Pipeline` và được fit lại ở mỗi fold; tham số học được khớp thống kê của train.
- Kịch bản A (chính) chỉ dùng biến biết được lúc đầu ca; `idle_time`, `idle_men` bị loại ở mọi kịch bản.
- Mô hình chính, siêu tham số và chính sách cảnh báo được chọn trên train và dự báo out-of-fold; k cố định bằng 3. Hiệu chỉnh sigmoid chỉ fit trên điểm out-of-fold của các khối đứng trước; learning curve và phiếu cảnh báo chỉ huấn luyện trên các ngày đứng trước.

Tập test có 12 ngày và đã được xem trong quá trình phát triển phương pháp, nên các con số trên test có thể hơi lạc quan; kết luận chính dựa vào kiểm định chéo và khoảng tin cậy.

## 6. Trích dẫn dữ liệu

*Productivity Prediction of Garment Employees* [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C51S6D (CC BY 4.0).
Bài báo giới thiệu: Imran, A. A., Rahim, M. S., & Ahmed, T. (2021). Mining the productivity data of the garment industry. *International Journal of Business Intelligence and Data Mining, 19*(3).
