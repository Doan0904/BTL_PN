"""Lấy bộ dữ liệu 'Productivity Prediction of Garment Employees' (UCI ID 597, CC BY 4.0).

Cách dùng (chạy từ thư mục gốc của repo):
    python src/get_data.py                      # tải zip chính thức từ UCI
    python src/get_data.py --method ucimlrepo   # dùng gói ucimlrepo (id=597)
    python src/get_data.py --from-file duong/dan/garments_worker_productivity.csv

Trích dẫn: Productivity Prediction of Garment Employees [Dataset]. (2020).
UCI Machine Learning Repository. https://doi.org/10.24432/C51S6D
"""
import argparse
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_FILE, ensure_dirs  # noqa: E402

UCI_ZIP_URL = ("https://archive.ics.uci.edu/static/public/597/"
               "productivity+prediction+of+garment+employees.zip")
EXPECTED_SHAPE = (1197, 15)
EXPECTED_COLUMNS = [
    "date", "quarter", "department", "day", "team", "targeted_productivity", "smv",
    "wip", "over_time", "incentive", "idle_time", "idle_men", "no_of_style_change",
    "no_of_workers", "actual_productivity",
]


def validate(path: Path) -> pd.DataFrame:
    """Kiểm tra số dòng/cột và tên cột; báo lỗi rõ ràng nếu không khớp."""
    df = pd.read_csv(path)
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Kích thước {df.shape} khác kỳ vọng {EXPECTED_SHAPE}. "
                         "File có thể không phải bản gốc của UCI ID 597.")
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Tên cột không khớp. Nhận được: {list(df.columns)}")
    return df


def download_zip(url: str = UCI_ZIP_URL, timeout: int = 60) -> bytes:
    """Tải file CSV từ zip chính thức của UCI, trả về nội dung bytes của CSV."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError(f"Không thấy file CSV trong zip: {zf.namelist()}")
        return zf.read(csvs[0])


def download_ucimlrepo() -> pd.DataFrame:
    """Tải bằng gói ucimlrepo (cần `pip install ucimlrepo`)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=597)
    return ds.data.original[EXPECTED_COLUMNS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file", type=Path, default=None,
                    help="Dùng bản CSV có sẵn thay vì tải (ví dụ khi UCI bị chặn mạng).")
    ap.add_argument("--method", choices=["zip", "ucimlrepo"], default="zip")
    ap.add_argument("--out", type=Path, default=DATA_FILE)
    args = ap.parse_args(argv)

    ensure_dirs()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file is not None:
        if not args.from_file.exists():
            sys.exit(f"Không tìm thấy file: {args.from_file}")
        if args.from_file.resolve() != args.out.resolve():
            shutil.copyfile(args.from_file, args.out)
        print(f"Đã sao chép {args.from_file} -> {args.out}")
    else:
        try:
            if args.method == "zip":
                args.out.write_bytes(download_zip())
                print(f"Đã tải từ {UCI_ZIP_URL}")
            else:
                download_ucimlrepo().to_csv(args.out, index=False)
                print("Đã tải bằng ucimlrepo (id=597)")
        except Exception as exc:  # mạng bị chặn, UCI lỗi, thiếu gói...
            sys.exit(f"Không tải được dữ liệu ({type(exc).__name__}: {exc}).\n"
                     "Hãy tải tay từ https://archive.ics.uci.edu/dataset/597 rồi chạy:\n"
                     "    python src/get_data.py --from-file <đường dẫn tới CSV>")

    df = validate(args.out)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"OK: {df.shape[0]} dòng x {df.shape[1]} cột -> {args.out}")
    print(f"SHA-256: {sha}  (chỉ để đối chiếu; không dùng làm điều kiện kiểm tra)")


if __name__ == "__main__":
    main()
