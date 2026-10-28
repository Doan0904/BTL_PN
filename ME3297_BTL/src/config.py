"""Cấu hình chung: đường dẫn, hạt giống ngẫu nhiên, tên file dữ liệu."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "garments_worker_productivity.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

SEED = 42  # hạt giống cho mọi thành phần ngẫu nhiên
FIG_DPI = 150


def ensure_dirs():
    """Tạo thư mục đầu ra nếu chưa có."""
    for d in (DATA_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
