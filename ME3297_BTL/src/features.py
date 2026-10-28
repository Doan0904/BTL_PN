"""Làm sạch xác định, tạo nhãn và đặc trưng trễ.

Mọi bước ở đây là biến đổi XÁC ĐỊNH: không học tham số từ dữ liệu (không median,
không scaler...). Các bước có tham số học từ dữ liệu nằm trong Pipeline (pipeline.py).
"""
import numpy as np
import pandas as pd

KEY = ["team", "department"]                       # cặp (tổ, bộ phận) dùng cho đặc trưng trễ
LAG_COLS = ["lag1_miss", "lag1_gap", "roll5_gap_mean", "roll5_miss_rate"]
FLAG_COLS = ["is_first_obs"]                       # đúng 1 cờ cho "chưa có lịch sử của cặp"
SHIFT_ONLY_HELPERS = ["gap", "miss"]               # cột nguồn của đặc trưng trễ


def load_raw(path) -> pd.DataFrame:
    """Đọc CSV gốc, giữ nguyên mọi giá trị (kể cả nhãn lỗi)."""
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch nhãn department (sweing -> sewing, bỏ khoảng trắng thừa), parse ngày, sắp xếp theo (date, team, department)."""
    df = raw.copy()
    df["department"] = df["department"].str.strip().replace({"sweing": "sewing"})
    assert set(df["department"]) == {"sewing", "finishing"}, "Nhãn department vẫn lỗi"
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["day"] = df["day"].str.strip()
    df = df.sort_values(["date", "team", "department"]).reset_index(drop=True)
    assert not df.duplicated(["date", "team", "department"]).any(), "Trùng khoá (date, team, department)"
    return df


def week_of_month(date: pd.Series) -> pd.Series:
    """1-7 -> 1, 8-14 -> 2, 15-21 -> 3, 22-28 -> 4, 29-31 -> 5."""
    return ((date.dt.day - 1) // 7 + 1).astype(int)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """miss = actual < targeted. Thêm gap = actual - targeted (âm là hụt định mức)."""
    out = df.copy()
    out["gap"] = out["actual_productivity"] - out["targeted_productivity"]
    out["miss"] = (out["actual_productivity"] < out["targeted_productivity"]).astype(int)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng trễ theo cặp (team, department), chỉ dùng các ngày làm việc TRƯỚC.

    - lag1_*: giá trị ở dòng liền trước của cùng cặp (ngày làm việc gần nhất của cặp đó).
    - roll5_*: trung bình 5 dòng liền trước (shift(1) rồi rolling(5)); cần >= 1 quan sát.
    Dòng đầu tiên của mỗi cặp để NaN (điền median trong Pipeline) và được gắn cờ is_first_obs = 1.
    Yêu cầu df đã được sắp theo ngày (clean()) và đã có cột gap, miss.
    """
    out = df.copy()
    g = out.groupby(KEY, sort=False)
    out["lag1_miss"] = g["miss"].shift(1).astype(float)
    out["lag1_gap"] = g["gap"].shift(1)
    out["roll5_gap_mean"] = g["gap"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    out["roll5_miss_rate"] = g["miss"].transform(
        lambda s: s.astype(float).shift(1).rolling(5, min_periods=1).mean())
    out["is_first_obs"] = out["lag1_miss"].isna().astype(int)   # chỉ phụ thuộc vào việc cặp đã có dòng trước hay chưa
    return out


def build_dataset(path) -> pd.DataFrame:
    """Đọc CSV -> làm sạch -> nhãn -> week_of_month -> đặc trưng trễ. Có kiểm tra rò rỉ."""
    df = clean(load_raw(path))
    df.insert(0, "row_id", np.arange(len(df)))   # định danh dòng ổn định (dùng khi phá hoà top-k)
    df = add_label(df)
    df["week_of_month"] = week_of_month(df["date"])
    df = add_lag_features(df)
    assert_lags_first_day_nan(df)
    assert_lags_causal(df)
    return df


# ----------------------------------------------------------------------------
# Các assert chống rò rỉ cho đặc trưng trễ
# ----------------------------------------------------------------------------
def assert_lags_first_day_nan(df: pd.DataFrame) -> None:
    """Dòng đầu của mỗi cặp (team, department) phải có mọi đặc trưng trễ = NaN,
    và mọi dòng còn lại phải có giá trị."""
    first = ~df.duplicated(KEY, keep="first")
    assert df.loc[first, LAG_COLS].isna().all().all(), "Ngày đầu của cặp có đặc trưng trễ khác NaN"
    assert df.loc[~first, LAG_COLS].notna().all().all(), "Dòng không phải ngày đầu mà đặc trưng trễ NaN"
    assert (df["is_first_obs"].to_numpy() == first.to_numpy().astype(int)).all(), "is_first_obs không khớp dòng đầu của cặp"


def assert_lags_causal(df: pd.DataFrame, n_trials: int = 5, seed: int = 42) -> None:
    """Kiểm tra nhân quả: đổi toàn bộ nhãn/năng suất từ ngày d trở đi thì đặc trưng trễ của
    các dòng có date <= d phải KHÔNG đổi (nghĩa là không dùng thông tin từ tương lai/cùng ngày)."""
    rng = np.random.default_rng(seed)
    dates = np.sort(df["date"].unique())
    cols = LAG_COLS + FLAG_COLS
    base = df[cols].to_numpy()
    for d in rng.choice(dates[5:-1], size=n_trials, replace=False):
        d = pd.Timestamp(d)
        alt = df[["date", "team", "department", "targeted_productivity", "actual_productivity"]].copy()
        future = alt["date"] >= d
        alt.loc[future, "actual_productivity"] = rng.uniform(0.0, 1.5, future.sum())  # nhiễu mạnh
        alt = add_lag_features(add_label(alt))
        past = (df["date"] <= d).to_numpy()
        assert np.allclose(base[past], alt[cols].to_numpy()[past], equal_nan=True), \
            f"Đặc trưng trễ phụ thuộc dữ liệu từ ngày {d.date()} trở đi (rò rỉ tương lai)"
