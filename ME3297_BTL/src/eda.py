"""Các bảng khám phá dữ liệu. Hàm chỉ trả về DataFrame; việc lưu file do run_all.py làm."""
import numpy as np
import pandas as pd

from cost import SAM_PER_DAY, damage
from features import LAG_COLS
from pipeline import SCENARIOS

# Ý nghĩa cột (theo mô tả UCI ID 597) và thời điểm biết (quyết định ra lúc đầu ca)
_DICT = {
    "date": ("Ngày ghi nhận (định dạng M/D/YYYY)", "ngày", "Có", "Dùng để tạo week_of_month, đặc trưng trễ và tách train/test"),
    "quarter": ("Tuần trong tháng do nguồn đặt tên 'Quarter1-5' (không phải quý)", "nhãn", "Có", "Không dùng; thay bằng week_of_month (tạo lại từ date)"),
    "department": ("Bộ phận: sewing (may) hoặc finishing (hoàn thiện)", "nhãn", "Có", "A, B (one-hot); làm sạch nhãn lỗi (sweing -> sewing, bỏ khoảng trắng thừa)"),
    "day": ("Thứ trong tuần", "nhãn", "Có", "A, B (one-hot)"),
    "team": ("Mã tổ 1-12 (biến phân loại, không phải thứ tự)", "mã", "Có", "A, B (one-hot)"),
    "targeted_productivity": ("Năng suất mục tiêu (định mức) do quản lý đặt cho tổ trong ngày", "tỷ lệ 0-1", "Có", "A, B"),
    "smv": ("Standard Minute Value: thời gian chuẩn để làm một sản phẩm của mã hàng", "phút/sản phẩm", "Có", "A, B"),
    "wip": ("Work in progress: số sản phẩm dở dang (chỉ ghi nhận cho sewing)", "sản phẩm", "Có (chỉ sewing)", "A, B (điền 0 cho finishing, log1p, chuẩn hoá)"),
    "over_time": ("Tổng số phút tăng ca của tổ", "phút", "Không chắc", "Chỉ kịch bản B (độ nhạy)"),
    "incentive": ("Tiền thưởng tài chính cho tổ", "BDT", "Không chắc (có thể trả theo kết quả)", "Chỉ kịch bản B (độ nhạy); rủi ro rò rỉ chính"),
    "idle_time": ("Thời gian sản xuất bị gián đoạn", "phút", "Không (phát sinh trong ngày)", "Loại khỏi mọi kịch bản"),
    "idle_men": ("Số công nhân ngừng việc do gián đoạn", "người", "Không (phát sinh trong ngày)", "Loại khỏi mọi kịch bản"),
    "no_of_style_change": ("Số lần đổi mã hàng trong ngày", "lần", "Gần như có (theo kế hoạch)", "Chỉ kịch bản B (độ nhạy)"),
    "no_of_workers": ("Số công nhân của tổ (có thể lẻ, ví dụ 30,5)", "người", "Có (theo kế hoạch nhân sự)", "A, B"),
    "actual_productivity": ("Năng suất thực tế (nguồn của nhãn)", "tỷ lệ 0-1,1", "Không", "Chỉ để tạo nhãn và đặc trưng trễ"),
}
_DERIVED = {
    "miss": ("Nhãn: 1 nếu actual < targeted", "0/1", "Biết cuối ngày", "Biến mục tiêu"),
    "gap": ("actual - targeted (âm là hụt định mức)", "tỷ lệ", "Biết cuối ngày", "Chỉ để tạo nhãn, đặc trưng trễ và thiệt hại"),
    "week_of_month": ("Tuần trong tháng tạo từ date: 1-7 -> 1, ..., 29-31 -> 5", "1-5", "Có", "A, B"),
    "lag1_miss": ("Tổ-bộ phận có trượt ở ngày làm việc trước đó không", "0/1", "Có", "A, B"),
    "lag1_gap": ("Chênh lệch năng suất ở ngày làm việc trước", "tỷ lệ", "Có", "A, B"),
    "roll5_gap_mean": ("Trung bình chênh lệch năng suất 5 ngày làm việc trước", "tỷ lệ", "Có", "A, B"),
    "roll5_miss_rate": ("Tỷ lệ trượt trong 5 ngày làm việc trước", "tỷ lệ 0-1", "Có", "A, B"),
}


def data_dictionary(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Từ điển dữ liệu cho mọi cột thô và cột tạo thêm (sau làm sạch)."""
    rows = []
    for col, (meaning, unit, known, usage) in {**_DICT, **_DERIVED}.items():
        s = clean_df[col]
        num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "cot": col, "y_nghia": meaning, "don_vi": unit, "biet_dau_ca": known, "su_dung": usage,
            "kieu": "số" if num else ("ngày" if pd.api.types.is_datetime64_any_dtype(s) else "phân loại"),
            "n_thieu": int(s.isna().sum()), "pct_thieu": round(100 * s.isna().mean(), 2),
            "n_khac_nhau": int(s.nunique()),
            "min": s.min() if (num or pd.api.types.is_datetime64_any_dtype(s)) else "",
            "trung_vi": s.median() if num else "",
            "max": s.max() if (num or pd.api.types.is_datetime64_any_dtype(s)) else "",
        })
    return pd.DataFrame(rows)


def missing_outliers(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Thiếu và ngoại lai (quy tắc 1,5 x IQR) cho các cột số gốc. Ngoại lai chỉ để mô tả, không loại bỏ."""
    rows = []
    for col in [c for c in _DICT if pd.api.types.is_numeric_dtype(clean_df[c])]:
        s = clean_df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append({"cot": col, "n_thieu": int(clean_df[col].isna().sum()),
                     "pct_thieu": round(100 * clean_df[col].isna().mean(), 2),
                     "q1": q1, "q3": q3, "can_duoi_iqr": lo, "can_tren_iqr": hi,
                     "n_ngoai_lai": int(((s < lo) | (s > hi)).sum()),
                     "pct_ngoai_lai": round(100 * ((s < lo) | (s > hi)).mean(), 2),
                     "pct_bang_0": round(100 * (s == 0).mean(), 2), "do_lech": s.skew(),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows)


def target_distribution(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Phân bố nhãn miss: toàn bộ, train/test, theo bộ phận, tuần trong tháng, tổ."""
    parts = [("Toàn bộ", df), ("Train", train), ("Test", test)]
    parts += [(f"Bộ phận = {d}", g) for d, g in df.groupby("department")]
    parts += [(f"Tuần trong tháng = {w}", g) for w, g in df.groupby("week_of_month")]
    parts += [(f"Tổ {t} (cả hai bộ phận)", g) for t, g in df.groupby("team")]
    return pd.DataFrame([{"nhom": n, "n_dong": len(g), "n_miss": int(g["miss"].sum()),
                          "ty_le_miss": g["miss"].mean()} for n, g in parts])


def quality_checks(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Đo các vấn đề chất lượng dữ liệu bằng số liệu thật."""
    rpd = df.groupby("date").size()
    n_pairs = df[["team", "department"]].drop_duplicates().shape[0]
    expected = n_pairs * df["date"].nunique()
    q5 = df[df["targeted_productivity"] < 0.5]
    fin = df["department"] == "finishing"
    q = [
        ("Nhãn department lỗi", f"Giá trị thô: {raw['department'].value_counts().to_dict()}"),
        ("wip thiếu có cấu trúc", f"{int(df['wip'].isna().sum())} dòng ({100*df['wip'].isna().mean():.1f}%); "
                                        f"thiếu ở finishing: {100*df.loc[fin,'wip'].isna().mean():.0f}%, ở sewing: {100*df.loc[~fin,'wip'].isna().mean():.0f}%"),
        ("wip lệch phải", f"max = {df['wip'].max():.0f}; trung vị = {df['wip'].median():.0f}; độ lệch = {df['wip'].skew():.2f}"),
        ("actual_productivity > 1", f"{int((df['actual_productivity'] > 1).sum())} dòng; max = {df['actual_productivity'].max():.3f}"),
        ("targeted_productivity < 0,5", f"{len(q5)} dòng; giá trị: {[float(v) for v in sorted(q5['targeted_productivity'].unique())]}; "
                                              f"tỷ lệ miss ở nhóm này = {q5['miss'].mean():.3f}"),
        ("no_of_workers không nguyên", f"{int((df['no_of_workers'] % 1 != 0).sum())} dòng"),
        ("quarter thực chất là tuần trong tháng", f"{raw['quarter'].value_counts().sort_index().to_dict()}; "
                                                         f"tuần 5 (ngày 29-31) có {int((df['week_of_month'] == 5).sum())} dòng"),
        ("idle_time/idle_men gần như hằng số", f"idle_time>0: {int((df['idle_time']>0).sum())} dòng; idle_men>0: {int((df['idle_men']>0).sum())} dòng"),
        ("Tổ-ngày bị thiếu", f"{len(df)} dòng so với kỳ vọng {expected} ({n_pairs} cặp tổ-bộ phận x {df['date'].nunique()} ngày); "
                                    f"số dòng mỗi ngày: min {rpd.min()}, trung vị {rpd.median():.0f}, max {rpd.max()}; "
                                    f"ngày đủ {n_pairs} dòng: {int((rpd == n_pairs).sum())}"),
        ("Phạm vi thời gian", f"{df['date'].min().date()} đến {df['date'].max().date()}; {df['date'].nunique()} ngày làm việc; "
                                      f"ngày trong tuần có dữ liệu: {sorted(df['day'].unique())}"),
        ("Dòng trùng / khoá trùng", f"{int(raw.duplicated().sum())} / {int(df.duplicated(['date','team','department']).sum())}"),
    ]
    return pd.DataFrame(q, columns=["van_de", "do_duoc"])


def rows_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Số dòng theo ngày và số cặp (tổ, bộ phận) thiếu mỗi ngày."""
    n_pairs = df[["team", "department"]].drop_duplicates().shape[0]
    out = df.groupby("date").size().rename("n_dong").reset_index()
    out["n_cap_thieu"] = n_pairs - out["n_dong"]
    out["thu"] = out["date"].dt.day_name()
    return out


def damage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mục 6.1: phân phối thiệt hại mỗi lần trượt (phút chuẩn) và sản phẩm thiếu, theo bộ phận."""
    d = df.assign(damage_sam=damage(df, "a"))
    d = d[d["miss"] == 1].copy()
    d["san_pham_thieu"] = d["damage_sam"] / d["smv"]
    rows = []
    for name, g in [("Tất cả", d)] + [(k, v) for k, v in d.groupby("department")]:
        for col, label in (("damage_sam", "Thiệt hại (phút chuẩn)"), ("san_pham_thieu", "Sản phẩm thiếu (cái)")):
            s = g[col]
            rows.append({"nhom": name, "chi_tieu": label, "n_lan_truot": len(s), "trung_binh": s.mean(),
                         "trung_vi": s.median(), "p25": s.quantile(.25), "p75": s.quantile(.75),
                         "p90": s.quantile(.90), "max": s.max()})
    return pd.DataFrame(rows)


def feature_scenarios_table() -> pd.DataFrame:
    """Danh sách cột theo kịch bản A/B."""
    rows = []
    for name, sc in SCENARIOS.items():
        for grp, cols in (("số", sc.numeric), ("wip", sc.wip), ("trễ", sc.lag), ("cờ", sc.flags), ("phân loại", sc.categorical)):
            for c in cols:
                rows.append({"kich_ban": name, "nhom": grp, "cot": c})
    return pd.DataFrame(rows)
