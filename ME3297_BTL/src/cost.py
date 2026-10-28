"""Mô hình chi phí và chính sách cảnh báo top-k mỗi ngày.

Hai lăng kính thiệt hại, cột `lens` có trong mọi bảng chi phí:
  (a) chi phí lao động : thiệt hại = (target - actual) * workers * 480                 [phút chuẩn, SAM]
  (b) giao hàng        : thiệt hại = (target - actual) * workers * 480 / smv           [sản phẩm thiếu]
Lợi ích ròng / ngày = r * (tổng thiệt hại của các tổ trượt được cảnh báo đúng) - C_int * (số cảnh báo)
  - lăng kính (a): C_int = 240 phút chuẩn / lần;
  - lăng kính (b): C_int = 240 / trung vị smv của TRAIN (sản phẩm / lần), độ nhạy x0,5 .. x2.
Quy đổi tiền (BDT) chỉ có ý nghĩa ở lăng kính (a): nhân với c_min (BDT / phút chuẩn).

Chính sách: k cố định = 3 mỗi ngày.
  - top-k theo xác suất p                      (tên chính sách = tên mô hình)
  - thiệt hại kỳ vọng: điểm = p x quy mô mục tiêu biết đầu ca     (hậu tố "_EV")
  - thiệt hại kỳ vọng dùng p đã hiệu chỉnh sigmoid           (hậu tố "_EVcal")
  - baseline B0 (không cảnh báo), B1 (hôm qua trượt), B2 (mọi tổ finishing)
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import SEED

SAM_PER_DAY = 480            # số phút làm việc chuẩn mỗi ca
WORKDAYS_PER_MONTH = 26      # giả định: tuần làm việc Thứ Bảy-Thứ Năm (dữ liệu không có thứ Sáu)
K_GRID = list(range(1, 9))   # đường cong theo k (chỉ là phân tích độ nhạy)
K_BASE = 3                   # k cố định theo nguồn lực hỗ trợ

LENSES = ["a", "b"]
LENS_LABEL = {"a": "(a) chi phí lao động (phút chuẩn)", "b": "(b) giao hàng (sản phẩm thiếu)"}
LENS_UNIT = {"a": "phút chuẩn", "b": "sản phẩm"}
PRIMARY_LENS = "b"           # lăng kính chính cho khuyến nghị

R_BASE = 0.30
C_INT_A = 240.0              # phút chuẩn / lần hỗ trợ
C_MIN_BASE = 1.0             # BDT / phút chuẩn
R_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
C_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]          # C_int = hệ số x giá trị cơ sở: (a) 120..480, (b) x0,5..x2
C_MIN_GRID = [0.64, 1.0, 2.0]
# Kịch bản độ nhạy: (r, hệ số C_int, c_min)
SCENARIO_DEFS = {"Thận trọng": (0.10, 2.0, 0.64), "Cơ sở": (R_BASE, 1.0, C_MIN_BASE), "Lạc quan": (0.50, 0.5, 2.00)}


@dataclass(frozen=True)
class CostSetup:
    """Tham số chi phí cơ sở; smv_median là trung vị smv của TẬP TRAIN."""
    smv_median: float

    def c_int(self, lens: str) -> float:
        return C_INT_A if lens == "a" else C_INT_A / self.smv_median

    def base(self, lens: str) -> dict:
        return {"r": R_BASE, "c_int": self.c_int(lens), "c_min": C_MIN_BASE}

    def scenarios(self, lens: str) -> dict:
        return {n: {"r": r, "c_int": f * self.c_int(lens), "c_min": cm} for n, (r, f, cm) in SCENARIO_DEFS.items()}

    def c_int_grid(self, lens: str) -> list:
        return [f * self.c_int(lens) for f in C_FACTORS]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"lens": ln, "lens_label": LENS_LABEL[ln], "unit": LENS_UNIT[ln], "r_base": R_BASE,
                              "c_int_base": self.c_int(ln), "smv_median_train": self.smv_median,
                              "c_int_min": self.c_int(ln) * C_FACTORS[0], "c_int_max": self.c_int(ln) * C_FACTORS[-1]}
                             for ln in LENSES])


# ----------------------------------------------------------------------------
# Thiệt hại và quy mô mục tiêu theo lăng kính
# ----------------------------------------------------------------------------
def damage(df: pd.DataFrame, lens: str) -> np.ndarray:
    """Thiệt hại nếu tổ-ngày trượt (0 nếu đạt). Chỉ dùng để ĐÁNH GIÁ, không phải đặc trưng."""
    shortfall = (df["targeted_productivity"] - df["actual_productivity"]).clip(lower=0)
    d = shortfall * df["no_of_workers"] * SAM_PER_DAY
    return (d if lens == "a" else d / df["smv"]).to_numpy()


def exposure(df: pd.DataFrame, lens: str) -> np.ndarray:
    """Quy mô mục tiêu biết trước đầu ca: targeted * workers * 480 (a) hoặc chia thêm smv (b)."""
    e = df["targeted_productivity"] * df["no_of_workers"] * SAM_PER_DAY
    return (e if lens == "a" else e / df["smv"]).to_numpy()


# ----------------------------------------------------------------------------
# Điểm xếp hạng của các chính sách
# ----------------------------------------------------------------------------
def policy_scores(frame: pd.DataFrame, p_scores: dict, lens: str, ev_models=(), cal_scores: dict | None = None) -> dict:
    """Từ điểm p của mô hình + baseline, thêm các chính sách thiệt hại kỳ vọng cho lăng kính `lens`.

    ev_models: mô hình có xác suất -> thêm "<m>_EV" = p x exposure. cal_scores = {m: p đã hiệu chỉnh} -> "<m>_EVcal".
    """
    out = dict(p_scores)
    ex = exposure(frame, lens)
    for m in ev_models:
        out[f"{m}_EV"] = p_scores[m] * ex
    for m, pc in (cal_scores or {}).items():
        out[f"{m}_EVcal"] = pc * ex
    return out


def split_policy(name: str) -> tuple:
    """'Logistic_EV' -> ('Logistic', 'EV'); 'Logistic' -> ('Logistic', 'p'); 'B1' -> ('B1', 'baseline')."""
    for suf in ("_EVcal", "_EV"):
        if name.endswith(suf):
            return name[: -len(suf)], suf[1:]
    return name, ("baseline" if name in ("B0", "B1", "B2") else "p")


# ----------------------------------------------------------------------------
# Chính sách top-k mỗi ngày
# ----------------------------------------------------------------------------
def _tiebreak(row_id: np.ndarray) -> np.ndarray:
    """Số ngẫu nhiên cố định (seed 42) gắn với từng row_id để phá hoà ổn định giữa các lần gọi."""
    row_id = np.asarray(row_id, dtype=int)
    return np.random.default_rng(SEED).random(int(row_id.max()) + 1)[row_id]


def top_k_alerts(df: pd.DataFrame, score: np.ndarray, k: int) -> np.ndarray:
    """Mỗi ngày cảnh báo k tổ có điểm cao nhất (hoà thì phá bằng số ngẫu nhiên cố định)."""
    n = len(df)
    if k <= 0 or n == 0:
        return np.zeros(n, dtype=bool)
    day = pd.factorize(df["date"])[0]
    tb = _tiebreak(df["row_id"].to_numpy())
    order = np.lexsort((tb, -np.asarray(score, dtype=float), day))
    day_sorted = day[order]
    rank_in_day = np.arange(n) - np.searchsorted(day_sorted, day_sorted, side="left")
    alerts = np.zeros(n, dtype=bool)
    alerts[order[rank_in_day < k]] = True
    return alerts


def policy_alerts(df: pd.DataFrame, score: np.ndarray, k: int, name: str = "") -> np.ndarray:
    """B0 (lớp đa số) không bao giờ cảnh báo; các chính sách khác dùng top-k."""
    if name == "B0":
        return np.zeros(len(df), dtype=bool)
    return top_k_alerts(df, score, k)


# ----------------------------------------------------------------------------
# Thống kê chính sách và lợi ích ròng
# ----------------------------------------------------------------------------
def policy_stats(df: pd.DataFrame, alerts: np.ndarray, dmg: np.ndarray) -> dict:
    """Đại lượng đủ (sufficient statistics) của một chính sách trên tập df cho một vector thiệt hại.

    Lợi ích ròng tuyến tính theo r và C_int nên mọi bảng độ nhạy tính được từ các số này.
    Mọi số 'per_day' chia cho số ngày duy nhất trong df.
    """
    miss = df["miss"].to_numpy().astype(bool)
    alerts = np.asarray(alerts, dtype=bool)
    n_days = df["date"].nunique()
    tp = alerts & miss
    day = pd.factorize(df["date"])[0]
    pos_day = np.bincount(day, weights=miss.astype(float))
    tp_day = np.bincount(day, weights=tp.astype(float))
    al_day = np.bincount(day, weights=alerts.astype(float))
    has_pos, has_al = pos_day > 0, al_day > 0
    return {
        "n_days": n_days,
        "alerts_per_day": alerts.sum() / n_days,
        "tp_per_day": tp.sum() / n_days,
        "misses_per_day": miss.sum() / n_days,
        "caught_damage_per_day": dmg[tp].sum() / n_days,       # thiệt hại của các tổ trượt được bắt đúng
        "total_damage_per_day": dmg[miss].sum() / n_days,
        "recall_pooled": tp.sum() / miss.sum() if miss.sum() else np.nan,
        "precision_pooled": tp.sum() / alerts.sum() if alerts.sum() else np.nan,
        "recall_daymean": float(np.mean(tp_day[has_pos] / pos_day[has_pos])) if has_pos.any() else np.nan,
        "precision_daymean": float(np.mean(tp_day[has_al] / al_day[has_al])) if has_al.any() else np.nan,
    }


def net_from_stats(stats: dict, lens: str, r: float, c_int: float, c_min: float = C_MIN_BASE) -> dict:
    """Lợi ích ròng mỗi ngày và mỗi tháng. Tiền (BDT) chỉ có ở lăng kính (a)."""
    saved = r * stats["caught_damage_per_day"]
    cost = c_int * stats["alerts_per_day"]
    net = saved - cost
    out = {"saved_per_day": saved, "cost_per_day": cost, "net_per_day": net,
           "saved_per_month": saved * WORKDAYS_PER_MONTH, "net_per_month": net * WORKDAYS_PER_MONTH,
           "net_bdt_per_day": np.nan, "net_bdt_per_month": np.nan}
    if lens == "a":
        out["net_bdt_per_day"] = net * c_min
        out["net_bdt_per_month"] = net * c_min * WORKDAYS_PER_MONTH
    return out


def evaluate_policy(df, alerts, lens, setup: CostSetup, dmg=None, **override) -> dict:
    """Thống kê + lợi ích ròng ở tham số chi phí cơ sở của lăng kính (hoặc tham số ghi đè)."""
    p = {**setup.base(lens), **override}
    st = policy_stats(df, alerts, damage(df, lens) if dmg is None else dmg)
    return {**st, **net_from_stats(st, lens, p["r"], p["c_int"], p["c_min"])}


def daily_net(df, alerts, dmg, r, c_int) -> pd.Series:
    """Lợi ích ròng từng ngày (chỉ số là ngày); dùng cho bootstrap theo ngày."""
    miss = df["miss"].to_numpy().astype(bool)
    alerts = np.asarray(alerts, dtype=bool)
    val = r * np.where(alerts & miss, dmg, 0.0) - c_int * alerts
    return pd.Series(val).groupby(df["date"].to_numpy()).sum()


def policy_curve(df, scores: dict, lens: str, setup: CostSetup, ks=K_GRID) -> pd.DataFrame:
    """Bảng chính sách theo k (độ nhạy theo k). Baseline 'B0' luôn không cảnh báo."""
    dmg = damage(df, lens)
    rows = []
    for name, sc in scores.items():
        for k in ks:
            al = policy_alerts(df, sc, k, name)
            rows.append({"policy": name, "k": k, **evaluate_policy(df, al, lens, setup, dmg)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Độ nhạy, hoà vốn, bootstrap
# ----------------------------------------------------------------------------
def sensitivity_scenarios(df, scores: dict, lens: str, setup: CostSetup, k: int = K_BASE) -> pd.DataFrame:
    """Tác động dưới 3 kịch bản giả định (thận trọng / cơ sở / lạc quan) cho từng chính sách."""
    dmg = damage(df, lens)
    rows = []
    for name, sc in scores.items():
        st = policy_stats(df, policy_alerts(df, sc, k, name), dmg)
        for label, p in setup.scenarios(lens).items():
            rows.append({"policy": name, "k": 0 if name == "B0" else k, "cost_scenario": label, **p,
                         "alerts_per_day": st["alerts_per_day"], "tp_per_day": st["tp_per_day"],
                         "recall_pooled": st["recall_pooled"], "precision_pooled": st["precision_pooled"],
                         **net_from_stats(st, lens, p["r"], p["c_int"], p["c_min"])})
    return pd.DataFrame(rows)


def sensitivity_one_at_a_time(df, scores: dict, lens: str, setup: CostSetup, k: int = K_BASE) -> pd.DataFrame:
    """Đổi một tham số mỗi lần (r, C_int, c_min) quanh giá trị cơ sở; trả lợi ích ròng/ngày."""
    dmg = damage(df, lens)
    grids = (("r", R_GRID), ("c_int", setup.c_int_grid(lens)), ("c_min", C_MIN_GRID if lens == "a" else []))
    rows = []
    for name, sc in scores.items():
        st = policy_stats(df, policy_alerts(df, sc, k, name), dmg)
        for param, grid in grids:
            for v in grid:
                p = {**setup.base(lens), param: v}
                rows.append({"policy": name, "k": 0 if name == "B0" else k, "param": param, "value": v,
                             **net_from_stats(st, lens, p["r"], p["c_int"], p["c_min"])})
    return pd.DataFrame(rows)


def net_grid(df, score, lens: str, setup: CostSetup, k: int = K_BASE, name: str = "") -> pd.DataFrame:
    """Lưới lợi ích ròng/ngày theo (r, C_int) cho một chính sách; dùng cho bản đồ nhiệt."""
    st = policy_stats(df, policy_alerts(df, score, k, name), damage(df, lens))
    return pd.DataFrame([{"r": r, "c_int": c, "c_int_factor": f, "net_per_day": net_from_stats(st, lens, r, c)["net_per_day"]}
                         for r in R_GRID for f, c in zip(C_FACTORS, setup.c_int_grid(lens))])


def break_even(stats_model: dict, stats_ref: dict | None, r: float, c_int: float) -> dict:
    """Ngưỡng hoà vốn từ NB = r*S - C*A.

    - Không có tham chiếu: hoà vốn so với không làm gì (B0): r* = C*A/S, C* = r*S/A.
    - Có tham chiếu (B1, B2): r* và C* để NB_mô_hình = NB_tham_chiếu; NaN nếu không có giao điểm dương.
    """
    S_m, A_m = stats_model["caught_damage_per_day"], stats_model["alerts_per_day"]
    S_r, A_r = (0.0, 0.0) if stats_ref is None else (stats_ref["caught_damage_per_day"], stats_ref["alerts_per_day"])
    dS, dA = S_m - S_r, A_m - A_r
    r_star = c_int * dA / dS if dS != 0 and (c_int * dA / dS) > 0 else np.nan
    c_star = r * dS / dA if dA != 0 and (r * dS / dA) > 0 else np.nan
    return {"delta_S_per_day": dS, "delta_alerts_per_day": dA, "r_star": r_star, "c_int_star": c_star}


def bootstrap_diff(daily_a: pd.Series, daily_b: pd.Series, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Bootstrap theo NGÀY cho hiệu lợi ích ròng trung bình/ngày (a - b)."""
    idx = daily_a.index.union(daily_b.index)
    d = (daily_a.reindex(idx, fill_value=0.0) - daily_b.reindex(idx, fill_value=0.0)).to_numpy()
    means = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"diff_mean": d.mean(), "ci_low": np.percentile(means, 2.5), "ci_high": np.percentile(means, 97.5),
            "p_diff_gt0": float((means > 0).mean()), "n_days": len(d)}


def bootstrap_mean(daily: pd.Series, n_boot: int = 2000, seed: int = SEED) -> tuple:
    """Khoảng tin cậy 95% bootstrap theo ngày cho lợi ích ròng trung bình/ngày."""
    v = daily.to_numpy()
    m = v[np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ----------------------------------------------------------------------------
# Báo cáo chi phí trọn gói cho một tập đánh giá (OOF hoặc test), một lăng kính
# ----------------------------------------------------------------------------
def cost_report(frame: pd.DataFrame, scores: dict, lens: str, setup: CostSetup, focus: list, source: str,
                k: int = K_BASE) -> dict:
    """Mọi bảng chi phí tại k cố định cho một tập đánh giá và một lăng kính.

    scores: {chính sách: mảng điểm} (mô hình, "<m>_EV", "<m>_EVcal", B0, B1, B2).
    focus: các chính sách cần bảng độ nhạy/hoà vốn (mô hình chính, chính sách được chọn, B1, B2, B0).
    """
    dmg = damage(frame, lens)
    base = setup.base(lens)
    out = {}
    curve = policy_curve(frame, scores, lens, setup)
    curve.insert(0, "source", source)
    out["curve"] = curve

    daily, rows = {}, []
    for m, sc in scores.items():
        al = policy_alerts(frame, sc, k, m)
        daily[m] = daily_net(frame, al, dmg, base["r"], base["c_int"])
        lo, hi = bootstrap_mean(daily[m])
        rows.append({"source": source, "policy": m, "k": 0 if m == "B0" else k, **evaluate_policy(frame, al, lens, setup, dmg),
                     "net_per_day_ci_low": lo, "net_per_day_ci_high": hi})
    out["summary"] = pd.DataFrame(rows)

    rows = []
    for m in scores:
        if m in ("B0", "B1", "B2"):
            continue
        for ref in ("B1", "B2"):
            rows.append({"source": source, "policy": m, "k": k, "ref": ref, **bootstrap_diff(daily[m], daily[ref])})
    out["bootstrap"] = pd.DataFrame(rows)

    fscores = {m: scores[m] for m in focus}
    out["sens_scenarios"] = sensitivity_scenarios(frame, fscores, lens, setup, k).assign(source=source)
    out["sens_one_at_a_time"] = sensitivity_one_at_a_time(frame, fscores, lens, setup, k).assign(source=source)
    out["net_grid"] = pd.concat([net_grid(frame, scores[m], lens, setup, k, m).assign(policy=m) for m in focus if m != "B0"],
                                ignore_index=True).assign(source=source)

    st = {m: policy_stats(frame, policy_alerts(frame, scores[m], k, m), dmg) for m in scores}
    be = []
    for m in focus:
        if m in ("B0", "B1", "B2"):
            continue
        for ref, label in ((None, "B0 (không làm gì)"), ("B1", "B1"), ("B2", "B2")):
            be.append({"source": source, "policy": m, "k": k, "ref": label,
                       **break_even(st[m], None if ref is None else st[ref], base["r"], base["c_int"])})
    out["break_even"] = pd.DataFrame(be)
    for key, t in out.items():
        t.insert(0, "lens", lens)
    return out


def per_fold_net(frame: pd.DataFrame, scores: dict, policies: list, lens: str, setup: CostSetup, k: int = K_BASE) -> pd.DataFrame:
    """Chính sách top-k tính riêng từng fold rolling-origin (để có trung bình ± độ lệch chuẩn qua fold)."""
    rows = []
    for m in policies:
        for f, g in frame.groupby("fold"):
            idx = g.index.to_numpy()
            al = policy_alerts(g, scores[m][idx], k, m)
            rows.append({"lens": lens, "policy": m, "k": 0 if m == "B0" else k, "fold": f, **evaluate_policy(g, al, lens, setup)})
    return pd.DataFrame(rows)
