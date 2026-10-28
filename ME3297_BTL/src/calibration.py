"""Độ tin cậy của xác suất (reliability, Brier) và hiệu chỉnh sigmoid (Platt) không rò rỉ theo thời gian.

Hiệu chỉnh không rò rỉ: bộ hiệu chỉnh cho khối validate j chỉ được fit trên điểm OUT-OF-FOLD của các khối < j
(các điểm này đến từ mô hình chỉ thấy dữ liệu còn sớm hơn nữa, tức là cùng bộ chia RollingOriginByDate).
Vì vậy khối validate 1 không có bản hiệu chỉnh; các so sánh "đã/chưa hiệu chỉnh" dùng chung các khối 2..5.
Với test: bộ hiệu chỉnh fit trên TOÀN BỘ điểm OOF của train rồi áp lên điểm của mô hình fit trên toàn bộ train.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

N_BINS = 5          # số nhóm theo phân vị của p (OOF khoảng 780 dòng, test 254 dòng)
OFF_GAP = 0.05      # |trung bình p - tỷ lệ trượt thực| vượt ngưỡng này thì coi là lệch rõ
EPS = 1e-3


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def reliability_bins(y, p, n_bins: int = N_BINS) -> pd.DataFrame:
    """Chia theo phân vị của p thành n_bins nhóm (đều số dòng); trả trung bình p và tỷ lệ trượt thực của nhóm."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins)):
        rows.append({"bin": b + 1, "n": len(idx), "mean_pred": p[idx].mean(), "frac_pos": y[idx].mean()})
    return pd.DataFrame(rows)


def calib_stats(y, p) -> dict:
    """Brier, Brier của hằng số (tỷ lệ trượt của chính tập đó), hiệu lệch trung bình, ECE, độ dốc/chặn hiệu chỉnh."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = reliability_bins(y, p)
    ece = float((bins["n"] * (bins["mean_pred"] - bins["frac_pos"]).abs()).sum() / bins["n"].sum())
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y.astype(int))
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y.mean() - y) ** 2))
    return {"n": len(y), "prevalence": y.mean(), "mean_pred": p.mean(), "gap_mean_pred_minus_prev": p.mean() - y.mean(),
            "brier": brier, "brier_const": ref, "brier_skill": 1 - brier / ref, "ece": ece,
            "calib_slope": float(lr.coef_[0, 0]), "calib_intercept": float(lr.intercept_[0])}


def is_clearly_off(st: dict) -> bool:
    """Lệch rõ (xét trên OOF): trung bình p lệch > 0,05 so với tỷ lệ trượt, hoặc Brier tệ hơn dự báo hằng số."""
    return bool(abs(st["gap_mean_pred_minus_prev"]) > OFF_GAP or st["brier"] > st["brier_const"])


class Sigmoid:
    """Platt scaling: hồi quy logistic một biến trên logit(p) (gần như không phạt)."""

    def __init__(self, p_fit, y_fit):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_fit).reshape(-1, 1), np.asarray(y_fit).astype(int))

    def predict(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def calibrate_oof(oof: pd.DataFrame, y_by_row: pd.Series) -> pd.DataFrame:
    """Điểm OOF đã hiệu chỉnh cho fold >= 2, mỗi fold dùng bộ hiệu chỉnh fit trên các fold trước nó.

    oof: cột row_id, fold, score (xác suất chưa hiệu chỉnh). y_by_row: nhãn theo row_id.
    """
    out = []
    for f in sorted(oof["fold"].unique()):
        if f < 2:
            continue
        past, cur = oof[oof["fold"] < f], oof[oof["fold"] == f]
        cal = Sigmoid(past["score"].to_numpy(), y_by_row.loc[past["row_id"]].to_numpy())
        out.append(pd.DataFrame({"row_id": cur["row_id"].to_numpy(), "fold": f, "score_raw": cur["score"].to_numpy(),
                                 "score_cal": cal.predict(cur["score"].to_numpy())}))
    return pd.concat(out, ignore_index=True)


def calibrate_test(oof: pd.DataFrame, y_by_row: pd.Series, test_score: np.ndarray) -> np.ndarray:
    """Bộ hiệu chỉnh fit trên toàn bộ điểm OOF của train, áp lên điểm test."""
    return Sigmoid(oof["score"].to_numpy(), y_by_row.loc[oof["row_id"]].to_numpy()).predict(test_score)
