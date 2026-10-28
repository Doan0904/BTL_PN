"""Learning curve theo thời gian.

Cố định một khối validate của train (khối 5 = khối cuối; lặp lại với khối 4 để thấy độ dao động),
huấn luyện trên m ngày LIỀN TRƯỚC khối đó (m = 5, 10, 15, ... đến tối đa), đo PR-AUC trên khối validate.
B1 là đường ngang vì không học tham số. Siêu tham số là bộ đã chọn trên train (không dò lại theo m).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from evaluate import fit_one, select_X
from pipeline import baseline_scores, rank_score

STEP = 5


def m_values(max_days: int, step: int = STEP) -> list:
    """5, 10, ..., và luôn có giá trị tối đa."""
    ms = list(range(step, max_days + 1, step))
    if ms[-1] != max_days:
        ms.append(max_days)
    return ms


def learning_curve(train: pd.DataFrame, cv, best_params: dict, scenario: str = "A", models=("Logistic",),
                   val_blocks=(5, 4)) -> pd.DataFrame:
    """Bảng: val_block, model, m_days, n_train, n_val, prevalence_val, pr_auc.

    best_params: {mô hình: siêu tham số}. Mỗi lần chỉ dùng các dòng của m ngày ngay trước khối validate.
    """
    block = cv.block_of_rows(train)
    X_all, y_all = select_X(train, scenario), train["miss"].to_numpy()
    rows = []
    for vb in val_blocks:
        va = np.flatnonzero(block == vb)
        first_val_day = train["date"].iloc[va].min()
        days_before = np.sort(train.loc[train["date"] < first_val_day, "date"].unique())
        prev = float(y_all[va].mean())
        b1 = average_precision_score(y_all[va], baseline_scores(train.iloc[va], "B1"))
        for m in m_values(len(days_before)):
            use = days_before[-m:]
            tr = np.flatnonzero(train["date"].isin(use).to_numpy())
            assert train["date"].iloc[tr].max() < first_val_day, "Train của learning curve phải đứng trước validate"
            base = {"val_block": vb, "m_days": m, "n_train": len(tr), "n_val": len(va), "prevalence_val": prev}
            rows.append({**base, "model": "B1", "pr_auc": b1})
            for name in models:
                est = fit_one(name, scenario, best_params[name], X_all.iloc[tr], y_all[tr])
                rows.append({**base, "model": name, "pr_auc": average_precision_score(y_all[va], rank_score(est, X_all.iloc[va]))})
    return pd.DataFrame(rows)
