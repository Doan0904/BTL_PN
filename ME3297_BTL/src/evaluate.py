"""Dò siêu tham số (GridSearchCV + rolling-origin), dự báo out-of-fold, đánh giá test, thí nghiệm rò rỉ."""
import ast
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, fbeta_score, make_scorer, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from config import SEED
from pipeline import (BASELINES, GRID_ORDER, MODEL_NAMES, PARAM_GRIDS, SCENARIOS, baseline_predictions, baseline_scores,
                      build_model_pipeline, rank_score)

METRICS = ["average_precision", "roc_auc", "f2", "recall", "precision"]
METRIC_LABELS = {"average_precision": "PR-AUC", "roc_auc": "ROC-AUC", "f2": "F2", "recall": "Recall", "precision": "Precision"}

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
}


def select_X(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Cột đầu vào của kịch bản + cột date (bộ chia cần; ColumnTransformer sẽ bỏ qua date)."""
    return df[["date"] + SCENARIOS[scenario].columns]


def metric_row(y, score, pred) -> dict:
    """5 chỉ số: PR-AUC, ROC-AUC (theo điểm) và F2, recall, precision (theo dự báo nhị phân)."""
    return {
        "average_precision": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
    }


def params_to_str(params: dict) -> str:
    return repr({k: params[k] for k in sorted(params)})


def params_from_str(s: str) -> dict:
    return ast.literal_eval(s)


# ----------------------------------------------------------------------------
# GridSearchCV với rolling-origin theo ngày
# ----------------------------------------------------------------------------
def run_grid_search(model: str, scenario: str, train: pd.DataFrame, cv, n_jobs: int = 2) -> GridSearchCV:
    """Dò lưới nhỏ trên tập train; chọn theo PR-AUC trung bình qua các fold."""
    gs = GridSearchCV(
        build_model_pipeline(model, scenario), PARAM_GRIDS[model], scoring=SCORING, refit="average_precision",
        cv=cv, n_jobs=n_jobs, error_score="raise", return_train_score=False,
    )
    with warnings.catch_warnings():
        # Bộ lọc phòng hờ: đã kiểm tra trực tiếp, với lưới hiện tại không có ConvergenceWarning nào xuất hiện
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(select_X(train, scenario), train["miss"].to_numpy())
    return gs


def grid_results_table(gs: GridSearchCV, model: str, scenario: str) -> pd.DataFrame:
    """Toàn bộ kết quả lưới (mean/std theo fold cho 5 chỉ số) để đưa vào phụ lục."""
    res = pd.DataFrame(gs.cv_results_)
    out = pd.DataFrame({"scenario": scenario, "model": model, "params": res["params"].map(params_to_str)})
    for m in METRICS:
        out[f"{m}_mean"] = res[f"mean_test_{m}"]
        out[f"{m}_std"] = res[f"std_test_{m}"]
    out["rank_by_pr_auc"] = res["rank_test_average_precision"]
    out["is_best"] = np.arange(len(res)) == gs.best_index_
    return out


# ----------------------------------------------------------------------------
# Dự báo out-of-fold (rolling-origin) cho mô hình đã chọn
# ----------------------------------------------------------------------------
def fit_one(model: str, scenario: str, params: dict, X, y):
    """Huấn luyện một pipeline mới với siêu tham số cho trước."""
    est = build_model_pipeline(model, scenario).set_params(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(X, y)
    return est


def oof_predict(model: str, scenario: str, params: dict, train: pd.DataFrame, cv, gs: GridSearchCV | None = None):
    """Dự báo out-of-fold trên các khối validate (khối 0 không có dự báo vì chỉ dùng để huấn luyện).

    Trả về (predictions, fold_metrics). Nếu truyền gs, kiểm tra điểm PR-AUC từng fold khớp với GridSearchCV.
    """
    X, y = select_X(train, scenario), train["miss"].to_numpy()
    pred_rows, metric_rows = [], []
    for f, (tr, va) in enumerate(cv.split(X), start=1):
        est = fit_one(model, scenario, params, X.iloc[tr], y[tr])
        score, pred = rank_score(est, X.iloc[va]), est.predict(X.iloc[va])
        pred_rows.append(pd.DataFrame({"scenario": scenario, "model": model, "row_id": train["row_id"].iloc[va].to_numpy(),
                                       "fold": f, "score": score, "pred": pred}))
        metric_rows.append({"scenario": scenario, "model": model, "fold": f, "n_val": len(va),
                            "prevalence": y[va].mean(), **metric_row(y[va], score, pred)})
    folds = pd.DataFrame(metric_rows)
    if gs is not None:
        ref = np.array([gs.cv_results_[f"split{i}_test_average_precision"][gs.best_index_] for i in range(len(folds))])
        assert np.allclose(ref, folds["average_precision"].to_numpy(), atol=1e-9), \
            f"PR-AUC từng fold không khớp GridSearchCV ({model}, {scenario})"
    return pd.concat(pred_rows, ignore_index=True), folds


def baseline_oof(train: pd.DataFrame, cv, names=BASELINES):
    """Baseline trên cùng các khối validate: trả về (predictions, fold_metrics)."""
    pred_rows, metric_rows = [], []
    X = train[["date"]]
    for f, (_, va) in enumerate(cv.split(X), start=1):
        part = train.iloc[va]
        y = part["miss"].to_numpy()
        for b in names:
            score, pred = baseline_scores(part, b), baseline_predictions(part, b)
            pred_rows.append(pd.DataFrame({"scenario": "-", "model": b, "row_id": part["row_id"].to_numpy(),
                                           "fold": f, "score": score, "pred": pred}))
            metric_rows.append({"scenario": "-", "model": b, "fold": f, "n_val": len(va), "prevalence": y.mean(),
                                **metric_row(y, score, pred)})
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(metric_rows)


def summarize_folds(folds: pd.DataFrame, cols=METRICS, by=("scenario", "model")) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn mẫu (ddof=1) qua các fold."""
    g = folds.groupby(list(by), sort=False)
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std"))
    out["n_folds"] = g.size()
    return out.reset_index()


# ----------------------------------------------------------------------------
# Đánh giá trên test (gọi đúng một lần ở cuối)
# ----------------------------------------------------------------------------
def fit_final_and_predict_test(model: str, scenario: str, params: dict, train: pd.DataFrame, test: pd.DataFrame):
    """Fit lại trên TOÀN BỘ train với siêu tham số đã chọn, dự báo test. Trả về (estimator, predictions)."""
    est = fit_one(model, scenario, params, select_X(train, scenario), train["miss"].to_numpy())
    Xte = select_X(test, scenario)
    preds = pd.DataFrame({"scenario": scenario, "model": model, "row_id": test["row_id"].to_numpy(), "fold": 0,
                          "score": rank_score(est, Xte), "pred": est.predict(Xte)})
    return est, preds


def baseline_test(test: pd.DataFrame, names=BASELINES) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"scenario": "-", "model": b, "row_id": test["row_id"].to_numpy(), "fold": 0,
                                    "score": baseline_scores(test, b), "pred": baseline_predictions(test, b)})
                      for b in names], ignore_index=True)


def metrics_from_predictions(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Chỉ số tổng hợp trên toàn bộ tập (dùng cho test). labels cần cột row_id, miss."""
    merged = pred.merge(labels[["row_id", "miss"]], on="row_id", how="left", validate="m:1")
    rows = []
    for (sc, m), g in merged.groupby(["scenario", "model"], sort=False):
        rows.append({"scenario": sc, "model": m, "n": len(g), "prevalence": g["miss"].mean(),
                     **metric_row(g["miss"].to_numpy(), g["score"].to_numpy(), g["pred"].to_numpy())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Thí nghiệm rò rỉ: k-fold ngẫu nhiên vs rolling-origin (cùng mô hình, cùng siêu tham số)
# ----------------------------------------------------------------------------
def random_cv_folds(model: str, scenario: str, params: dict, train: pd.DataFrame,
                    n_splits: int = 5, n_repeats: int = 3) -> pd.DataFrame:
    """RepeatedStratifiedKFold(5x3, seed 42) trên tập train; trả về chỉ số từng fold."""
    X, y = select_X(train, scenario), train["miss"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for i, (tr, va) in enumerate(rskf.split(X, y), start=1):
        est = fit_one(model, scenario, params, X.iloc[tr], y[tr])
        rows.append({"scenario": scenario, "model": model, "fold": i, "n_val": len(va), "prevalence": y[va].mean(),
                     **metric_row(y[va], rank_score(est, X.iloc[va]), est.predict(X.iloc[va]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Quy trình trọn gói cho một kịch bản
# ----------------------------------------------------------------------------
def select_models(train: pd.DataFrame, scenario: str, cv, models=MODEL_NAMES, n_jobs: int = 2) -> dict:
    """Dò lưới + dự báo OOF cho mọi mô hình của một kịch bản (KHÔNG đụng tới test).

    Trả về dict: grid (bảng mọi cấu hình), best (tên -> siêu tham số), oof (dự báo OOF), folds (chỉ số từng fold).
    """
    grids, best, oofs, folds = [], {}, [], []
    for m in models:
        gs = run_grid_search(m, scenario, train, cv, n_jobs=n_jobs)
        best[m] = dict(gs.best_params_)
        grids.append(grid_results_table(gs, m, scenario))
        o, f = oof_predict(m, scenario, best[m], train, cv, gs)
        oofs.append(o); folds.append(f)
    return {"grid": pd.concat(grids, ignore_index=True), "best": best,
            "oof": pd.concat(oofs, ignore_index=True), "folds": pd.concat(folds, ignore_index=True)}


def align_scores(frame: pd.DataFrame, preds: pd.DataFrame, scenario: str) -> dict:
    """{mô hình: điểm} căn theo thứ tự dòng của frame (qua row_id); thêm điểm baseline tính từ frame."""
    scores = {}
    for m, g in preds[preds["scenario"] == scenario].groupby("model", sort=False):
        s = g.set_index("row_id")["score"]
        scores[m] = s.reindex(frame["row_id"]).to_numpy()
        assert not np.isnan(scores[m]).any(), f"Thiếu dự báo cho {m}"
    for b in BASELINES:
        scores[b] = baseline_scores(frame, b)
    return scores


def oof_frame(train: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Các dòng train có dự báo OOF (các khối validate 1..5; khối 0 không có) kèm số fold."""
    fold = preds.drop_duplicates("row_id").set_index("row_id")["fold"]
    out = train[train["row_id"].isin(fold.index)].copy()
    out["fold"] = out["row_id"].map(fold)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# kiểm tra siêu tham số tốt nhất có nằm ở biên lưới không
# ----------------------------------------------------------------------------
def grid_boundary_check(best_by_scenario: dict) -> pd.DataFrame:
    """Bảng: với mỗi (kịch bản, mô hình, tham số) giá trị tốt nhất có nằm ở biên dưới/biên trên của lưới không.

    best_by_scenario = {kịch bản: {mô hình: siêu tham số tốt nhất}}.
    """
    rows = []
    for sc, best in best_by_scenario.items():
        for m, grid in PARAM_GRIDS.items():
            for param, vals in grid.items():
                kind = GRID_ORDER.get((m, param))
                bv = best[m][param]
                note, lo, hi = "", False, False
                if kind is None:
                    note = "không kiểm tra (max_depth=None là 'không giới hạn')"
                elif kind == "numeric":
                    order = sorted(v for v in vals if isinstance(v, (int, float)) and v is not None)
                    lo, hi = bv == order[0], bv == order[-1]
                elif kind == "lower_only":
                    order = sorted(v for v in vals if isinstance(v, (int, float)))
                    lo = bv == order[0]
                    note = "'scale' nằm ngoài thang số nên không xác định biên trên"
                else:
                    lo, hi = bv == kind[0], bv == kind[-1]
                    if bv not in kind:
                        note = "tuỳ chọn ngoài thang thứ tự (không tính là biên)"
                rows.append({"scenario": sc, "model": m, "param": param, "grid": repr(vals), "best": repr(bv),
                             "at_lower_edge": lo, "at_upper_edge": hi, "at_edge": lo or hi, "note": note})
    return pd.DataFrame(rows)
