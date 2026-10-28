"""ColumnTransformer, kịch bản đặc trưng A/B, mô hình và lưới siêu tham số.

Mọi bước học tham số từ dữ liệu (imputer, scaler, encoder) nằm TRONG Pipeline,
nên khi dùng với GridSearchCV/cross-validation chúng chỉ được fit trên phần train của từng fold.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import SEED
from features import FLAG_COLS, LAG_COLS


# ----------------------------------------------------------------------------
# Kịch bản đặc trưng
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    numeric: list = field(default_factory=list)       # số: median -> scaler
    wip: list = field(default_factory=lambda: ["wip"])  # wip: điền 0 -> log1p -> scaler
    lag: list = field(default_factory=lambda: list(LAG_COLS))  # trễ: median (ngày đầu của cặp)
    flags: list = field(default_factory=lambda: list(FLAG_COLS))  # is_first_obs, đi thẳng vào mô hình
    categorical: list = field(default_factory=lambda: ["department", "day", "team"])

    @property
    def columns(self) -> list:
        """Tên các cột đầu vào thô mà pipeline của kịch bản này dùng."""
        return self.numeric + self.wip + self.lag + self.flags + self.categorical


_BASE_NUMERIC = ["targeted_productivity", "smv", "no_of_workers", "week_of_month"]

SCENARIOS = {
    "A": Scenario("A", "Nghiêm ngặt: chỉ biến chắc chắn biết đầu ca + đặc trưng trễ (kịch bản chính)",
                  numeric=_BASE_NUMERIC),
    "B": Scenario("B", "Nới lỏng: A + incentive, over_time, no_of_style_change (phân tích độ nhạy)",
                  numeric=_BASE_NUMERIC + ["incentive", "over_time", "no_of_style_change"]),
}


def build_preprocessor(scenario: str = "A", scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer cho kịch bản A/B.

    scale=False chỉ dùng để vẽ cây quyết định với ngưỡng theo đơn vị gốc (cây không cần chuẩn hoá).
    """
    sc = SCENARIOS[scenario]
    scaler = [("scale", StandardScaler())] if scale else []
    num = Pipeline([("impute", SimpleImputer(strategy="median"))] + scaler)
    wip = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),   # thiếu = finishing -> 0
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ] + scaler)
    lag = Pipeline([("impute", SimpleImputer(strategy="median"))] + scaler)
    transformers = [
        ("num", num, sc.numeric),
        ("wip", wip, sc.wip),
        ("lag", lag, sc.lag),                                              # median học trong Pipeline
        ("flag", "passthrough", sc.flags),                                 # is_first_obs (0/1), không chuẩn hoá
        ("cat", OneHotEncoder(handle_unknown="ignore"), sc.categorical),
    ]
    return ColumnTransformer(transformers, remainder="drop")


# ----------------------------------------------------------------------------
# Mô hình và lưới siêu tham số (nhỏ, vì chỉ có ~950 dòng train)
# ----------------------------------------------------------------------------
MAIN_FAMILIES = ["DecisionTree", "SVM_RBF", "MLP"]       # họ chính (Tuần 6-13)
CONTROL_FAMILIES = ["Logistic", "RandomForest"]           # đối chứng mở rộng
MODEL_NAMES = MAIN_FAMILIES + CONTROL_FAMILIES

MODEL_LABELS = {
    "DecisionTree": "Cây quyết định", "SVM_RBF": "SVM (RBF)", "MLP": "Mạng nơ-ron (MLP)",
    "Logistic": "Hồi quy logistic", "RandomForest": "Random Forest",
}


def make_classifier(name: str):
    """Bộ phân loại với giá trị mặc định cố định; class_weight='balanced' nơi hỗ trợ (MLP không hỗ trợ)."""
    if name == "DecisionTree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED)
    if name == "SVM_RBF":
        # Không bật probability=True (tốn thời gian, thêm CV nội bộ); xếp hạng dùng decision_function.
        return SVC(kernel="rbf", class_weight="balanced", random_state=SEED)
    if name == "MLP":
        return MLPClassifier(solver="adam", max_iter=1000, early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=20, random_state=SEED)
    if name == "Logistic":
        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1)
    raise KeyError(name)


# lưới nhỏ (dữ liệu ít); sau khi chạy có kiểm tra tham số nào nằm ở biên lưới (grid_boundary_check.csv)
# và thăm dò thêm một nấc ra ngoài biên (grid_edge_probe.csv).
PARAM_GRIDS = {
    "DecisionTree": {"clf__max_depth": [2, 3, 4, 5, 6], "clf__min_samples_leaf": [5, 10, 20, 30, 40]},
    "SVM_RBF": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.003, 0.01, 0.03]},
    "MLP": {"clf__hidden_layer_sizes": [(8,), (16,), (32,), (64,), (16, 8)], "clf__alpha": [1e-3, 1e-2, 1e-1, 1.0]},
    "Logistic": {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "RandomForest": {"clf__min_samples_leaf": [2, 3, 5, 10, 20], "clf__max_depth": [4, 8, None]},
}
# Thang thứ tự để kiểm tra biên lưới: "numeric" = sắp theo giá trị số; danh sách = thứ tự tường minh (tuỳ chọn ngoài
# danh sách, vd (16, 8), không thuộc thang nên không tính là biên); "lower_only" = chỉ có biên dưới ('scale' nằm ngoài
# thang); None = không kiểm tra (max_depth=None của RF là "không giới hạn", không mở rộng thêm được).
GRID_ORDER = {
    ("DecisionTree", "clf__max_depth"): "numeric", ("DecisionTree", "clf__min_samples_leaf"): "numeric",
    ("SVM_RBF", "clf__C"): "numeric", ("SVM_RBF", "clf__gamma"): "lower_only",
    ("MLP", "clf__hidden_layer_sizes"): [(8,), (16,), (32,), (64,)], ("MLP", "clf__alpha"): "numeric",
    ("Logistic", "clf__C"): "numeric",
    ("RandomForest", "clf__min_samples_leaf"): "numeric", ("RandomForest", "clf__max_depth"): None,
}


def build_model_pipeline(name: str, scenario: str = "A", scale: bool = True) -> Pipeline:
    """Pipeline hoàn chỉnh: tiền xử lý (prep) + bộ phân loại (clf)."""
    return Pipeline([("prep", build_preprocessor(scenario, scale=scale)), ("clf", make_classifier(name))])


def feature_names(prep: ColumnTransformer) -> list:
    """Tên đặc trưng sau biến đổi (bỏ tiền tố của ColumnTransformer; wip -> log1p(wip))."""
    names = []
    for raw in prep.get_feature_names_out():
        n = raw.split("__", 1)[1] if "__" in raw else raw
        names.append("log1p(wip)" if raw.startswith("wip__") else n)
    return names


def raw_feature_of(name: str, scenario: str = "A") -> str:
    """Ánh xạ tên đặc trưng sau biến đổi về cột đầu vào thô (vd 'team_3' -> 'team', 'log1p(wip)' -> 'wip')."""
    if name == "log1p(wip)":
        return "wip"
    for c in SCENARIOS[scenario].categorical:
        if name.startswith(c + "_"):
            return c
    return name


# ----------------------------------------------------------------------------
# Baseline: không học tham số; chỉ dùng đặc trưng trễ có sẵn
# ----------------------------------------------------------------------------
BASELINES = ["B0", "B1", "B2"]
BASELINE_LABELS = {
    "B0": "B0: lớp đa số (không cảnh báo)",
    "B1": "B1: hôm qua trượt -> cảnh báo",
    "B2": "B2: mọi tổ finishing",
}


def baseline_scores(df: pd.DataFrame, name: str) -> np.ndarray:
    """Điểm xếp hạng cho baseline. B1/B2 xếp theo lag1_gap (âm nhất trước) để phá hoà."""
    gap = df["lag1_gap"].fillna(0.0).to_numpy()   # ngày đầu của cặp: coi như không có thông tin
    if name == "B0":
        return np.zeros(len(df))
    if name == "B1":
        return -gap                               # lag1_miss = 1 <=> lag1_gap < 0, nên xếp theo -gap đã đặt các tổ trượt hôm qua lên đầu
    if name == "B2":
        return 10.0 * (df["department"] == "finishing").to_numpy() - gap
    raise KeyError(name)


def baseline_predictions(df: pd.DataFrame, name: str) -> np.ndarray:
    """Dự báo nhị phân của baseline (để tính F2/recall/precision)."""
    if name == "B0":
        return np.zeros(len(df), dtype=int)
    if name == "B1":
        return df["lag1_miss"].fillna(0.0).to_numpy().astype(int)
    if name == "B2":
        return (df["department"] == "finishing").to_numpy().astype(int)
    raise KeyError(name)


def rank_score(estimator, X) -> np.ndarray:
    """Điểm rủi ro dùng để xếp hạng: predict_proba[:, 1] nếu có, ngược lại decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)
