"""Diễn giải mô hình (mục 7): permutation importance, PDP/ICE, cây độ sâu <= 3, hệ số logistic."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import SEED
from evaluate import select_X
from pipeline import SCENARIOS, build_preprocessor, feature_names, rank_score
from plots import finish


def permutation_importance_table(estimators: dict, test: pd.DataFrame, scenario: str = "A",
                                 n_repeats: int = 30) -> pd.DataFrame:
    """Permutation importance trên TEST theo average_precision. Xáo trộn từng cột đầu vào thô (trước pipeline)."""
    cols = SCENARIOS[scenario].columns
    X, y = test[cols], test["miss"].to_numpy()
    rows = []
    for model, est in estimators.items():
        r = permutation_importance(est, X, y, scoring="average_precision", n_repeats=n_repeats,
                                   random_state=SEED, n_jobs=1)
        for i, c in enumerate(cols):
            rows.append({"scenario": scenario, "model": model, "feature": c,
                         "mean": r.importances_mean[i], "std": r.importances_std[i]})
    return pd.DataFrame(rows)


def top_features(pi: pd.DataFrame, model: str, n: int = 4) -> list:
    return pi[pi["model"] == model].sort_values("mean", ascending=False)["feature"].head(n).tolist()


def pdp_table(est, X: pd.DataFrame, features: list, n_grid: int = 20, n_ice: int = 40) -> pd.DataFrame:
    """PDP + ICE tự cài đặt. Đặc trưng số: lưới theo phân vị 5-95%; phân loại: từng giá trị.

    wip chỉ tính trên các dòng sewing (finishing không có wip nên ép giá trị wip cho finishing là phi thực tế).
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for f in features:
        Xf = X[X["department"] == "sewing"] if f == "wip" else X
        numeric = pd.api.types.is_numeric_dtype(Xf[f]) and f != "team"
        if numeric:
            vals = np.unique(np.quantile(Xf[f].dropna(), np.linspace(0.05, 0.95, n_grid)))
        else:
            vals = np.sort(Xf[f].unique())
        sample = Xf.iloc[rng.choice(len(Xf), size=min(n_ice, len(Xf)), replace=False)]
        for order, v in enumerate(vals):
            Xv = Xf.copy(); Xv[f] = v
            rows.append({"feature": f, "curve": "pdp", "order": order, "x": float(v) if numeric else np.nan,
                         "x_label": str(v), "y": float(rank_score(est, Xv).mean()), "is_numeric": numeric})
            Xs = sample.copy(); Xs[f] = v
            for j, s in enumerate(rank_score(est, Xs)):
                rows.append({"feature": f, "curve": f"ice_{j}", "order": order, "x": float(v) if numeric else np.nan,
                             "x_label": str(v), "y": float(s), "is_numeric": numeric})
    return pd.DataFrame(rows)


def _readable_rule(name: str, thr: float, left: bool) -> str:
    """Điều kiện nút cây theo đơn vị gốc: left = (giá trị <= ngưỡng)."""
    binary = name.startswith(("department_", "day_", "team_", "is_first_obs"))
    if binary:
        return f"NOT {name}" if left else name
    if name == "log1p(wip)":
        t = np.expm1(thr)
        return f"wip <= {t:,.0f}" if left else f"wip > {t:,.0f}"
    return f"{name} <= {thr:.3g}" if left else f"{name} > {thr:.3g}"


def fit_interpretable_tree(train: pd.DataFrame, dt_params: dict, scenario: str = "A", max_depth: int = 3):
    """Cây để vẽ: tiền xử lý KHÔNG chuẩn hoá (ngưỡng theo đơn vị gốc), độ sâu <= 3, cùng min_samples_leaf đã chọn.

    Chuẩn hoá là biến đổi đơn điệu nên không làm đổi cấu trúc cây; hàm trả thêm tỷ lệ dự báo trùng
    với cây cùng siêu tham số trên dữ liệu chuẩn hoá để kiểm chứng điều này.
    """
    depth = min(max_depth, dt_params.get("clf__max_depth", max_depth))
    leaf = dt_params.get("clf__min_samples_leaf", 10)
    X, y = train[SCENARIOS[scenario].columns], train["miss"].to_numpy()
    prep = build_preprocessor(scenario, scale=False).fit(X)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=SEED)
    tree.fit(prep.transform(X), y)
    prep_s = build_preprocessor(scenario, scale=True).fit(X)
    tree_s = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=SEED)
    tree_s.fit(prep_s.transform(X), y)
    agreement = float((tree.predict(prep.transform(X)) == tree_s.predict(prep_s.transform(X))).mean())
    return prep, tree, agreement


def tree_leaf_table(prep, tree, train: pd.DataFrame, scenario: str = "A") -> pd.DataFrame:
    """Bảng các luật (lá) của cây: điều kiện, số dòng train, tỷ lệ trượt thực, xác suất theo trọng số cân bằng."""
    names = feature_names(prep)
    Xt = prep.transform(train[SCENARIOS[scenario].columns])
    leaf_id = tree.apply(Xt)
    t = tree.tree_
    rows = []
    for leaf in np.unique(leaf_id):
        path = tree.decision_path(Xt[np.flatnonzero(leaf_id == leaf)[:1]]).indices
        conds = []
        for node in path:
            if t.children_left[node] == -1:
                continue
            go_left = Xt[np.flatnonzero(leaf_id == leaf)[0], t.feature[node]] <= t.threshold[node]
            conds.append(_readable_rule(names[t.feature[node]], t.threshold[node], go_left))
        m = train["miss"].to_numpy()[leaf_id == leaf]
        w = t.value[leaf][0]
        rows.append({"leaf": int(leaf), "rule": " AND ".join(conds), "n_train": len(m), "n_miss": int(m.sum()),
                     "miss_rate_train": m.mean(), "p_miss_weighted": w[1] / w.sum()})
    return pd.DataFrame(rows).sort_values("p_miss_weighted", ascending=False).reset_index(drop=True)


def plot_tree_figure(prep, tree, name=None):
    """Hình 17: vẽ cây (xanh: nghiêng về 'đạt', cam: nghiêng về 'trượt'); ngưỡng theo đơn vị gốc."""
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(tree, feature_names=feature_names(prep), class_names=["đạt", "trượt"], filled=True, rounded=True,
              impurity=False, proportion=False, fontsize=9, ax=ax)
    ax.set_title(f"Cây quyết định độ sâu {tree.get_depth()} (class_weight='balanced'; "
                 "'value' là số mẫu có trọng số cân bằng; log1p(wip) = ln(1 + wip))", fontsize=10)
    ax.grid(False)
    return finish(fig, name)


def logistic_coefficients(est) -> pd.DataFrame:
    """Hệ số logistic (đặc trưng số đã chuẩn hoá; one-hot và is_first_obs theo thang 0/1), sắp theo |hệ số|."""
    names = feature_names(est.named_steps["prep"])
    df = pd.DataFrame({"feature": names, "coef": est.named_steps["clf"].coef_[0]})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
