"""Chạy toàn bộ quy trình và ghi MỌI bảng vào outputs/tables/*.csv, MỌI hình vào outputs/figures/*.png.

Cách chạy (từ thư mục gốc repo, sau khi đã có data/garments_worker_productivity.csv):
    python src/run_all.py

Thứ tự: kiểm tra dữ liệu và EDA -> dò siêu tham số + dự báo OOF (kịch bản A, B) -> mô hình chính theo PR-AUC CV
-> hiệu chỉnh xác suất trên OOF -> chi phí hai lăng kính + chọn chính sách trên OOF -> learning curve
-> thí nghiệm rò rỉ -> ĐÁNH GIÁ TEST (một lần) -> kiểm tra cấu hình gần hoà -> chi phí trên test -> diễn giải -> độ nhạy khi loại định mức thấp -> phiếu cảnh báo mẫu
-> RESULTS_SUMMARY.md.
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import alert_sheet as AS
import calibration as CAL
import cost as C
import eda
import evaluate as E
import interpret as I
import learning_curve as LC
import plots as P
from config import DATA_FILE, ROOT, TABLES_DIR, ensure_dirs
from features import assert_lags_causal, assert_lags_first_day_nan, build_dataset, load_raw
from pipeline import BASELINES, MODEL_NAMES, SCENARIOS, feature_names
from splits import (RollingOriginByDate, TEST_START, assert_cv_folds_valid, assert_no_date_overlap, split_train_test)

T0 = time.time()
FORBIDDEN_FEATURES = {"actual_productivity", "gap", "miss", "idle_time", "idle_men", "quarter", "date", "row_id", "damage_sam"}
PROBA_MODELS = ["DecisionTree", "MLP", "Logistic", "RandomForest"]   # có xác suất đúng nghĩa (SVM chỉ có decision_function)
SAMPLE_SHEET_DATES = ["2015-02-26", "2015-03-05", "2015-03-11"]      # phiếu mẫu cho các ngày test


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Ghi bảng CSV (6 chữ số có nghĩa để ổn định giữa các lần chạy)."""
    ensure_dirs()
    df.to_csv(TABLES_DIR / name, index=False, float_format="%.6g")
    return df


def save_fig(fig, name):
    """finish() đã lưu khi truyền name; ở đây chỉ đóng figure để giải phóng bộ nhớ."""
    import matplotlib.pyplot as plt
    plt.close(fig)


def label_scenario(df, scenario, source=None):
    df = df.copy()
    df.insert(0, "scenario", scenario)
    if source:
        df.insert(1, "source", source)
    return df


# ----------------------------------------------------------------------------
# Kiểm tra chống rò rỉ
# ----------------------------------------------------------------------------
class Checks:
    """Ghi lại các assert chống rò rỉ đã chạy (để có bảng leakage_checks.csv)."""

    def __init__(self):
        self.rows = []

    def ok(self, name, detail):
        self.rows.append({"kiem_tra": name, "ket_qua": "đạt", "chi_tiet": detail})

    def table(self):
        return pd.DataFrame(self.rows)


def check_pipeline_fit_on_train(est, train, test, df, scenario, checks: Checks, label):
    """Tham số học trong pipeline phải khớp thống kê của TRAIN, không phải của test hay toàn bộ dữ liệu."""
    prep = est.named_steps["prep"]
    sc = SCENARIOS[scenario]
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    assert np.allclose(scaler.mean_, train[sc.numeric].mean().to_numpy()), "Scaler không fit trên train"
    assert not np.allclose(scaler.mean_, df[sc.numeric].mean().to_numpy(), rtol=1e-9, atol=1e-12), \
        "Scaler trùng thống kê toàn bộ dữ liệu: nghi ngờ đã fit cả test"
    med = prep.named_transformers_["lag"].named_steps["impute"].statistics_
    assert np.allclose(med, train[sc.lag].median().to_numpy(), equal_nan=True), "Imputer lag không fit trên train"
    checks.ok(f"Pipeline fit chỉ trên train ({label})", "mean_ của StandardScaler và median của imputer khớp thống kê train, khác thống kê toàn bộ")


# ----------------------------------------------------------------------------
# Giai đoạn 1: dữ liệu và EDA
# ----------------------------------------------------------------------------
def stage_data_eda(raw, df, train, test, cv, checks):
    log("EDA và bảng chất lượng dữ liệu")
    assert_lags_first_day_nan(df)
    checks.ok("Đặc trưng trễ ngày đầu của mỗi cặp (team, department) là NaN",
              f"{int((~df.duplicated(['team', 'department'])).sum())} cặp; 4 cột trễ đều NaN ở dòng đầu, không NaN ở các dòng sau; is_first_obs = 1 đúng ở dòng đầu")
    assert_lags_causal(df)
    checks.ok("Đặc trưng trễ nhân quả (kiểm tra nhiễu tương lai)",
              "Thay năng suất từ ngày d trở đi bằng nhiễu ngẫu nhiên không làm đổi đặc trưng trễ của các dòng có ngày <= d (5 ngày d thử)")
    assert_no_date_overlap(train, test)
    checks.ok("Không ngày nào xuất hiện ở cả train và test",
              f"train {train['date'].min().date()}..{train['date'].max().date()} ({train['date'].nunique()} ngày); "
              f"test {test['date'].min().date()}..{test['date'].max().date()} ({test['date'].nunique()} ngày)")
    assert_cv_folds_valid(cv, train)
    checks.ok("Rolling-origin: mỗi fold validate đứng sau train, không chia ngày", f"{cv.get_n_splits()} fold, {cv.n_blocks} khối ngày")
    for s, sc in SCENARIOS.items():
        assert FORBIDDEN_FEATURES.isdisjoint(sc.columns), f"Kịch bản {s} chứa cột cấm"
    assert not {"incentive", "over_time", "no_of_style_change"} & set(SCENARIOS["A"].columns)
    checks.ok("Kịch bản A không chứa incentive/over_time/no_of_style_change/idle_*; không kịch bản nào chứa cột nhãn",
              f"A: {SCENARIOS['A'].columns}")

    save_table(eda.quality_checks(raw, df), "eda_quality_checks.csv")
    save_table(eda.data_dictionary(df), "eda_data_dictionary.csv")
    save_table(eda.missing_outliers(df), "eda_missing_outliers.csv")
    save_table(eda.target_distribution(df, train, test), "eda_target_distribution.csv")
    save_table(eda.rows_per_day(df), "eda_rows_per_day.csv")
    save_table(eda.damage_summary(df), "eda_damage_per_miss.csv")
    save_table(eda.feature_scenarios_table(), "feature_scenarios.csv")
    save_table(cv.fold_table(train), "rolling_origin_folds.csv")
    save_table(pd.DataFrame([
        {"tap": "train", "tu_ngay": train["date"].min().date(), "den_ngay": train["date"].max().date(), "n_ngay": train["date"].nunique(),
         "n_dong": len(train), "ty_le_miss": train["miss"].mean()},
        {"tap": "test", "tu_ngay": test["date"].min().date(), "den_ngay": test["date"].max().date(), "n_ngay": test["date"].nunique(),
         "n_dong": len(test), "ty_le_miss": test["miss"].mean()},
    ]), "split_summary.csv")

    save_fig(P.plot_productivity_distribution(df, "fig01_phan_bo_nang_suat.png"), None)
    save_fig(P.plot_miss_rates(df, "fig02_ty_le_miss.png"), None)
    save_fig(P.plot_wip_missing(df, "fig03_ma_tran_thieu_wip.png"), None)
    save_fig(P.plot_boxplots(df, "fig04_hop_rau.png"), None)
    save_fig(P.plot_miss_timeseries(df, TEST_START, "fig05_chuoi_thoi_gian_miss.png"), None)
    save_fig(P.plot_wip_log1p(df, "fig22_wip_truoc_sau_log1p.png"), None)
    save_fig(P.plot_wip_pipeline(train, "fig23_wip_trong_pipeline.png"), None)
    save_fig(P.plot_rolling_origin(train, test, cv, "fig06_rolling_origin.png"), None)


# ----------------------------------------------------------------------------
# Giai đoạn 2: dò siêu tham số, OOF
# ----------------------------------------------------------------------------
def stage_selection(train, cv):
    sel = {}
    for s in ("A", "B"):
        log(f"Dò siêu tham số + OOF, kịch bản {s}")
        sel[s] = E.select_models(train, s, cv)
    base_pred, base_folds = E.baseline_oof(train, cv)
    return sel, base_pred, base_folds


def save_selection_tables(sel, base_pred, base_folds):
    save_table(pd.concat([sel[s]["grid"] for s in sel], ignore_index=True), "grid_search_results.csv")
    bp = [{"scenario": s, "model": m, "params": E.params_to_str(p)} for s in sel for m, p in sel[s]["best"].items()]
    save_table(pd.DataFrame(bp), "best_params.csv")
    folds = pd.concat([sel[s]["folds"] for s in sel] + [base_folds], ignore_index=True)
    save_table(folds, "cv_fold_scores.csv")
    summ = E.summarize_folds(folds)
    save_table(summ, "cv_summary.csv")
    oof = pd.concat([sel[s]["oof"] for s in sel] + [base_pred], ignore_index=True)
    save_table(oof, "predictions_oof.csv")
    return folds, summ, oof


# ----------------------------------------------------------------------------
# Giai đoạn 3: mô hình chính, hiệu chỉnh, chọn chính sách - chỉ dùng dữ liệu train/OOF
# ----------------------------------------------------------------------------
# Một nấc nữa vượt biên còn bị chạm, theo đúng phía bị chạm (chỉ thăm dò để ghi chú, KHÔNG đổi lưới/lựa chọn chính thức)
EDGE_PROBE = {
    ("DecisionTree", "clf__max_depth"): {"lower": 1, "upper": 8},
    ("DecisionTree", "clf__min_samples_leaf"): {"lower": 2, "upper": 60},
    ("SVM_RBF", "clf__C"): {"lower": 0.01, "upper": 100.0},
    ("SVM_RBF", "clf__gamma"): {"lower": 0.001},
    ("MLP", "clf__hidden_layer_sizes"): {"lower": (4,), "upper": (128,)},
    ("MLP", "clf__alpha"): {"lower": 1e-4, "upper": 10.0},
    ("Logistic", "clf__C"): {"lower": 0.001, "upper": 1000.0},
    ("RandomForest", "clf__min_samples_leaf"): {"lower": 1, "upper": 40},
}


def stage_edge_probe(train, cv, sel, boundary):
    """Với mỗi tham số nằm ở biên lưới, so PR-AUC CV của giá trị biên với một nấc ngoài biên
    (các tham số khác giữ nguyên giá trị tốt nhất). Chỉ để báo cáo, không thay đổi mô hình được chọn."""
    from sklearn.model_selection import cross_validate
    rows = []
    for r in boundary[boundary["at_edge"]].itertuples():
        side = "lower" if r.at_lower_edge else "upper"
        extra = EDGE_PROBE.get((r.model, r.param), {}).get(side)
        if extra is None:
            continue
        best = sel[r.scenario]["best"][r.model]
        X, y = E.select_X(train, r.scenario), train["miss"].to_numpy()
        res = {}
        for label, val in (("at_edge", best[r.param]), ("beyond_edge", extra)):
            est = E.build_model_pipeline(r.model, r.scenario).set_params(**{**best, r.param: val})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sc = cross_validate(est, X, y, cv=cv, scoring="average_precision")["test_score"]
            res[label] = (val, sc.mean(), sc.std(ddof=1))
        rows.append({"scenario": r.scenario, "model": r.model, "param": r.param, "edge": side, "value_at_edge": repr(res["at_edge"][0]),
                     "cv_pr_auc_at_edge": res["at_edge"][1], "value_beyond": repr(res["beyond_edge"][0]),
                     "cv_pr_auc_beyond": res["beyond_edge"][1], "delta": res["beyond_edge"][1] - res["at_edge"][1]})
    return pd.DataFrame(rows)


def stage_main_model(summ, sel):
    """Mô hình chính = PR-AUC CV (rolling-origin, kịch bản A) trung bình cao nhất; k cố định = 3.

    policy_model (dùng cho chính sách thiệt hại kỳ vọng và phiếu cảnh báo) = mô hình chính nếu có xác suất, ngược lại là mô hình có xác suất tốt nhất.
    """
    a = summ[(summ["scenario"] == "A") & summ["model"].isin(MODEL_NAMES)].copy()
    a["order"] = a["model"].map({m: i for i, m in enumerate(MODEL_NAMES)})
    a = a.sort_values(["average_precision_mean", "order"], ascending=[False, True], kind="stable").reset_index(drop=True)
    main = a["model"].iloc[0]
    pol = main if main in PROBA_MODELS else a[a["model"].isin(PROBA_MODELS)]["model"].iloc[0]
    t = pd.DataFrame({"model": a["model"], "rank_cv_pr_auc": np.arange(1, len(a) + 1),
                      "cv_pr_auc_mean": a["average_precision_mean"], "cv_pr_auc_std": a["average_precision_std"],
                      "has_probability": a["model"].isin(PROBA_MODELS),
                      "params": [E.params_to_str(sel["A"]["best"][m]) for m in a["model"]]})
    t["is_main"] = t["model"] == main
    t["is_policy_model"] = t["model"] == pol
    return main, pol, t


def stage_calibration_oof(train, sel):
    """Độ tin cậy của xác suất trên OOF kịch bản A: thống kê của p, và p hiệu chỉnh sigmoid không rò rỉ (khối >= 2)."""
    y = train.set_index("row_id")["miss"]
    oof = sel["A"]["oof"]
    cal, stats, bins = {}, [], []
    for m in PROBA_MODELS:
        o = oof[oof["model"] == m]
        c = CAL.calibrate_oof(o, y)
        cal[m] = c
        for source, version, p, rid in (("oof", "raw", o["score"].to_numpy(), o["row_id"]),
                                        ("oof_b2_5", "raw", c["score_raw"].to_numpy(), c["row_id"]),
                                        ("oof_b2_5", "cal", c["score_cal"].to_numpy(), c["row_id"])):
            yy = y.loc[rid].to_numpy()
            stats.append({"source": source, "model": m, "version": version, **CAL.calib_stats(yy, p)})
            bins.append(CAL.reliability_bins(yy, p).assign(source=source, model=m, version=version))
    return cal, stats, bins


def cal_arrays(cal_oof, frame):
    """{mô hình: p hiệu chỉnh} căn theo thứ tự dòng của frame (frame chỉ gồm các dòng có bản hiệu chỉnh)."""
    return {m: c.set_index("row_id")["score_cal"].reindex(frame["row_id"]).to_numpy() for m, c in cal_oof.items()}


def stage_policy_choice(frame, p_scores, cal_scores, pol_model, off, setup, checks):
    """Chọn cách xếp hạng (p / thiệt hại kỳ vọng / có hiệu chỉnh) của mô hình chính sách, CHỈ trên OOF, k = 3.

    Tiêu chí: lợi ích ròng/ngày lớn nhất ở lăng kính chính (b). Nếu p của mô hình lệch rõ thì xét thêm bản hiệu chỉnh
    (khi đó mọi ứng viên được so trên cùng các khối 2..5); ngược lại so trên toàn bộ OOF (khối 1..5).
    """
    use_cal = bool(off[pol_model])
    source = "oof_b2_5" if use_cal else "oof"
    rows = []
    for lens in C.LENSES:
        sc = C.policy_scores(frame, p_scores, lens, [pol_model], {pol_model: cal_scores[pol_model]} if use_cal else None)
        for ranking, name in (("p", pol_model), ("EV", f"{pol_model}_EV"), ("EVcal", f"{pol_model}_EVcal")):
            if name not in sc:
                continue
            ev = C.evaluate_policy(frame, C.policy_alerts(frame, sc[name], C.K_BASE, name), lens, setup)
            rows.append({"lens": lens, "model": pol_model, "ranking": ranking, "policy": name, "source": source,
                         "net_per_day": ev["net_per_day"], "recall_pooled": ev["recall_pooled"], "precision_pooled": ev["precision_pooled"]})
    t = pd.DataFrame(rows)
    prim = t[t["lens"] == C.PRIMARY_LENS].reset_index(drop=True)
    best = prim.loc[prim["net_per_day"].idxmax(), "ranking"]          # idxmax lấy ứng viên đầu tiên khi hoà
    t["chosen"] = t["ranking"] == best
    t["criterion"] = f"lợi ích ròng/ngày ở lăng kính {C.PRIMARY_LENS}, k = {C.K_BASE}, trên {source}"
    checks.ok("Chính sách thiệt hại kỳ vọng được chọn chỉ dựa trên OOF (không dùng test)",
              f"chọn '{best}' cho {pol_model} trên nguồn {source}; p lệch rõ = {use_cal}")
    return t, best, (f"{pol_model}" if best == "p" else f"{pol_model}_{best}")


def new_store():
    return {k: [] for k in ("curve", "summary", "bootstrap", "sens_scenarios", "sens_one_at_a_time", "net_grid", "break_even")}


def run_cost_reports(store, scenario, source, frame, p_scores, cal_scores, focus, setup):
    """Chạy cost_report cho mọi lăng kính và cộng dồn vào store (có cột scenario, lens, source)."""
    for lens in C.LENSES:
        sc = C.policy_scores(frame, p_scores, lens, PROBA_MODELS, cal_scores)
        f = [x for x in focus if x in sc]
        rep = C.cost_report(frame, sc, lens, setup, f, source)
        for key, t in rep.items():
            store[key].append(t.assign(scenario=scenario)[["scenario"] + [c for c in t.columns]])
    return store


# ----------------------------------------------------------------------------
# Giai đoạn 4: thí nghiệm rò rỉ
# ----------------------------------------------------------------------------
def stage_leakage(train, sel):
    log("Thí nghiệm rò rỉ: k-fold ngẫu nhiên 5x3 vs rolling-origin")
    rand = []
    for s in sel:
        for m in MODEL_NAMES:
            rand.append(E.random_cv_folds(m, s, sel[s]["best"][m], train))
    rand = pd.concat(rand, ignore_index=True)
    roll = pd.concat([sel[s]["folds"] for s in sel], ignore_index=True)
    fold_tab = pd.concat([rand.assign(scheme="random"), roll.assign(scheme="rolling")], ignore_index=True)
    save_table(fold_tab, "leakage_experiment_fold_scores.csv")
    rows = []
    for (s, m, scheme), g in fold_tab.groupby(["scenario", "model", "scheme"], sort=False):
        for metric in E.METRICS:
            rows.append({"scenario": s, "model": m, "scheme": scheme, "metric": metric, "mean": g[metric].mean(),
                         "std": g[metric].std(ddof=1), "n_folds": len(g)})
    summ = pd.DataFrame(rows)
    save_table(summ, "leakage_experiment_summary.csv")
    wide = summ.pivot_table(index=["scenario", "model", "metric"], columns="scheme", values="mean").reset_index()
    wide["gap_random_minus_rolling"] = wide["random"] - wide["rolling"]
    save_table(wide, "leakage_experiment_gap.csv")
    return summ


# ----------------------------------------------------------------------------
# Giai đoạn 5: ĐÁNH GIÁ TEST (gọi đúng một lần)
# ----------------------------------------------------------------------------
def stage_test(df, train, test, sel, checks):
    log("ĐÁNH GIÁ TEST (một lần): fit lại trên toàn bộ train rồi dự báo test")
    preds, estimators = [], {}
    for s in sel:
        for m in MODEL_NAMES:
            est, p = E.fit_final_and_predict_test(m, s, sel[s]["best"][m], train, test)
            check_pipeline_fit_on_train(est, train, test, df, s, checks, f"{m}, kịch bản {s}")
            preds.append(p); estimators[(s, m)] = est
    base = E.baseline_test(test)
    allp = pd.concat(preds + [base], ignore_index=True)
    save_table(allp, "predictions_test.csv")
    tm = E.metrics_from_predictions(allp, test)
    save_table(tm, "test_metrics.csv")
    return allp, estimators, tm


def stage_near_tie(train, test, sel, main_model, test_metrics):
    """Độ nhạy của kết quả test với việc phá hoà siêu tham số: hai cấu hình đứng đầu trên CV của mô hình chính
    (kịch bản A) được fit trên toàn train và đánh giá trên test. Chỉ để báo cáo, không đổi lựa chọn chính thức."""
    log("Kiểm tra cấu hình gần hoà của mô hình chính trên test")
    g = sel["A"]["grid"]
    g = g[g["model"] == main_model].sort_values("average_precision_mean", ascending=False, kind="stable").head(2)
    rows = []
    for rank, r in enumerate(g.itertuples(), start=1):
        _, p = E.fit_final_and_predict_test(main_model, "A", E.params_from_str(r.params), train, test)
        t = E.metrics_from_predictions(p, test).iloc[0]
        rows.append({"model": main_model, "cv_rank": rank, "params": r.params, "cv_pr_auc_mean": r.average_precision_mean,
                     "test_pr_auc": t["average_precision"], "test_roc_auc": t["roc_auc"]})
    t = pd.DataFrame(rows)
    official = test_metrics[(test_metrics["scenario"] == "A") & (test_metrics["model"] == main_model)]["average_precision"].iloc[0]
    assert abs(t["test_pr_auc"].iloc[0] - official) < 1e-9, "Cấu hình hạng 1 phải trùng kết quả test chính thức"
    return t


def collect(tabs: list) -> pd.DataFrame:
    return pd.concat(tabs, ignore_index=True)


def main():
    ensure_dirs()
    warnings.filterwarnings("ignore", category=FutureWarning)
    checks = Checks()
    raw = load_raw(DATA_FILE)
    df = build_dataset(DATA_FILE)            # có assert đặc trưng trễ
    train, test = split_train_test(df)
    cv = RollingOriginByDate(n_blocks=6)
    log(f"Dữ liệu: {df.shape}; train {len(train)} dòng / {train['date'].nunique()} ngày; test {len(test)} dòng / {test['date'].nunique()} ngày")

    stage_data_eda(raw, df, train, test, cv, checks)

    # ---- dò siêu tham số + OOF (chưa đụng tới test)
    sel, base_pred, base_folds = stage_selection(train, cv)
    folds, summ, oof = save_selection_tables(sel, base_pred, base_folds)
    save_fig(P.plot_cv_fold_scores(folds, base_folds, "A", "fig07_diem_theo_fold_A.png"), None)
    boundary = E.grid_boundary_check({s: sel[s]["best"] for s in sel})
    save_table(boundary, "grid_boundary_check.csv")
    log(f"Biên lưới: {int(boundary['at_edge'].sum())} tham số vẫn nằm ở biên lưới: "
        + "; ".join(f"{r.scenario}/{r.model}/{r.param.replace('clf__', '')}={r.best}" for r in boundary[boundary['at_edge']].itertuples()))

    save_table(stage_edge_probe(train, cv, sel, boundary), "grid_edge_probe.csv")

    # ---- mô hình chính; thiết lập chi phí (trung vị smv của TRAIN)
    main_model, pol_model, mc = stage_main_model(summ, sel)
    save_table(mc, "model_choice.csv")
    setup = C.CostSetup(smv_median=float(train["smv"].median()))
    save_table(setup.table(), "cost_setup.csv")
    log(f"Mô hình chính (PR-AUC CV cao nhất, kịch bản A): {main_model}; mô hình cho thiệt hại kỳ vọng: {pol_model}; "
        f"C_int lăng kính b = 240 / {setup.smv_median:.2f} = {setup.c_int('b'):.2f} sản phẩm/lần")

    # ---- dự báo OOF dạng bảng chi phí
    frames = {s: E.oof_frame(train, sel[s]["oof"]) for s in sel}
    p_scores = {s: E.align_scores(frames[s], sel[s]["oof"], s) for s in sel}

    # ---- hiệu chỉnh xác suất (OOF)
    log("Độ tin cậy của xác suất và hiệu chỉnh sigmoid không rò rỉ (OOF)")
    cal_oof, cal_stats, cal_bins = stage_calibration_oof(train, sel)
    off = {m: CAL.is_clearly_off(next(r for r in cal_stats if r["source"] == "oof" and r["model"] == m)) for m in PROBA_MODELS}
    checks.ok("Hiệu chỉnh sigmoid không rò rỉ: bộ hiệu chỉnh của khối j chỉ fit trên điểm OOF của các khối < j",
              f"khối 1 không có bản hiệu chỉnh; {len(cal_oof)} mô hình; test dùng bộ hiệu chỉnh fit trên toàn bộ OOF của train")

    # ---- chọn chính sách trên OOF (khối 2..5 khi có bản hiệu chỉnh)
    fA = frames["A"]
    m25 = (fA["fold"] >= 2).to_numpy()
    f25 = fA[m25].reset_index(drop=True)
    p25 = {m: v[m25] for m, v in p_scores["A"].items()}
    c25 = cal_arrays(cal_oof, f25)
    pc, ranking, chosen_policy = stage_policy_choice(f25 if off[pol_model] else fA, p25 if off[pol_model] else p_scores["A"], c25,
                                                    pol_model, off, setup, checks)
    save_table(pc, "policy_choice.csv")
    use_cal = chosen_policy.endswith("_EVcal")
    log(f"Chính sách được chọn trên OOF = {chosen_policy} (p lệch rõ: {off}; dùng p hiệu chỉnh: {use_cal})")
    focus = list(dict.fromkeys([main_model, pol_model, f"{pol_model}_EV", f"{pol_model}_EVcal", chosen_policy, "B1", "B2", "B0"]))

    # ---- chi phí trên OOF: hai lăng kính, k = 3
    log("Chi phí trên dự báo out-of-fold (hai lăng kính, k = 3)")
    store = new_store()
    run_cost_reports(store, "A", "oof", fA, p_scores["A"], None, focus, setup)
    run_cost_reports(store, "A", "oof_b2_5", f25, p25, c25, focus, setup)
    focus_b = list(dict.fromkeys([main_model, pol_model, f"{pol_model}_EV", "B1", "B2", "B0"]))
    run_cost_reports(store, "B", "oof", frames["B"], p_scores["B"], None, focus_b, setup)
    pf = []
    for lens in C.LENSES:
        sc = C.policy_scores(fA, p_scores["A"], lens, PROBA_MODELS)
        pf.append(C.per_fold_net(fA, sc, [x for x in focus if x in sc], lens, setup))
    pf_all = collect(pf)
    save_table(pf_all, "cost_per_fold_oof.csv")
    g = pf_all.groupby(["lens", "policy", "k"], sort=False)
    cols = ["alerts_per_day", "tp_per_day", "recall_pooled", "precision_pooled", "net_per_day"]
    save_table(g[cols].mean().add_suffix("_mean").join(g[cols].std(ddof=1).add_suffix("_std")).reset_index(), "cost_fold_summary_oof.csv")

    # ---- learning curve (chỉ dùng train)
    log("Learning curve theo thời gian")
    lc_models = list(dict.fromkeys([main_model, "Logistic"]))
    lc = LC.learning_curve(train, cv, sel["A"]["best"], "A", lc_models, val_blocks=(5, 4))
    save_table(lc, "learning_curve.csv")
    save_fig(P.plot_learning_curve(lc, lc_models, "fig21_learning_curve.png"), None)
    checks.ok("Learning curve: mọi dòng huấn luyện đứng trước khối validate cố định", "m = 5, 10, ... ngày liền trước khối 5 và khối 4")

    # ---- thí nghiệm rò rỉ
    leak = stage_leakage(train, sel)
    save_fig(P.plot_leakage(leak, "fig08_ro_ri_ngau_nhien_vs_thoi_gian.png"), None)

    # ---- TEST (một lần)
    test_preds, estimators, test_metrics = stage_test(df, train, test, sel, checks)
    names_A = feature_names(estimators[("A", "Logistic")].named_steps["prep"])
    assert not any(n.startswith(("wip_missing", "lag_missing")) for n in names_A) and names_A.count("is_first_obs") == 1
    checks.ok("Không có cờ thiếu trùng lặp (wip_missing/lag_missing); đúng một cờ is_first_obs", f"{len(names_A)} đặc trưng sau biến đổi (kịch bản A)")
    save_fig(P.plot_scenario_ab(summ, test_metrics, "fig09_kich_ban_A_vs_B.png"), None)
    save_table(stage_near_tie(train, test, sel, main_model, test_metrics), "near_tie_check.csv")

    # ---- hiệu chỉnh xác suất trên test
    y_tr = train.set_index("row_id")["miss"]
    tA = {s: E.align_scores(test, test_preds, s) for s in sel}
    cal_test = {}
    for m in PROBA_MODELS:
        o = sel["A"]["oof"][sel["A"]["oof"]["model"] == m]
        cal_test[m] = CAL.calibrate_test(o, y_tr, tA["A"][m])
        for version, p in (("raw", tA["A"][m]), ("cal", cal_test[m])):
            cal_stats.append({"source": "test", "model": m, "version": version, **CAL.calib_stats(test["miss"].to_numpy(), p)})
            cal_bins.append(CAL.reliability_bins(test["miss"].to_numpy(), p).assign(source="test", model=m, version=version))
    cs = pd.DataFrame(cal_stats)
    cs["clearly_off_on_oof"] = cs["model"].map(off)
    cs["used_in_d16"] = (cs["model"] == pol_model) & use_cal & (cs["version"] == "cal")
    save_table(cs, "calibration_summary.csv")
    save_table(collect(cal_bins), "calibration_bins.csv")
    save_fig(P.plot_reliability(collect(cal_bins), cs, PROBA_MODELS, "fig20_do_tin_cay_xac_suat.png"), None)

    # ---- bảng chi phí trên test (cùng chính sách đã chọn trên OOF)
    log("Bảng chi phí trên test, độ nhạy, hoà vốn")
    run_cost_reports(store, "A", "test", test, tA["A"], cal_test, focus, setup)
    run_cost_reports(store, "B", "test", test, tA["B"], None, focus_b, setup)
    reports = {}
    for key, lst in store.items():
        reports[key] = collect(lst)
        save_table(reports[key], f"cost_{key}.csv")

    cv_ = reports["curve"]
    cA = cv_[cv_["scenario"] == "A"]
    lines = list(dict.fromkeys([m for m in MODEL_NAMES] + [f"{pol_model}_EV", "B1", "B2"]))
    save_fig(P.plot_net_benefit_vs_k(cA, "oof", C.K_BASE, "Độ nhạy theo k - dự báo out-of-fold (kịch bản A, giả định cơ sở)", "fig10_loi_ich_rong_theo_k_oof.png", lines), None)
    save_fig(P.plot_net_benefit_vs_k(cA, "test", C.K_BASE, "Độ nhạy theo k - tập test (kịch bản A, giả định cơ sở)", "fig11_loi_ich_rong_theo_k_test.png", lines), None)
    save_fig(P.plot_recall_precision_k(cA[cA["source"] == "oof"], "Recall@k và Precision@k - dự báo out-of-fold (kịch bản A)", "fig12_recall_precision_theo_k_oof.png"), None)
    sA = reports["sens_scenarios"]; sA = sA[sA["scenario"] == "A"]
    bar_pols = [pol_model, f"{pol_model}_EV", f"{pol_model}_EVcal", "B1", "B2"]
    save_fig(P.plot_cost_scenarios(sA, bar_pols, "fig13_do_nhay_3_kich_ban.png"), None)
    ngA = reports["net_grid"]; ngA = ngA[ngA["scenario"] == "A"]
    save_fig(P.plot_netgrid_heatmap(ngA, chosen_policy, "B1", "fig14_do_nhay_r_cint.png", "oof_b2_5"), None)
    sm = reports["summary"]; sm = sm[sm["scenario"] == "A"]
    save_fig(P.plot_policy_bars(sm, bar_pols, "fig19_chinh_sach_thiet_hai_ky_vong_k3.png",
                                "Lợi ích ròng/ngày tại k = 3: top-k theo p, thiệt hại kỳ vọng, B1, B2 (KTC 95% bootstrap theo ngày)"), None)

    # ---- diễn giải (kịch bản A, mô hình chính)
    log("Diễn giải mô hình")
    est_A = {m: estimators[("A", m)] for m in MODEL_NAMES}
    pi = I.permutation_importance_table(est_A, test, "A", n_repeats=30)
    save_table(pi, "perm_importance_test.csv")
    save_fig(P.plot_perm_importance(pi, main_model, "fig15_permutation_importance.png"), None)
    top = I.top_features(pi, main_model, 4)
    X_tr = train[SCENARIOS["A"].columns]
    pdp = I.pdp_table(est_A[main_model], X_tr, top)
    save_table(pdp, "pdp_values.csv")
    ylabel = "Điểm dự báo trung bình (xác suất trượt)" if main_model in PROBA_MODELS else "Điểm dự báo trung bình (decision function)"
    save_fig(P.plot_pdp(pdp, top, P.SHORT[main_model], ylabel, "fig16_pdp_ice.png"), None)
    prep, tree, agree = I.fit_interpretable_tree(train, sel["A"]["best"]["DecisionTree"], "A", max_depth=3)
    save_table(I.tree_leaf_table(prep, tree, train, "A"), "tree_depth3_rules.csv")
    save_fig(I.plot_tree_figure(prep, tree, "fig17_cay_quyet_dinh_do_sau_3.png"), None)
    checks.ok("Cây vẽ (không chuẩn hoá) trùng dự báo với cây cùng tham số trên dữ liệu chuẩn hoá", f"tỷ lệ dự báo trùng trên train = {agree:.4f}")
    coef = I.logistic_coefficients(est_A["Logistic"])
    save_table(coef, "logistic_coefficients.csv")
    save_fig(P.plot_coefficients(coef, "fig18_he_so_logistic.png"), None)

    # ---- độ nhạy khi loại định mức thấp
    stage_d13(train, test, cv, setup, folds, test_metrics)

    # ---- phiếu cảnh báo mẫu cho các ngày test (huấn luyện trên mọi ngày < D)
    log("phiếu cảnh báo mẫu")
    for d in SAMPLE_SHEET_DATES:
        res = AS.make_sheet(df, d, C.K_BASE, C.PRIMARY_LENS, pol_model, sel["A"]["best"][pol_model], ranking)
        AS.write_sheet(res)
    checks.ok("Phiếu cảnh báo: chỉ huấn luyện trên các ngày < D", f"đã sinh {len(SAMPLE_SHEET_DATES)} phiếu mẫu: {', '.join(SAMPLE_SHEET_DATES)}")

    save_table(checks.table(), "leakage_checks.csv")
    write_results_summary()
    log("Hoàn tất")


def stage_d13(train, test, cv, setup, folds, test_metrics):
    """Độ nhạy: chạy lại kịch bản A khi loại các dòng targeted_productivity < 0,5 (cả train và test); k = 3, top-k theo p."""
    log("Độ nhạy: loại các dòng targeted_productivity < 0,5")
    keep_tr = train["targeted_productivity"] >= 0.5
    keep_te = test["targeted_productivity"] >= 0.5
    tr, te = train[keep_tr].reset_index(drop=True), test[keep_te].reset_index(drop=True)
    sel = E.select_models(tr, "A", cv)
    base_pred, base_folds = E.baseline_oof(tr, cv)
    fr = E.oof_frame(tr, sel["oof"])
    sc_oof = E.align_scores(fr, sel["oof"], "A")
    preds_te = []
    for m in MODEL_NAMES:
        _, p = E.fit_final_and_predict_test(m, "A", sel["best"][m], tr, te)
        preds_te.append(p)
    pt = collect(preds_te)
    sc_te = E.align_scores(te, pt, "A")
    tm = E.metrics_from_predictions(collect([pt, E.baseline_test(te)]), te).set_index("model")
    fsum = E.summarize_folds(collect([sel["folds"], base_folds])).set_index("model")
    rows = []
    for m in MODEL_NAMES + BASELINES:
        k = 0 if m == "B0" else C.K_BASE
        net = {}
        for lens in C.LENSES:
            net[f"oof_net_{lens}"] = C.evaluate_policy(fr, C.policy_alerts(fr, sc_oof[m], k, m), lens, setup)["net_per_day"]
            net[f"test_net_{lens}"] = C.evaluate_policy(te, C.policy_alerts(te, sc_te[m], k, m), lens, setup)["net_per_day"]
        rows.append({"model": m, "k": k, "n_train": len(tr), "n_test": len(te),
                     "cv_pr_auc_mean": fsum.loc[m, "average_precision_mean"], "cv_pr_auc_std": fsum.loc[m, "average_precision_std"],
                     "cv_roc_auc_mean": fsum.loc[m, "roc_auc_mean"], "cv_roc_auc_std": fsum.loc[m, "roc_auc_std"],
                     "test_pr_auc": tm.loc[m, "average_precision"], "test_roc_auc": tm.loc[m, "roc_auc"], **net})
    d13 = pd.DataFrame(rows)
    main_cv = E.summarize_folds(folds[folds["scenario"].isin(["A", "-"])]).set_index("model")
    main_te = test_metrics[test_metrics["scenario"].isin(["A", "-"])].set_index("model")
    d13["main_cv_pr_auc_mean"] = d13["model"].map(main_cv["average_precision_mean"])
    d13["main_test_pr_auc"] = d13["model"].map(main_te["average_precision"])
    d13["delta_cv_pr_auc"] = d13["cv_pr_auc_mean"] - d13["main_cv_pr_auc_mean"]
    d13["delta_test_pr_auc"] = d13["test_pr_auc"] - d13["main_test_pr_auc"]
    save_table(d13, "d13_sensitivity_low_target.csv")


# ----------------------------------------------------------------------------
# RESULTS_SUMMARY.md: sinh tự động từ outputs/tables để mọi con số khớp bảng
# ----------------------------------------------------------------------------
def write_results_summary():
    from summary import build_summary
    (ROOT / "RESULTS_SUMMARY.md").write_text(build_summary(), encoding="utf-8")


if __name__ == "__main__":
    main()
