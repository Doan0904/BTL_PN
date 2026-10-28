"""Sinh RESULTS_SUMMARY.md từ outputs/tables/*.csv (mọi con số lấy từ bảng, không gõ tay)."""
import pandas as pd

from config import ROOT, TABLES_DIR
from pipeline import MODEL_LABELS

ORDER = ["B0", "B1", "B2", "DecisionTree", "SVM_RBF", "MLP", "Logistic", "RandomForest"]
LABEL = {**MODEL_LABELS, "B0": "B0 (lớp đa số)", "B1": "B1 (hôm qua trượt)", "B2": "B2 (mọi tổ finishing)"}


def pol_label(name: str) -> str:
    """'RandomForest_EV' -> 'Random Forest - thiệt hại kỳ vọng'."""
    for suf, lab in (("_EVcal", "thiệt hại kỳ vọng (p hiệu chỉnh)"), ("_EV", "thiệt hại kỳ vọng")):
        if name.endswith(suf):
            return f"{LABEL[name[:-len(suf)]]} - {lab}"
    return LABEL[name] + ("" if name.startswith("B") else " - top-k theo p")


def _t(name):
    return pd.read_csv(TABLES_DIR / name)


def _md(rows: list, header: list) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def _f(x, d=3):
    return "-" if pd.isna(x) else f"{x:.{d}f}"


def build_summary() -> str:
    cv, tm = _t("cv_summary.csv"), _t("test_metrics.csv")
    split, folds = _t("split_summary.csv"), _t("rolling_origin_folds.csv")
    L = []
    add = L.append
    add("# Tóm tắt kết quả (sinh tự động bởi `src/run_all.py` từ `outputs/tables/*.csv`)\n")
    add("Mọi con số dưới đây được đọc trực tiếp từ các bảng CSV; chạy lại `python src/run_all.py` sẽ ghi đè file này. "
        "Test chỉ được dùng để đánh giá; mọi lựa chọn (mô hình chính, siêu tham số, chính sách thiệt hại kỳ vọng, hiệu chỉnh) "
        "được thực hiện trên train/OOF.\n")

    add("## 1. Dữ liệu và cách chia\n")
    add(_md([[r.tap, r.tu_ngay, r.den_ngay, r.n_ngay, r.n_dong, _f(r.ty_le_miss)] for r in split.itertuples()],
            ["Tập", "Từ ngày", "Đến ngày", "Số ngày", "Số dòng", "Tỷ lệ miss"]))
    add("\nRolling-origin theo ngày trên train (5 fold):\n")
    add(_md([[r.fold, f"{r.train_from} đến {r.train_to}", f"{r.val_from} đến {r.val_to}", r.n_train, r.n_val] for r in folds.itertuples()],
            ["Fold", "Huấn luyện", "Validate", "Dòng train", "Dòng validate"]))

    for sc, title in (("A", "2. Kịch bản A (chính): CV rolling-origin (trung bình ± độ lệch chuẩn qua 5 fold) và test"),
                      ("B", "3. Kịch bản B (độ nhạy): CV rolling-origin và test")):
        add(f"\n## {title}\n")
        rows = []
        for m in ORDER:
            sel = cv[((cv.scenario == sc) | (cv.scenario == "-")) & (cv.model == m)]
            tst = tm[((tm.scenario == sc) | (tm.scenario == "-")) & (tm.model == m)]
            if sel.empty:
                continue
            c, t = sel.iloc[0], tst.iloc[0]
            rows.append([LABEL[m], _pm(c.average_precision_mean, c.average_precision_std), _pm(c.roc_auc_mean, c.roc_auc_std),
                         _pm(c.f2_mean, c.f2_std), _pm(c.recall_mean, c.recall_std), _pm(c.precision_mean, c.precision_std),
                         _f(t.average_precision), _f(t.roc_auc), _f(t.f2), _f(t.recall), _f(t.precision)])
        add(_md(rows, ["Mô hình", "PR-AUC (CV)", "ROC-AUC (CV)", "F2 (CV)", "Recall (CV)", "Precision (CV)",
                       "PR-AUC (test)", "ROC-AUC (test)", "F2 (test)", "Recall (test)", "Precision (test)"]))
        add("\nF2/Recall/Precision tính tại ngưỡng mặc định của `predict()` (MLP không có class_weight nên recall thấp).")

    add("\n## 4. Thí nghiệm rò rỉ: k-fold ngẫu nhiên 5x3 so với rolling-origin (cùng mô hình, cùng siêu tham số)\n")
    gap = _t("leakage_experiment_gap.csv")
    sm = _t("leakage_experiment_summary.csv")
    for sc in ("A", "B"):
        rows = []
        for m in ORDER[3:]:
            r = {}
            for metric in ("average_precision", "roc_auc"):
                g = gap[(gap.scenario == sc) & (gap.model == m) & (gap.metric == metric)].iloc[0]
                sd = sm[(sm.scenario == sc) & (sm.model == m) & (sm.metric == metric)].set_index("scheme")["std"]
                r[metric] = (f"{g['random']:.3f} ± {sd['random']:.3f}", f"{g['rolling']:.3f} ± {sd['rolling']:.3f}", f"{g['gap_random_minus_rolling']:+.3f}")
            rows.append([LABEL[m], *r["average_precision"], *r["roc_auc"]])
        add(f"\nKịch bản {sc}:\n")
        add(_md(rows, ["Mô hình", "PR-AUC ngẫu nhiên", "PR-AUC rolling", "Chênh", "ROC-AUC ngẫu nhiên", "ROC-AUC rolling", "Chênh"]))

    add("\n## 5. Kịch bản A so với B (PR-AUC)\n")
    rows = []
    for m in ORDER[3:]:
        a = cv[(cv.scenario == "A") & (cv.model == m)].iloc[0]; b = cv[(cv.scenario == "B") & (cv.model == m)].iloc[0]
        ta = tm[(tm.scenario == "A") & (tm.model == m)].iloc[0]; tb = tm[(tm.scenario == "B") & (tm.model == m)].iloc[0]
        rows.append([LABEL[m], _f(a.average_precision_mean), _f(b.average_precision_mean), f"{b.average_precision_mean - a.average_precision_mean:+.3f}",
                     _f(ta.average_precision), _f(tb.average_precision), f"{tb.average_precision - ta.average_precision:+.3f}"])
    add(_md(rows, ["Mô hình", "CV A", "CV B", "B - A", "Test A", "Test B", "B - A"]))

    add("\n## 6. Mô hình chính và siêu tham số\n")
    mc = _t("model_choice.csv")
    add("Mô hình chính = PR-AUC CV (rolling-origin, kịch bản A) trung bình cao nhất; k cố định = 3.\n")
    add(_md([[r.rank_cv_pr_auc, LABEL[r.model], _pm(r.cv_pr_auc_mean, r.cv_pr_auc_std), "x" if r.is_main else "", "x" if r.is_policy_model else "", r.params]
             for r in mc.itertuples()], ["Hạng", "Mô hình", "PR-AUC CV", "Mô hình chính", "Mô hình cho thiệt hại kỳ vọng", "Siêu tham số chọn trên train"]))
    main, polm = mc[mc.is_main].model.iloc[0], mc[mc.is_policy_model].model.iloc[0]
    bd = _t("grid_boundary_check.csv")
    hit = bd[bd.at_edge]
    add("\nSiêu tham số nằm ở biên lưới:\n")
    add(_md([[r.scenario, LABEL[r.model], r.param.replace("clf__", ""), r.grid, r.best, "dưới" if r.at_lower_edge else "trên"] for r in hit.itertuples()],
            ["Kịch bản", "Mô hình", "Tham số", "Lưới", "Giá trị chọn", "Biên"]) if len(hit) else "Không có tham số nào ở biên.")
    pr = _t("grid_edge_probe.csv")
    if len(pr):
        add("\nThăm dò một nấc nữa vượt biên (chỉ để ghi chú, không đổi lựa chọn chính thức):\n")
        add(_md([[r.scenario, LABEL[r.model], r.param.replace("clf__", ""), r.value_at_edge, _f(r.cv_pr_auc_at_edge), r.value_beyond, _f(r.cv_pr_auc_beyond), f"{r.delta:+.3f}"] for r in pr.itertuples()],
                ["Kịch bản", "Mô hình", "Tham số", "Giá trị ở biên", "PR-AUC CV", "Giá trị vượt biên", "PR-AUC CV", "Chênh"]))
    nt = _t("near_tie_check.csv")
    add(f"\nĐộ nhạy của kết quả test với việc phá hoà siêu tham số ({LABEL[main]}, hai cấu hình đứng đầu trên CV; chỉ để báo cáo):\n")
    add(_md([[r.cv_rank, r.params, f"{r.cv_pr_auc_mean:.4f}", _f(r.test_pr_auc), _f(r.test_roc_auc)] for r in nt.itertuples()],
            ["Hạng CV", "Siêu tham số", "PR-AUC CV", "PR-AUC test", "ROC-AUC test"]))

    add("\n## 7. Độ tin cậy của xác suất và hiệu chỉnh sigmoid (kịch bản A)\n")
    cal = _t("calibration_summary.csv")
    add("Brier càng thấp càng tốt; Brier-hằng là Brier của dự báo \"luôn bằng tỷ lệ trượt\" của chính tập đó. Hiệu chỉnh sigmoid không rò rỉ: "
        "bộ hiệu chỉnh của khối validate j chỉ fit trên điểm OOF của các khối < j (nên chỉ có từ khối 2); test dùng bộ fit trên toàn bộ OOF. "
        "\"Lệch rõ\" = trung bình p lệch hơn 0,05 so với tỷ lệ trượt thực hoặc Brier tệ hơn dự báo hằng (xét trên OOF).\n")
    rows = []
    for src, name in (("oof", "OOF khối 1-5"), ("oof_b2_5", "OOF khối 2-5"), ("test", "Test")):
        for m in ("DecisionTree", "MLP", "Logistic", "RandomForest"):
            for ver in ("raw", "cal"):
                g = cal[(cal.source == src) & (cal.model == m) & (cal.version == ver)]
                if g.empty:
                    continue
                r = g.iloc[0]
                rows.append([name, LABEL[m], "chưa hiệu chỉnh" if ver == "raw" else "sigmoid", _f(r.prevalence), _f(r.mean_pred), _f(r.brier, 4), _f(r.brier_const, 4),
                             _f(r.ece), _f(r.calib_slope, 2), "có" if (ver == "raw" and r.clearly_off_on_oof) else ""])
    add(_md(rows, ["Tập", "Mô hình", "Phiên bản", "Tỷ lệ trượt thực", "TB p", "Brier", "Brier-hằng", "ECE", "Độ dốc hiệu chỉnh", "Lệch rõ"]))
    add("\nĐồ thị: `outputs/figures/fig20_do_tin_cay_xac_suat.png`; các nhóm: `calibration_bins.csv`.")

    add("\n## 8. Chi phí, chính sách k = 3 và lợi ích ròng (hai lăng kính)\n")
    su = _t("cost_setup.csv")
    add(_md([[r.lens_label, f"{r.r_base:g}", f"{r.c_int_base:.2f} {r.unit}/lần", f"{r.c_int_min:.2f} đến {r.c_int_max:.2f}"] for r in su.itertuples()],
            ["Lăng kính", "r cơ sở", "C_int cơ sở", "Dải độ nhạy C_int"]))
    add(f"\nLăng kính (b) là lăng kính chính cho khuyến nghị; C_int(b) = 240 / trung vị smv của train = 240 / {su.smv_median_train.iloc[0]:.2f}. "
        "Lợi ích ròng/ngày = r x thiệt hại của các tổ trượt được cảnh báo đúng - C_int x số cảnh báo. Quy đổi BDT chỉ có ở lăng kính (a).\n")
    pc = _t("policy_choice.csv")
    add(f"**Chọn chính sách trên OOF** cho {LABEL[polm]} (k = 3, tiêu chí: lợi ích ròng/ngày ở lăng kính b):\n")
    add(_md([[r.lens, r.ranking, r.source, _f(r.net_per_day, 1), _f(r.recall_pooled), _f(r.precision_pooled), "x" if r.chosen else ""] for r in pc.itertuples()],
            ["Lăng kính", "Xếp hạng", "Nguồn", "Lợi ích ròng/ngày", "Recall@3", "Precision@3", "Được chọn (theo lăng kính b)"]))
    ch = pc[pc.chosen].ranking.iloc[0]
    chosen = polm if ch == "p" else f"{polm}_{ch}"
    add(f"\nChính sách được chọn: **{pol_label(chosen)}**. (Lưu ý: ở lăng kính (a) thứ tự có thể khác; xem bảng.)\n")

    cs = _t("cost_summary.csv")
    cs = cs[cs.scenario == "A"]
    show = list(dict.fromkeys([main, polm, f"{polm}_EV", f"{polm}_EVcal", "B1", "B2", "B0"]))
    for lens, ln, unit in (("b", "Lăng kính (b) - sản phẩm thiếu (CHÍNH)", "sản phẩm"), ("a", "Lăng kính (a) - phút chuẩn", "phút chuẩn")):
        add(f"\n### {ln}\n")
        rows = []
        for src, name in (("oof_b2_5", "OOF khối 2-5"), ("test", "Test")):
            for pol in show:
                g = cs[(cs.lens == lens) & (cs.source == src) & (cs.policy == pol)]
                if g.empty:
                    continue
                r = g.iloc[0]
                bdt = _f(r.net_bdt_per_month, 0) if lens == "a" else "-"
                rows.append([name, pol_label(pol), _f(r.alerts_per_day, 2), _f(r.tp_per_day, 2), _f(r.recall_pooled), _f(r.precision_pooled),
                             _f(r.net_per_day, 1), f"[{r.net_per_day_ci_low:.1f}; {r.net_per_day_ci_high:.1f}]", bdt])
        add(_md(rows, ["Tập", "Chính sách", "Cảnh báo/ngày", "Tổ trượt bắt được/ngày", "Recall@3", "Precision@3", f"Lợi ích ròng/ngày ({unit})", "KTC 95% (bootstrap theo ngày)", "Lợi ích ròng/tháng (BDT, 26 ngày)"]))
        bs = _t("cost_bootstrap.csv")
        bs = bs[(bs.scenario == "A") & (bs.lens == lens) & (bs.policy.isin([main, polm, f"{polm}_EV", f"{polm}_EVcal"]))]
        add(f"\nHiệu lợi ích ròng/ngày ({unit}) so với B1 và B2 tại k = 3 (bootstrap theo ngày, 2000 lần):\n")
        add(_md([[("OOF khối 2-5" if r.source == "oof_b2_5" else "Test"), pol_label(r.policy), r.ref, f"{r.diff_mean:+.1f}", f"[{r.ci_low:.1f}; {r.ci_high:.1f}]", _f(r.p_diff_gt0, 2)]
                 for r in bs[bs.source.isin(["oof_b2_5", "test"])].itertuples()], ["Tập", "Chính sách", "So với", "Chênh trung bình", "KTC 95%", "P(chênh > 0)"]))

    fs = _t("cost_fold_summary_oof.csv")
    add("\nLợi ích ròng/ngày theo fold rolling-origin tại k = 3 (trung bình ± độ lệch chuẩn qua 5 fold, OOF khối 1-5):\n")
    add(_md([[("b" if r.lens == "b" else "a"), pol_label(r.policy), _pm(r.net_per_day_mean, r.net_per_day_std, 1)] for r in fs.sort_values(["lens", "policy"], ascending=[False, True]).itertuples()],
            ["Lăng kính", "Chính sách", "Lợi ích ròng/ngày"]))

    sn = _t("cost_sens_scenarios.csv")
    sn = sn[(sn.scenario == "A") & (sn.policy.isin([chosen, f"{polm}_EV", "B1", "B2"]))]
    add("\nTác động dưới 3 kịch bản giả định (thận trọng r=0,10 và C_int x2; cơ sở r=0,30 và x1; lạc quan r=0,50 và x0,5; c_min 0,64/1,0/2,0 chỉ ở lăng kính a), k = 3:\n")
    rows = []
    for lens in ("b", "a"):
        for src, name in (("oof_b2_5", "OOF khối 2-5"), ("test", "Test")):
            for pol in dict.fromkeys([chosen, f"{polm}_EV", "B1", "B2"]):
                for cs_ in ("Thận trọng", "Cơ sở", "Lạc quan"):
                    g = sn[(sn.lens == lens) & (sn.source == src) & (sn.policy == pol) & (sn.cost_scenario == cs_)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    rows.append([lens, name, pol_label(pol), cs_, _f(r.c_int, 1), _f(r.alerts_per_day, 2), _f(r.tp_per_day, 2), _f(r.saved_per_month, 0), _f(r.net_per_month, 0),
                                 _f(r.net_bdt_per_month, 0) if lens == "a" else "-"])
    add(_md(rows, ["Lăng kính", "Tập", "Chính sách", "Kịch bản", "C_int", "Cảnh báo/ngày", "Tổ trượt bắt được/ngày", "Thiệt hại cứu được/tháng", "Lợi ích ròng/tháng", "Lợi ích ròng/tháng (BDT)"]))

    be = _t("cost_break_even.csv")
    be = be[(be.scenario == "A") & (be.policy == chosen)]
    add(f"\nNgưỡng hoà vốn của {pol_label(chosen)} (r* tại C_int cơ sở; C_int* tại r = 0,30; NaN = không có giao điểm dương):\n")
    add(_md([[r.lens, ("OOF khối 2-5" if r.source == "oof_b2_5" else "OOF khối 1-5" if r.source == "oof" else "Test"), r.ref, _f(r.delta_S_per_day, 1), _f(r.delta_alerts_per_day, 1), _f(r.r_star), _f(r.c_int_star, 1)]
             for r in be.itertuples() if r.source in ("oof_b2_5", "test")], ["Lăng kính", "Tập", "So với", "Chênh thiệt hại bắt được/ngày", "Chênh cảnh báo/ngày", "r*", "C_int*"]))
    add("\nĐường cong theo k (độ nhạy, không dùng để chọn k): `cost_curve.csv`, hình fig10/fig11.")

    add("\n## 9. Learning curve theo thời gian (kịch bản A, siêu tham số cố định)\n")
    lc = _t("learning_curve.csv")
    for vb in sorted(lc.val_block.unique(), reverse=True):
        g = lc[lc.val_block == vb]
        models = [m for m in g.model.unique() if m != "B1"]
        add(f"\nKhối validate {vb} ({int(g.n_val.iloc[0])} dòng, tỷ lệ trượt {g.prevalence_val.iloc[0]:.3f}); B1 = {g[g.model == 'B1'].pr_auc.iloc[0]:.3f} (không học):\n")
        w = g[g.model != "B1"].pivot(index="m_days", columns="model", values="pr_auc")
        add(_md([[int(m)] + [_f(w.loc[m, c]) for c in models] for m in w.index], ["m (ngày huấn luyện)"] + [LABEL[c] for c in models]))

    add("\n## 10. Diễn giải (kịch bản A, mô hình chính)\n")
    pi = _t("perm_importance_test.csv")
    add(f"Permutation importance trên test (giảm PR-AUC, 30 lần xáo trộn), top 6 cho {LABEL[main]}:\n")
    top = pi[pi.model == main].sort_values("mean", ascending=False).head(6)
    add(_md([[r.feature, _pm(r["mean"], r["std"], 4)] for _, r in top.iterrows()], ["Đặc trưng", "Giảm PR-AUC (TB ± std)"]))
    add("\nHạng đặc trưng theo permutation importance của từng mô hình (1 = quan trọng nhất):\n")
    rk = pi.assign(rank=pi.groupby("model")["mean"].rank(ascending=False, method="first").astype(int)).pivot(index="feature", columns="model", values="rank")
    rk = rk.assign(tb=rk.mean(axis=1)).sort_values("tb")
    add(_md([[f] + [int(rk.loc[f, m]) for m in MODEL_LABELS if m in rk.columns] for f in rk.index], ["Đặc trưng"] + [LABEL[m] for m in MODEL_LABELS if m in rk.columns]))
    co = _t("logistic_coefficients.csv").head(8)
    add("\nHệ số logistic lớn nhất (đặc trưng số đã chuẩn hoá; one-hot và is_first_obs theo thang 0/1):\n")
    add(_md([[r.feature, _f(r.coef)] for r in co.itertuples()], ["Đặc trưng", "Hệ số"]))
    tr = _t("tree_depth3_rules.csv")
    add("\nCác luật của cây độ sâu 3 (sắp theo xác suất trượt giảm dần):\n")
    add(_md([[r.rule, r.n_train, _f(r.miss_rate_train), _f(r.p_miss_weighted)] for r in tr.itertuples()],
            ["Luật", "Số dòng train", "Tỷ lệ trượt thực", "Xác suất (trọng số cân bằng)"]))

    add("\n## 11. Độ nhạy: loại các dòng targeted_productivity < 0,5 (kịch bản A, k = 3, top-k theo p)\n")
    d = _t("d13_sensitivity_low_target.csv")
    add(_md([[LABEL[r.model], _pm(r.cv_pr_auc_mean, r.cv_pr_auc_std), f"{r.delta_cv_pr_auc:+.3f}", _f(r.test_pr_auc), f"{r.delta_test_pr_auc:+.3f}",
              _f(r.oof_net_b, 1), _f(r.test_net_b, 1), _f(r.oof_net_a, 0), _f(r.test_net_a, 0)]
             for r in d.set_index("model").loc[ORDER].reset_index().itertuples()],
            ["Mô hình", "PR-AUC CV", "Chênh so với bản chính", "PR-AUC test", "Chênh so với bản chính", "Ròng OOF/ngày (b)", "Ròng test/ngày (b)", "Ròng OOF/ngày (a)", "Ròng test/ngày (a)"]))
    add(f"\nSố dòng sau khi loại: train {int(d.n_train.iloc[0])}, test {int(d.n_test.iloc[0])}.")

    add("\n## 12. Phiếu cảnh báo mẫu\n")
    sheets = sorted((ROOT / "outputs" / "alert_sheets").glob("*.md"))
    if sheets:
        ex = [x for x in sheets if "2015-03-05" in x.name] or sheets
        add(f"Tạo bằng `python src/alert_sheet.py --date 2015-03-05 --k 3 --lens b`. Các phiếu mẫu: " + ", ".join(f"`outputs/alert_sheets/{x.name}`" for x in sheets) + ".\n")
        add(ex[0].read_text(encoding="utf-8").replace("# ", "#### ", 1))

    add("\n## 13. Các kiểm tra chống rò rỉ đã chạy\n")
    lc_ = _t("leakage_checks.csv")
    add(_md([[r.kiem_tra, r.ket_qua] for r in lc_.itertuples()], ["Kiểm tra", "Kết quả"]))
    return "\n".join(L) + "\n"
