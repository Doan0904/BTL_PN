"""Phiếu cảnh báo đầu ca.

Cách dùng (sau khi đã chạy `python src/run_all.py` để có siêu tham số đã chọn trên train):
    python src/alert_sheet.py --date 2015-03-05 --k 3 [--lens b]

Huấn luyện mô hình chính (siêu tham số chọn trên train, kịch bản A) trên MỌI ngày < D, chấm điểm các tổ-bộ phận của
ngày D bằng đặc trưng biết đầu ca, rồi in phiếu: k tổ đáng hỗ trợ nhất, xác suất trượt, thiệt hại kỳ vọng và 2-3 lý do chính.
Nếu ngày D có trong dữ liệu, cột kết quả thực tế được in kèm CHỈ ĐỂ KIỂM TRA (không dùng khi chấm điểm).
Ghi thêm file md và csv vào outputs/alert_sheets/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import calibration as CAL
import cost as C
import evaluate as E
from config import DATA_FILE, ROOT, TABLES_DIR
from features import build_dataset
from pipeline import MODEL_LABELS, SCENARIOS, feature_names, raw_feature_of, rank_score
from splits import RollingOriginByDate

SHEETS_DIR = ROOT / "outputs" / "alert_sheets"
MIN_HISTORY_DAYS = 14
PROBA_MODELS = ["DecisionTree", "MLP", "Logistic", "RandomForest"]
FEATURE_VI = {
    "targeted_productivity": "định mức năng suất", "smv": "SMV (thời gian chuẩn)", "no_of_workers": "số công nhân",
    "week_of_month": "tuần trong tháng", "wip": "WIP tồn đầu ca", "lag1_miss": "hôm trước trượt (1/0)",
    "lag1_gap": "chênh lệch năng suất hôm trước", "roll5_gap_mean": "chênh lệch TB 5 ngày trước",
    "roll5_miss_rate": "tỷ lệ trượt 5 ngày trước", "is_first_obs": "chưa có lịch sử của cặp",
    "department": "bộ phận", "day": "thứ", "team": "tổ",
}
RANKING_LABEL = {"p": "top-k theo điểm rủi ro của mô hình (p chưa hiệu chỉnh)","EV": "thiệt hại kỳ vọng (p x quy mô mục tiêu)",
                 "EVcal": "thiệt hại kỳ vọng (p hiệu chỉnh sigmoid x quy mô mục tiêu)"}


# ----------------------------------------------------------------------------
# Lý do chính
# ----------------------------------------------------------------------------
def _fmt(v, feat):
    if isinstance(v, str):
        return v
    if feat in ("lag1_miss", "is_first_obs", "no_of_workers", "week_of_month"):
        return f"{v:.0f}"
    return "0" if abs(v) < 0.005 else f"{v:.3g}"


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def contributions_logistic(est, rows: pd.DataFrame, hist: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Đóng góp vào logit của từng đặc trưng thô: hệ số x (giá trị đã biến đổi - trung bình train).

    Với đặc trưng số đã chuẩn hoá đây chính là hệ số x giá trị chuẩn hoá; với one-hot/cờ là hệ số x (giá trị - tần suất).
    Các cột one-hot của cùng một biến phân loại được cộng lại.
    """
    prep, clf = est.named_steps["prep"], est.named_steps["clf"]
    names = feature_names(prep)
    Xt, mu = _dense(prep.transform(rows[cols])), _dense(prep.transform(hist[cols])).mean(axis=0)
    contrib = (Xt - mu) * clf.coef_[0]
    raw = [raw_feature_of(n) for n in names]
    out = pd.DataFrame(contrib, columns=raw).T.groupby(level=0).sum().T
    nan_wip = rows["wip"].isna().to_numpy()      # finishing không có wip: phần đóng góp của wip thực chất là hiệu ứng bộ phận
    out.loc[nan_wip, "department"] += out.loc[nan_wip, "wip"]
    out.loc[nan_wip, "wip"] = 0.0
    return out


def contributions_vs_median(est, rows: pd.DataFrame, hist: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Hiệu ứng so với giá trị tham chiếu: p(x) - p(x với đặc trưng j thay bằng trung vị train; biến phân loại: giá trị phổ biến).

    Cờ is_first_obs tham chiếu = 0. Đặc trưng thiếu (wip của finishing) được bỏ qua (hiệu ứng 0).
    """
    p0 = est.predict_proba(rows[cols])[:, 1]
    out = {}
    for c in cols:
        if pd.api.types.is_numeric_dtype(hist[c]) and c != "team":
            ref = 0 if c == "is_first_obs" else hist[c].median()
        else:
            ref = hist[c].mode().iloc[0]
        alt = rows[cols].copy()
        alt[c] = ref
        eff = p0 - est.predict_proba(alt)[:, 1]
        eff[rows[c].isna().to_numpy()] = 0.0
        out[c] = eff
    return pd.DataFrame(out, index=rows.index)


def reason_texts(contrib: pd.DataFrame, rows: pd.DataFrame, hist: pd.DataFrame, top: int = 3) -> list:
    """Tối đa `top` lý do làm TĂNG rủi ro (đóng góp > 0), sắp giảm dần; ghi giá trị và mốc tham chiếu."""
    texts = []
    for i in rows.index:
        c = contrib.loc[i].sort_values(ascending=False)
        c = c[c > 1e-9].head(top)
        parts = []
        for feat, val in c.items():
            v = rows.at[i, feat]
            ref = hist[feat].median() if pd.api.types.is_numeric_dtype(hist[feat]) and feat != "team" else None
            tail = f", trung vị train {_fmt(ref, feat)}" if ref is not None else ""
            parts.append(f"{FEATURE_VI.get(feat, feat)} = {_fmt(v, feat)}{tail}")
        texts.append("; ".join(parts) if parts else "(không có yếu tố nào vượt mức trung bình)")
    return texts


# ----------------------------------------------------------------------------
# Phiếu cảnh báo
# ----------------------------------------------------------------------------
def make_sheet(df: pd.DataFrame, date, k: int, lens: str, model: str, params: dict, ranking: str = "EV") -> dict:
    """Huấn luyện trên mọi ngày < date, chấm điểm ngày `date`. Trả về dict gồm bảng đầy đủ, bảng k tổ và metadata."""
    date = pd.Timestamp(date)
    assert model in PROBA_MODELS, "Phiếu cảnh báo cần mô hình có xác suất"
    hist = df[df["date"] < date].reset_index(drop=True)
    today = df[df["date"] == date].reset_index(drop=True)
    if today.empty:
        raise SystemExit(f"Ngày {date.date()} không có trong dữ liệu (cần các cột đặc trưng của ngày đó).")
    n_days = hist["date"].nunique()
    if n_days < MIN_HISTORY_DAYS:
        raise SystemExit(f"Chỉ có {n_days} ngày lịch sử trước {date.date()}; cần ít nhất {MIN_HISTORY_DAYS} ngày.")
    assert hist["date"].max() < date, "Phiếu cảnh báo chỉ được huấn luyện trên các ngày < D"
    cols = SCENARIOS["A"].columns

    est = E.fit_one(model, "A", params, E.select_X(hist, "A"), hist["miss"].to_numpy())
    p = rank_score(est, E.select_X(today, "A"))
    # hiệu chỉnh sigmoid chỉ từ lịch sử < D: điểm OOF của rolling-origin trên chính phần lịch sử.
    # Luôn tính để in xác suất đọc được theo nghĩa đen (điểm thô bị class_weight="balanced" đẩy lên).
    oof, _ = E.oof_predict(model, "A", params, hist, RollingOriginByDate(n_blocks=6))
    cal = CAL.Sigmoid(oof["score"].to_numpy(), hist.set_index("row_id").loc[oof["row_id"], "miss"].to_numpy())
    p_cal = cal.predict(p)
    p_used, note = p, ""
    if ranking == "EVcal":
        p_used = p_cal
        note = "xếp hạng bằng p đã hiệu chỉnh sigmoid (fit trên điểm out-of-fold của các ngày < D)"
    expo = C.exposure(today, lens)
    score = p_used if ranking == "p" else p_used * expo
    alerts = C.top_k_alerts(today, score, k)
    rank = pd.Series(-score).rank(method="first").astype(int).to_numpy()   # hạng 1 = ưu tiên nhất

    contrib = (contributions_logistic(est, today, hist, cols) if model == "Logistic"
               else contributions_vs_median(est, today, hist, cols))
    full = pd.DataFrame({
        "date": today["date"].dt.date, "rank": rank, "alert": alerts, "team": today["team"], "department": today["department"],
        "risk_score": p_used, "p_miss_cal": p_cal, "expected_damage": p_cal * expo, "damage_unit": C.LENS_UNIT[lens],
        "reasons": reason_texts(contrib, today, hist),
        "actual_productivity": today["actual_productivity"], "targeted_productivity": today["targeted_productivity"],
        "actual_miss": today["miss"],
    }).sort_values("rank").reset_index(drop=True)
    shortfall_frac = float(((hist["targeted_productivity"] - hist["actual_productivity"]) / hist["targeted_productivity"])[hist["miss"] == 1].mean())
    meta = {"date": date, "k": k, "lens": lens, "model": model, "ranking": ranking, "n_hist_days": n_days,
            "n_hist_rows": len(hist), "n_today": len(today), "calibration_note": note, "shortfall_frac": shortfall_frac,
            "reason_method": "hệ số x (giá trị chuẩn hoá - trung bình train)" if model == "Logistic" else "p(x) - p(x với đặc trưng đặt về trung vị train)"}
    return {"full": full, "sheet": full[full["alert"]].reset_index(drop=True), "meta": meta}


def _actual_text(r) -> str:
    return f"{'TRƯỢT' if r.actual_miss == 1 else 'đạt'} ({r.actual_productivity:.2f} so với định mức {r.targeted_productivity:.2f})"


def to_markdown(res: dict) -> str:
    m, s = res["meta"], res["sheet"]
    unit = C.LENS_UNIT[m["lens"]]
    lines = [f"# Phiếu cảnh báo đầu ca - {m['date'].date()} ({m['date'].day_name()})", "",
             f"- Mô hình: {MODEL_LABELS[m['model']]} (kịch bản A, siêu tham số chọn trên train), huấn luyện trên {m['n_hist_days']} ngày "
             f"({m['n_hist_rows']} dòng) trước ngày này.",
             f"- Xếp hạng: {RANKING_LABEL[m['ranking']]}; lăng kính {C.LENS_LABEL[m['lens']]}; k = {m['k']} tổ trong tổng {m['n_today']} tổ-bộ phận hôm nay."]
    if m["calibration_note"]:
        lines.append(f"- Lưu ý: {m['calibration_note']}.")
    lines += ["", f"| Hạng | Tổ | Bộ phận | Điểm rủi ro (dùng để xếp hạng) | Xác suất trượt ước tính (đã hiệu chỉnh) | Thiệt hại kỳ vọng ({unit}) | Lý do chính | Kết quả thực tế (chỉ để kiểm tra) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.rank} | {r.team} | {r.department} | {r.risk_score:.2f} | {r.p_miss_cal:.0%} | {r.expected_damage:,.0f} | {r.reasons} | {_actual_text(r)} |")
    hit = int(s["actual_miss"].sum())
    lines += ["", f"Kiểm tra (không dùng khi chấm điểm): {hit}/{len(s)} tổ được cảnh báo thực sự trượt; "
                  f"cả ngày có {int(res['full']['actual_miss'].sum())}/{m['n_today']} tổ-bộ phận trượt.", "",
              "Ghi chú: điểm rủi ro là đầu ra thô của mô hình, chỉ dùng để xếp hạng; nó cao hơn xác suất thật vì mô hình được huấn luyện "
              "với trọng số cân bằng lớp. Xác suất trượt ước tính là điểm này sau hiệu chỉnh sigmoid (fit trên điểm out-of-fold của các ngày < D), "
              "đọc được theo nghĩa đen. "
              "Thiệt hại kỳ vọng = xác suất trượt ước tính x quy mô mục tiêu của tổ "
              f"(targeted x số công nhân x 480{' / smv' if m['lens'] == 'b' else ''}); không phải ước tính tiền. "
              f"Khi trượt, năng suất thường thấp hơn định mức khoảng {m['shortfall_frac']:.0%} (trung bình lịch sử), nên thiệt hại thực thấp hơn quy mô mục tiêu. "
              f"Lý do chính tính bằng {m['reason_method']}; chỉ liệt kê yếu tố làm tăng rủi ro."]
    return "\n".join(lines) + "\n"


def write_sheet(res: dict, out_dir: Path = SHEETS_DIR) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = res["meta"]
    stem = f"alert_{m['date'].date()}_k{m['k']}_lens{m['lens']}"
    md, csv = out_dir / f"{stem}.md", out_dir / f"{stem}.csv"
    md.write_text(to_markdown(res), encoding="utf-8")
    res["full"].to_csv(csv, index=False, float_format="%.6g")
    return md, csv


# ----------------------------------------------------------------------------
# Giao diện dòng lệnh
# ----------------------------------------------------------------------------
def _read_config(ranking: str) -> tuple:
    """Mô hình, siêu tham số và kiểu xếp hạng đã chọn trên train (do run_all.py ghi ra)."""
    needed = [TABLES_DIR / n for n in ("model_choice.csv", "best_params.csv", "policy_choice.csv")]
    if not all(p.exists() for p in needed):
        raise SystemExit("Chưa có outputs/tables/model_choice.csv, best_params.csv, policy_choice.csv. Hãy chạy `python src/run_all.py` trước.")
    mc, bp, pc = (pd.read_csv(p) for p in needed)
    model = mc.loc[mc["is_policy_model"], "model"].iloc[0]
    params = E.params_from_str(bp[(bp["scenario"] == "A") & (bp["model"] == model)]["params"].iloc[0])
    if ranking == "auto":
        ranking = pc.loc[pc["chosen"], "ranking"].iloc[0]
    return model, params, ranking


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phiếu cảnh báo đầu ca: k tổ cần hỗ trợ trong ngày D.")
    ap.add_argument("--date", required=True, help="ngày D (YYYY-MM-DD), phải có trong dữ liệu")
    ap.add_argument("--k", type=int, default=C.K_BASE, help="số tổ cảnh báo (mặc định 3)")
    ap.add_argument("--lens", choices=C.LENSES, default=C.PRIMARY_LENS, help="đơn vị thiệt hại: a = phút chuẩn, b = sản phẩm (mặc định b)")
    ap.add_argument("--ranking", choices=["auto", "p", "EV", "EVcal"], default="auto", help="cách xếp hạng (auto: theo lựa chọn trên OOF)")
    ap.add_argument("--no-write", action="store_true", help="chỉ in, không ghi file")
    a = ap.parse_args(argv)
    model, params, ranking = _read_config(a.ranking)
    df = build_dataset(DATA_FILE)
    res = make_sheet(df, a.date, a.k, a.lens, model, params, ranking)
    print(to_markdown(res))
    if not a.no_write:
        md, csv = write_sheet(res)
        print(f"Đã ghi: {md.relative_to(ROOT)} và {csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
