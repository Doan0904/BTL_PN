"""Hàm vẽ hình (nhãn tiếng Việt, đủ trục/đơn vị/chú thích). Mỗi hàm trả về Figure; truyền `name` để lưu PNG (dpi >= 150)."""
import sys
import textwrap

import matplotlib

if "ipykernel" not in sys.modules:  # chạy script: dùng backend không giao diện; trong notebook giữ backend inline
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from config import FIG_DPI, FIGURES_DIR
from pipeline import BASELINE_LABELS, MODEL_LABELS

# Bảng màu Okabe-Ito (an toàn với mù màu); mỗi mô hình có màu + kiểu đường + marker cố định
MODEL_STYLE = {
    "DecisionTree": ("#0072B2", "-", "o"), "SVM_RBF": ("#E69F00", "-", "s"), "MLP": ("#009E73", "-", "^"),
    "Logistic": ("#CC79A7", "-", "D"), "RandomForest": ("#56B4E9", "-", "v"),
    "B0": ("#7f7f7f", ":", "x"), "B1": ("#D55E00", "--", "P"), "B2": ("#4d4d4d", "-.", "*"),
}
SHORT = {**MODEL_LABELS, "B0": "B0 (không cảnh báo)", "B1": "B1 (hôm qua trượt)", "B2": "B2 (mọi tổ finishing)"}
C_TRAIN, C_VAL, C_TEST, C_NEUTRAL = "#0072B2", "#E69F00", "#009E73", "#bdbdbd"
C_SEW, C_FIN = "#0072B2", "#D55E00"
INK, MUTED = "#222222", "#666666"


def setup_style():
    """Kiểu chung: nền trắng, lưới nhạt, font DejaVu Sans (hỗ trợ tiếng Việt)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": FIG_DPI,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


setup_style()


def finish(fig, name=None):
    """Bố cục gọn; lưu vào outputs/figures/<name> nếu có tên."""
    fig.tight_layout()
    if name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / name, dpi=FIG_DPI, bbox_inches="tight")
    return fig


def _line(ax, x, y, key, label=None, lw=1.8, ms=5, **kw):
    c, ls, m = MODEL_STYLE[key]
    ax.plot(x, y, color=c, ls=ls, marker=m, lw=lw, ms=ms, label=label or SHORT[key], **kw)


# ----------------------------------------------------------------------------
# EDA (mục 3.3)
# ----------------------------------------------------------------------------
def plot_productivity_distribution(df, name=None):
    """Hình 1: phân bố năng suất thực tế theo bộ phận, kèm đường định mức trung bình."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=False)
    bins = np.linspace(0, 1.15, 31)
    for ax, (dept, col) in zip(axes, (("sewing", C_SEW), ("finishing", C_FIN))):
        g = df[df["department"] == dept]
        ax.hist(g["actual_productivity"], bins=bins, color=col, alpha=0.75, edgecolor="white")
        t = g["targeted_productivity"].mean()
        ax.axvline(t, color=INK, ls="--", lw=1.6, label=f"Định mức trung bình = {t:.2f}")
        ax.axvline(g["actual_productivity"].mean(), color=INK, ls="-", lw=1.2, label=f"Thực tế trung bình = {g['actual_productivity'].mean():.2f}")
        ax.set_title(f"{'May (sewing)' if dept == 'sewing' else 'Hoàn thiện (finishing)'}  (n = {len(g)})")
        ax.set_xlabel("Năng suất thực tế (tỷ lệ so với chuẩn, không đơn vị)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Số tổ-ngày")
    fig.suptitle("Phân bố năng suất thực tế theo bộ phận", fontweight="bold")
    return finish(fig, name)


def plot_miss_rates(df, name=None):
    """Hình 2: tỷ lệ trượt định mức theo bộ phận, theo tổ và theo tuần trong tháng."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 2.2, 1.3]})
    overall = df["miss"].mean() * 100

    ax = axes[0]
    r = df.groupby("department")["miss"].mean() * 100
    ax.bar(["sewing", "finishing"], [r["sewing"], r["finishing"]], color=[C_SEW, C_FIN], width=0.6)
    for i, v in enumerate([r["sewing"], r["finishing"]]):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title("Theo bộ phận")

    ax = axes[1]
    t = df.pivot_table(index="team", columns="department", values="miss", aggfunc="mean") * 100
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["sewing"], 0.4, color=C_SEW, label="sewing")
    ax.bar(x + 0.2, t["finishing"], 0.4, color=C_FIN, label="finishing")
    ax.set_xticks(x, t.index)
    ax.set_xlabel("Mã tổ")
    ax.set_title("Theo tổ")
    ax.legend(ncol=2, loc="upper left")

    ax = axes[2]
    w = df.groupby("week_of_month")["miss"].agg(["mean", "size"])
    ax.bar(w.index, w["mean"] * 100, color="#7f7f7f", width=0.6)
    for i, (m, n) in w.iterrows():
        ax.text(i, m * 100 + 1.2, f"{m*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(w.index, [f"{i}\n(n={int(n)})" for i, n in w["size"].items()], fontsize=8.5)
    ax.set_xlabel("Tuần trong tháng (1: ngày 1-7, ..., 5: ngày 29-31)")
    ax.set_title("Theo tuần trong tháng")

    ymax = max(t.max().max(), r.max(), w["mean"].max() * 100) + 10
    for ax in axes:
        ax.axhline(overall, color=INK, ls="--", lw=1)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("Tỷ lệ tổ-ngày trượt định mức (%)")
    axes[0].text(0, overall + 1.5, f"TB toàn bộ = {overall:.1f}%", ha="center", fontsize=8.5)
    fig.suptitle("Tỷ lệ tổ-ngày không đạt định mức (miss = 1)", fontweight="bold")
    return finish(fig, name)


def plot_wip_missing(df, name=None):
    """Hình 3: ma trận thiếu dữ liệu (đen = thiếu) và tỷ lệ thiếu của wip theo bộ phận."""
    cols = ["targeted_productivity", "smv", "wip", "over_time", "incentive", "no_of_workers", "actual_productivity"]
    d = df.sort_values(["department", "date", "team"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.imshow(d[cols].isna().to_numpy().astype(int), aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8)
    n_fin = int((d["department"] == "finishing").sum())
    ax.axhline(n_fin - 0.5, color=C_FIN, lw=1.5)
    ax.text(len(cols) - 0.6, n_fin / 2, "finishing", rotation=90, va="center", ha="left", color=C_FIN, fontsize=9)
    ax.text(len(cols) - 0.6, n_fin + (len(d) - n_fin) / 2, "sewing", rotation=90, va="center", ha="left", color=C_SEW, fontsize=9)
    ax.set_ylabel("Dòng dữ liệu (sắp theo bộ phận, rồi theo ngày)")
    ax.set_title("Ma trận thiếu dữ liệu (đen = thiếu)")
    ax.grid(False)
    ax = axes[1]
    pct = df.groupby("department")["wip"].apply(lambda s: s.isna().mean() * 100)
    ax.bar(["sewing", "finishing"], [pct["sewing"], pct["finishing"]], color=[C_SEW, C_FIN], width=0.55)
    for i, v in enumerate([pct["sewing"], pct["finishing"]]):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Tỷ lệ dòng thiếu wip (%)")
    ax.set_title("wip thiếu có cấu trúc")
    return finish(fig, name)


def plot_boxplots(df, name=None):
    """Hình 4: hộp râu của incentive, over_time, wip (thang symlog vì đuôi dài và nhiều giá trị 0)."""
    spec = [("incentive", "Tiền thưởng incentive (BDT)"), ("over_time", "Tăng ca over_time (phút)"), ("wip", "Bán thành phẩm wip (sản phẩm)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    for ax, (col, lab) in zip(axes, spec):
        data, labels, colors = [], [], []
        for dept, c in (("sewing", C_SEW), ("finishing", C_FIN)):
            s = df.loc[df["department"] == dept, col].dropna()
            if len(s):
                data.append(s.to_numpy()); labels.append(f"{dept}\n(n={len(s)})"); colors.append(c)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5, flierprops={"marker": "o", "ms": 3, "alpha": 0.4},
                        medianprops={"color": INK})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.45)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylabel(lab + " - thang symlog")
        ax.set_title(col)
        if col == "wip":
            ax.text(0.5, 0.96, "finishing: không có wip (thiếu 100%)", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTED)
    fig.suptitle("Hộp râu các biến lệch phải (thang symlog: tuyến tính dưới 10, log phía trên)", fontweight="bold")
    return finish(fig, name)


def plot_wip_log1p(df, name=None):
    """Hình: wip của sewing trước và sau log1p (phân bố, độ lệch) và đường cong log1p."""
    w = df.loc[df["department"] == "sewing", "wip"].dropna().to_numpy()
    lw = np.log1p(w)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})

    def hist(ax, x, title, xlabel, bins):
        ax.hist(x, bins=bins, color=C_SEW, alpha=0.85, edgecolor="white")
        med = np.median(x)
        ax.axvline(med, color=INK, ls="--", lw=1.2, label=f"trung vị = {med:,.0f}".replace(",", ".") if med > 20 else f"trung vị = {med:.2f}")
        ax.axvline(x.max(), color=C_FIN, ls=":", lw=1.4, label=f"lớn nhất = {x.max():,.0f}".replace(",", ".") if x.max() > 20 else f"lớn nhất = {x.max():.2f}")
        ax.set_title(title + f"\nđộ lệch (skewness) = {pd.Series(x).skew():.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Số tổ-ngày (sewing)")
        ax.legend(loc="upper right")

    hist(axes[0], w, "Trước: wip gốc", "wip (sản phẩm dở dang)", 40)
    hist(axes[1], lw, "Sau: log1p(wip)", "log1p(wip) = ln(1 + wip)", 40)
    ax = axes[2]
    xs = np.linspace(0, 23500, 400)
    ax.plot(xs, np.log1p(xs), color=INK, lw=1.8)
    marks = [(100, C_SEW), (1000, C_SEW), (10000, C_FIN), (23122, C_FIN)]
    for v, c in marks:
        y = np.log1p(v)
        ax.plot([v, v], [0, y], color=c, lw=1, ls=":")
        ax.plot([0, v], [y, y], color=c, lw=1, ls=":")
        ax.plot(v, y, "o", color=c, ms=5)
        ax.annotate(f"{v:,}".replace(",", ".") + f" → {y:.2f}", (v, y), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("wip gốc")
    ax.set_ylabel("log1p(wip)")
    ax.set_title("Đường cong log1p: nén giá trị lớn\ngiữ nguyên thứ tự")
    ax.set_ylim(0, 11.5)
    return finish(fig, name)


def plot_wip_pipeline(train, name=None):
    """Hình: đặc trưng wip mô hình thực sự nhận (điền 0, chuẩn hoá bằng thống kê của train), không có và có log1p."""
    from sklearn.preprocessing import StandardScaler

    raw = train["wip"].fillna(0.0).to_numpy()[:, None]
    variants = [("Không log1p: điền 0 → chuẩn hoá", raw), ("Có log1p (cách dùng trong bài): điền 0 → log1p → chuẩn hoá", np.log1p(raw))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, (title, x) in zip(axes, variants):
        z = StandardScaler().fit_transform(x).ravel()
        fin = (train["department"] == "finishing").to_numpy()
        bins = np.linspace(z.min(), z.max(), 60)
        ax.hist(z[~fin], bins=bins, color=C_SEW, alpha=0.85, label="sewing", edgecolor="white")
        ax.hist(z[fin], bins=bins, color=C_FIN, alpha=0.85, label="finishing (wip điền 0)", edgecolor="white")
        sew = z[~fin]
        ax.set_title(title + f"\nsewing: 5%-95% nằm trong [{np.percentile(sew, 5):.2f}; {np.percentile(sew, 95):.2f}], lớn nhất = {z.max():.1f}")
        ax.set_xlabel("wip sau chuẩn hoá (đơn vị: độ lệch chuẩn)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Số tổ-ngày (tập train)")
    return finish(fig, name)


def plot_miss_timeseries(df, test_start, name=None):
    """Hình 5: tỷ lệ miss theo ngày (điểm) và trung bình trượt 7 ngày làm việc (đường)."""
    daily = df.groupby("date")["miss"].agg(["mean", "size"]).reset_index()
    daily["roll"] = daily["mean"].rolling(7, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.scatter(daily["date"], daily["mean"] * 100, s=18 + daily["size"], color="#7f7f7f", alpha=0.7, label="Tỷ lệ trượt từng ngày (kích thước ~ số tổ-ngày)")
    ax.plot(daily["date"], daily["roll"] * 100, color=C_VAL, lw=2.2, label="Trung bình trượt 7 ngày làm việc")
    ax.axhline(df["miss"].mean() * 100, color=INK, ls="--", lw=1, label=f"Trung bình toàn kỳ = {df['miss'].mean()*100:.1f}%")
    ax.axvspan(pd.Timestamp(test_start), df["date"].max() + pd.Timedelta(days=1), color=C_TEST, alpha=0.15, label="Giai đoạn test (từ 26/02/2015)")
    ax.set_xlabel("Ngày (2015)")
    ax.set_ylabel("Tỷ lệ tổ-ngày trượt định mức (%)")
    ax.set_title("Tỷ lệ miss theo ngày: biến động theo thời gian, nên phải kiểm định theo thời gian")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    return finish(fig, name)


# ----------------------------------------------------------------------------
# Kiểm định và kết quả
# ----------------------------------------------------------------------------
def plot_rolling_origin(train, test, cv, name=None):
    """Hình 6: sơ đồ rolling-origin theo ngày (5 fold) và tập test giữ lại."""
    block = cv.block_of_rows(train)
    dates = train["date"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    day = pd.Timedelta(days=1)
    for f in range(1, cv.n_blocks):
        y = cv.n_blocks - f
        tr = train.loc[block < f, "date"]; va = train.loc[block == f, "date"]; rest = train.loc[block > f, "date"]
        ax.barh(y, (tr.max() - tr.min()) + day, left=tr.min(), color=C_TRAIN, height=0.6)
        ax.barh(y, (va.max() - va.min()) + day, left=va.min(), color=C_VAL, height=0.6)
        if len(rest):
            ax.barh(y, (rest.max() - rest.min()) + day, left=rest.min(), color=C_NEUTRAL, height=0.6, alpha=0.6)
        ax.text(tr.min() - pd.Timedelta(days=1), y, f"Fold {f}", ha="right", va="center", fontsize=9)
    ax.barh(0, (test["date"].max() - test["date"].min()) + day, left=test["date"].min(), color=C_TEST, height=0.6)
    ax.text(train["date"].min() - pd.Timedelta(days=1), 0, "Test", ha="right", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Ngày (2015)")
    ax.set_title("Rolling-origin theo ngày: cửa sổ huấn luyện mở rộng, mọi dòng cùng ngày nằm cùng một phía")
    ax.legend(handles=[Patch(color=C_TRAIN, label="Huấn luyện"), Patch(color=C_VAL, label="Kiểm định (validate)"),
                       Patch(color=C_NEUTRAL, alpha=0.6, label="Chưa dùng ở fold này"), Patch(color=C_TEST, label="Test (chỉ dùng một lần ở cuối)")],
              loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.15))
    ax.grid(axis="y", visible=False)
    return finish(fig, name)


def plot_cv_fold_scores(folds, base_folds, scenario="A", name=None):
    """Hình 7: PR-AUC và ROC-AUC theo từng fold rolling-origin (kịch bản đã chọn), kèm B1 và tỷ lệ lớp dương."""
    f = folds[folds["scenario"] == scenario]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (metric, lab) in zip(axes, (("average_precision", "PR-AUC"), ("roc_auc", "ROC-AUC"))):
        for m in f["model"].unique():
            g = f[f["model"] == m]
            _line(ax, g["fold"], g[metric], m)
        b1 = base_folds[base_folds["model"] == "B1"]
        _line(ax, b1["fold"], b1[metric], "B1")
        if metric == "average_precision":
            b0 = base_folds[base_folds["model"] == "B0"]
            _line(ax, b0["fold"], b0[metric], "B0", label="B0 (= tỷ lệ lớp dương)")
        else:
            ax.axhline(0.5, color="#7f7f7f", ls=":", lw=1.2, label="Ngẫu nhiên (0,5)")
        ax.set_xticks(sorted(f["fold"].unique()))
        ax.set_xlabel("Fold rolling-origin (fold sau = giai đoạn muộn hơn)")
        ax.set_ylabel(f"{lab} trên tập validate (không đơn vị)")
        ax.set_title(f"{lab} theo fold - kịch bản {scenario}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.16))
    return finish(fig, name)


def plot_leakage(summary, name=None):
    """Hình 8: thí nghiệm rò rỉ - k-fold ngẫu nhiên 5x3 và rolling-origin (trung bình ± độ lệch chuẩn qua fold)."""
    models = [m for m in MODEL_LABELS if m in summary["model"].unique()]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharey="row")
    schemes = [("random", "k-fold ngẫu nhiên 5x3", "#CC79A7"), ("rolling", "Rolling-origin theo ngày", C_TRAIN)]
    for i, (metric, lab) in enumerate((("average_precision", "PR-AUC"), ("roc_auc", "ROC-AUC"))):
        for j, sc in enumerate(("A", "B")):
            ax = axes[i, j]
            s = summary[(summary["scenario"] == sc) & (summary["metric"] == metric)]
            x = np.arange(len(models))
            for o, (sch, slab, c) in enumerate(schemes):
                g = s[s["scheme"] == sch].set_index("model").loc[models]
                ax.bar(x + (o - 0.5) * 0.38, g["mean"], 0.38, yerr=g["std"], color=c, label=slab, capsize=2.5, error_kw={"lw": 1})
            ax.set_xticks(x, [MODEL_LABELS[m].replace(" (", "\n(") for m in models], fontsize=8)
            ax.set_title(f"{lab} - kịch bản {sc}")
            ax.set_ylabel(f"{lab} (không đơn vị)")
            if metric == "roc_auc":
                ax.axhline(0.5, color="#7f7f7f", ls=":", lw=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Chia ngẫu nhiên cho điểm cao hơn chia theo thời gian (cột và thanh sai số: TB ± độ lệch chuẩn qua các fold)", fontweight="bold", fontsize=10.5)
    return finish(fig, name)


def plot_scenario_ab(cv_summary, test_metrics, name=None):
    """Hình 9: PR-AUC của kịch bản A và B (CV với sai số ± std qua fold; test là điểm đơn)."""
    models = [m for m in MODEL_LABELS if m in set(cv_summary["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    x = np.arange(len(models))
    cols = {"A": C_TRAIN, "B": C_VAL}
    for o, sc in enumerate("AB"):
        c = cv_summary[cv_summary["scenario"] == sc].set_index("model").loc[models]
        axes[0].bar(x + (o - 0.5) * 0.38, c["average_precision_mean"], 0.38, yerr=c["average_precision_std"], color=cols[sc],
                    label=f"Kịch bản {sc}", capsize=2.5, error_kw={"lw": 1})
        t = test_metrics[test_metrics["scenario"] == sc].set_index("model").loc[models]
        axes[1].bar(x + (o - 0.5) * 0.38, t["average_precision"], 0.38, color=cols[sc], label=f"Kịch bản {sc}")
    prev_cv = cv_summary.loc[cv_summary["model"] == "B0", "average_precision_mean"].mean()
    prev_te = test_metrics.loc[test_metrics["model"] == "B0", "average_precision"].mean()
    for ax, p, t in ((axes[0], prev_cv, "Rolling-origin trên train (TB ± std qua fold)"), (axes[1], prev_te, "Test giữ lại")):
        ax.axhline(p, color="#7f7f7f", ls=":", lw=1.3, label="B0 = tỷ lệ lớp dương")
        ax.set_xticks(x, [MODEL_LABELS[m].replace(" (", "\n(") for m in models], fontsize=8)
        ax.set_title(t)
    axes[0].set_ylabel("PR-AUC (không đơn vị)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Kịch bản A (nghiêm ngặt) và B (thêm incentive, over_time, no_of_style_change)", fontweight="bold")
    return finish(fig, name)


# ----------------------------------------------------------------------------
# Chi phí (hai lăng kính)
# ----------------------------------------------------------------------------
LENS_TITLE = {"a": "Lăng kính (a): chi phí lao động", "b": "Lăng kính (b): giao hàng (chính)"}
LENS_YLABEL = {"a": "Lợi ích ròng TB mỗi ngày (phút chuẩn)", "b": "Lợi ích ròng TB mỗi ngày (sản phẩm)"}
SOURCE_LABEL = {"oof": "Out-of-fold (khối 1-5 của train)", "oof_b2_5": "Out-of-fold (khối 2-5 của train)", "test": "Test"}
SUFFIX_LABEL = {"p": "xếp theo p", "EV": "thiệt hại kỳ vọng", "EVcal": "thiệt hại kỳ vọng (p hiệu chỉnh)"}
SUFFIX_STYLE = {"p": "-", "EV": "--", "EVcal": ":"}


def policy_label(name: str) -> str:
    """'Logistic_EV' -> 'Hồi quy logistic - thiệt hại kỳ vọng'; baseline giữ nhãn gốc."""
    base, kind = (name, "baseline")
    for suf in ("_EVcal", "_EV"):
        if name.endswith(suf):
            base, kind = name[: -len(suf)], suf[1:]
            break
    else:
        kind = "baseline" if name in ("B0", "B1", "B2") else "p"
    return SHORT[base] if kind == "baseline" else f"{SHORT[base]} - {SUFFIX_LABEL[kind]}"


def _policy_line(ax, x, y, name, lw=1.8, ms=5, **kw):
    """Màu/marker theo mô hình gốc; kiểu đường theo loại chính sách (liền: p, đứt: EV, chấm: EV hiệu chỉnh)."""
    base = name.replace("_EVcal", "").replace("_EV", "")
    c, ls, m = MODEL_STYLE[base]
    if name != base:
        ls = SUFFIX_STYLE["EVcal" if name.endswith("_EVcal") else "EV"]
        lw += 0.5
    ax.plot(x, y, color=c, ls=ls, marker=m, lw=lw, ms=ms, label=policy_label(name), **kw)


def plot_net_benefit_vs_k(curve, source, k_base, title="", name=None, policies=None):
    """Hình 10/11: độ nhạy theo k - lợi ích ròng TB mỗi ngày ở hai lăng kính (k cố định = 3, đường dọc)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    c = curve[curve["source"] == source]
    keys = policies or list(dict.fromkeys(c["policy"]))
    for ax, lens in zip(axes, ("b", "a")):
        for m in keys:
            g = c[(c["lens"] == lens) & (c["policy"] == m)].sort_values("k")
            _policy_line(ax, g["k"], g["net_per_day"], m, lw=1.4, ms=4.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(k_base, color="#7f7f7f", ls="--", lw=1)
        ax.set_xlabel("k = số tổ cảnh báo mỗi ngày (k = 3 cố định; phần còn lại là độ nhạy)")
        ax.set_ylabel(LENS_YLABEL[lens])
        ax.set_title(LENS_TITLE[lens])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    if name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / name, dpi=FIG_DPI, bbox_inches="tight")
    return fig


def plot_recall_precision_k(curve, title="", name=None):
    """Hình 12: Recall@k và Precision@k (gộp mọi ngày) theo k; không phụ thuộc lăng kính."""
    c = curve[curve["lens"] == "b"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    keys = [m for m in list(MODEL_LABELS) + ["B1", "B2"] if m in set(c["policy"])]
    for ax, (col, lab) in zip(axes, (("recall_pooled", "Recall@k"), ("precision_pooled", "Precision@k"))):
        for m in keys:
            g = c[c["policy"] == m].sort_values("k")
            _line(ax, g["k"], g[col] * 100, m)
        ax.axvline(3, color="#7f7f7f", ls="--", lw=1)
        ax.set_xlabel("k = số tổ cảnh báo mỗi ngày")
        ax.set_ylabel(f"{lab} (%)")
        ax.set_title(lab)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle(title, fontweight="bold")
    return finish(fig, name)


def plot_policy_bars(summary, policies, name=None, title=""):
    """Hình 19: lợi ích ròng/ngày tại k = 3 của các chính sách (khoảng tin cậy 95% bootstrap theo ngày), lăng kính x tập."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for i, lens in enumerate(("b", "a")):
        for j, src in enumerate(("oof_b2_5", "test")):
            ax = axes[i, j]
            s = summary[(summary["lens"] == lens) & (summary["source"] == src)].set_index("policy").loc[policies]
            cols = [MODEL_STYLE[p.replace("_EVcal", "").replace("_EV", "")][0] for p in policies]
            x = np.arange(len(policies))
            err = np.vstack([s["net_per_day"] - s["net_per_day_ci_low"], s["net_per_day_ci_high"] - s["net_per_day"]])
            ax.bar(x, s["net_per_day"], color=cols, edgecolor="white",
                   hatch=None, yerr=err, capsize=3, error_kw={"lw": 1})
            for xi, p in zip(x, policies):
                if p.endswith("_EV") or p.endswith("_EVcal"):
                    ax.patches[xi].set_hatch("//")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, ["\n".join(textwrap.fill(part, 14) for part in policy_label(p).split(" - ")) for p in policies],
                          fontsize=7.5)
            ax.set_title(f"{LENS_TITLE[lens]} - {SOURCE_LABEL[src]}", fontsize=10)
            ax.set_ylabel(LENS_YLABEL[lens])
    fig.suptitle(title, fontweight="bold")
    return finish(fig, name)


def plot_cost_scenarios(sens, policies, name=None):
    """Hình 13: lợi ích ròng mỗi tháng dưới 3 kịch bản giả định (hàng: lăng kính; cột: OOF, test)."""
    order = ["Thận trọng", "Cơ sở", "Lạc quan"]
    unit = {"a": "nghìn phút chuẩn", "b": "nghìn sản phẩm"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    for i, lens in enumerate(("b", "a")):
        for j, src in enumerate(("oof_b2_5", "test")):
            ax = axes[i, j]
            s = sens[(sens["lens"] == lens) & (sens["source"] == src)]
            x = np.arange(len(order)); w = 0.8 / len(policies)
            for q, p in enumerate(policies):
                g = s[s["policy"] == p].set_index("cost_scenario").loc[order]
                base = p.replace("_EVcal", "").replace("_EV", "")
                bars = ax.bar(x + (q - (len(policies) - 1) / 2) * w, g["net_per_month"] / 1000, w, color=MODEL_STYLE[base][0],
                              label=policy_label(p), edgecolor="white")
                if p != base:
                    for bar in bars:
                        bar.set_hatch("//")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, order)
            ax.set_title(f"{LENS_TITLE[lens]} - {SOURCE_LABEL[src]}", fontsize=10)
            ax.set_ylabel(f"Lợi ích ròng/tháng ({unit[lens]}, 26 ngày)")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Độ nhạy theo 3 kịch bản giả định (r, C_int, c_min) tại k = 3", fontweight="bold")
    return finish(fig, name)


def plot_netgrid_heatmap(grid, policy, ref, name=None, source="oof_b2_5"):
    """Hình 14: lợi ích ròng/ngày của chính sách và hiệu so với tham chiếu, theo (r, C_int), cho cả hai lăng kính (OOF)."""
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for i, lens in enumerate(("b", "a")):
        g = grid[(grid["lens"] == lens) & (grid["source"] == source)]
        gm, gr = g[g["policy"] == policy], g[g["policy"] == ref]
        rs = sorted(gm["r"].unique()); fs = sorted(gm["c_int_factor"].unique())
        M = gm.pivot(index="r", columns="c_int_factor", values="net_per_day").loc[rs, fs].to_numpy()
        R = gr.pivot(index="r", columns="c_int_factor", values="net_per_day").loc[rs, fs].to_numpy()
        cabs = gm.drop_duplicates("c_int_factor").sort_values("c_int_factor")["c_int"].to_numpy()
        unit = "phút chuẩn" if lens == "a" else "sản phẩm"
        for ax, mat, ttl in ((axes[i, 0], M, f"{policy_label(policy)}: lợi ích ròng ({unit}/ngày)"),
                             (axes[i, 1], M - R, f"Chênh so với {policy_label(ref)} ({unit}/ngày)")):
            lim = max(abs(mat.min()), abs(mat.max()), 1e-9)
            im = ax.imshow(mat, origin="lower", cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
            ax.set_xticks(range(len(fs)), [f"x{f:g}\n({c:.0f})" if lens == "a" else f"x{f:g}\n({c:.1f})" for f, c in zip(fs, cabs)], fontsize=8)
            ax.set_yticks(range(len(rs)), [f"{r:.2f}" for r in rs])
            for a_ in range(len(rs)):
                for b_ in range(len(fs)):
                    ax.text(b_, a_, f"{mat[a_, b_]:.0f}", ha="center", va="center", fontsize=8.5,
                            color="white" if abs(mat[a_, b_]) > 0.55 * lim else INK)
            ax.set_xlabel(f"C_int: hệ số x giá trị cơ sở (giá trị, {unit}/lần)")
            ax.set_ylabel("r: tỷ lệ thiệt hại cứu được")
            ax.set_title(f"{LENS_TITLE[lens]}\n{ttl}", fontsize=9)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Độ nhạy theo r và C_int tại k = 3 ({SOURCE_LABEL[source]}; xanh: dương, đỏ: âm)", fontweight="bold")
    return finish(fig, name)


def plot_reliability(bins, stats, models, name=None):
    """Hình 20: đồ thị độ tin cậy (reliability) của xác suất, hàng: OOF/test; cột: mô hình. Kèm Brier."""
    fig, axes = plt.subplots(2, len(models), figsize=(3.6 * len(models), 7.0), sharex=True, sharey=True)
    for i, src in enumerate(("oof_b2_5", "test")):
        for j, m in enumerate(models):
            ax = axes[i, j]
            ax.plot([0, 1], [0, 1], color="#7f7f7f", ls="--", lw=1)
            col = MODEL_STYLE[m][0]
            for ver, ls, lab in (("raw", "-", "chưa hiệu chỉnh"), ("cal", ":", "hiệu chỉnh sigmoid")):
                g = bins[(bins["source"] == src) & (bins["model"] == m) & (bins["version"] == ver)].sort_values("bin")
                if g.empty:
                    continue
                ax.plot(g["mean_pred"], g["frac_pos"], color=col, ls=ls, marker="o" if ver == "raw" else "s", ms=5,
                        lw=1.8, mfc=col if ver == "raw" else "white", label=lab)
            st = stats[(stats["source"] == src) & (stats["model"] == m)].set_index("version")
            txt = f"Brier = {st.loc['raw', 'brier']:.3f}"
            if "cal" in st.index:
                txt += f"\nBrier (hiệu chỉnh) = {st.loc['cal', 'brier']:.3f}"
            ax.text(0.04, 0.95, txt, transform=ax.transAxes, va="top", fontsize=8, color=INK)
            ax.set_title(f"{SHORT[m]} - {'OOF khối 2-5' if src == 'oof_b2_5' else 'test'}", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 1:
                ax.set_xlabel("Xác suất dự báo TB trong nhóm")
            if j == 0:
                ax.set_ylabel("Tỷ lệ trượt thực trong nhóm")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Độ tin cậy của xác suất (5 nhóm theo phân vị của p; đường chéo = hiệu chỉnh hoàn hảo)", fontweight="bold")
    return finish(fig, name)


def plot_learning_curve(lc, models, name=None):
    """Hình 21: learning curve theo thời gian - PR-AUC trên khối validate cố định theo số ngày huấn luyện m."""
    blocks = sorted(lc["val_block"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.3 * len(blocks), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, vb in zip(axes, blocks):
        g = lc[lc["val_block"] == vb]
        for m in models:
            h = g[g["model"] == m].sort_values("m_days")
            _line(ax, h["m_days"], h["pr_auc"], m)
        b1 = g[g["model"] == "B1"].sort_values("m_days")
        ax.plot(b1["m_days"], b1["pr_auc"], color=MODEL_STYLE["B1"][0], ls="--", lw=1.8, label=SHORT["B1"] + " (không học)")
        ax.axhline(g["prevalence_val"].iloc[0], color="#7f7f7f", ls=":", lw=1.2, label="Tỷ lệ trượt của khối (đoán ngẫu nhiên)")
        ax.set_xlabel("m = số ngày huấn luyện ngay trước khối validate")
        ax.set_title(f"Khối validate {vb} ({int(g['n_val'].iloc[0])} dòng)")
    axes[0].set_ylabel("PR-AUC trên khối validate")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curve theo thời gian (kịch bản A, siêu tham số cố định)", fontweight="bold")
    return finish(fig, name)


# ----------------------------------------------------------------------------
# Diễn giải
# ----------------------------------------------------------------------------
def plot_perm_importance(pi, chosen, name=None):
    """Hình 15: permutation importance trên test (giảm PR-AUC khi xáo trộn một đặc trưng), mọi mô hình kịch bản A."""
    models = [m for m in MODEL_LABELS if m in set(pi["model"])]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    axes = axes.ravel()
    order = pi[pi["model"] == chosen].sort_values("mean", ascending=True)["feature"].tolist()
    for ax, m in zip(axes, models):
        g = pi[pi["model"] == m].set_index("feature").loc[order]
        c = MODEL_STYLE[m][0]
        ax.barh(order, g["mean"], xerr=g["std"], color=c, alpha=0.85, error_kw={"lw": 1}, capsize=2)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(SHORT[m] + (" (mô hình chính)" if m == chosen else ""), fontsize=10)
        ax.set_xlabel("Giảm PR-AUC khi xáo trộn (TB ± std, 30 lần)")
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle("Permutation importance trên tập test (kịch bản A; scoring = average precision)", fontweight="bold")
    return finish(fig, name)


def plot_pdp(pdp, features, model_label, ylabel="Điểm dự báo trung bình (xác suất trượt)", name=None):
    """Hình 16: partial dependence (đường đậm) và ICE (đường mảnh) cho các đặc trưng hàng đầu."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.9 * n, 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, features):
        g = pdp[pdp["feature"] == f]
        ice = g[g["curve"] != "pdp"]
        pd_ = g[g["curve"] == "pdp"].sort_values("order")
        numeric = pd_["is_numeric"].iloc[0]
        for _, c in ice.groupby("curve"):
            c = c.sort_values("order")
            if numeric:
                ax.plot(c["x"], c["y"], color="#9ecae1", lw=0.6, alpha=0.6)
        if numeric:
            ax.plot(pd_["x"], pd_["y"], color=C_TRAIN, lw=2.6, marker="o", ms=3.5)
        else:
            ax.bar(pd_["x_label"], pd_["y"], color=C_TRAIN, alpha=0.85)
            ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.set_xlabel(f)
        ax.set_title(f)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Partial dependence (đậm) và ICE (mảnh, mẫu 40 dòng) - {model_label}", fontweight="bold")
    return finish(fig, name)


def plot_coefficients(coef, name=None):
    """Hình 18: hệ số hồi quy logistic (đặc trưng số đã chuẩn hoá; one-hot giữ nguyên thang 0/1)."""
    c = coef.sort_values("coef")
    fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(c) + 1.6))
    ax.barh(c["feature"], c["coef"], color=np.where(c["coef"] > 0, C_FIN, C_TRAIN))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Hệ số logistic (dương: tăng nguy cơ trượt)")
    ax.set_title("Hệ số hồi quy logistic - kịch bản A")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(handles=[Patch(color=C_FIN, label="Tăng nguy cơ trượt"), Patch(color=C_TRAIN, label="Giảm nguy cơ trượt")], loc="lower right")
    return finish(fig, name)
