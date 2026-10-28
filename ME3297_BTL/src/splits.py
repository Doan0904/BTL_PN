"""Tách test theo ngày và bộ chia rolling-origin theo ngày."""
import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2015-02-26")  # mọi dòng có date >= mốc này là test


def split_train_test(df: pd.DataFrame, test_start: pd.Timestamp = TEST_START):
    """Tách theo ranh giới ngày; kiểm tra không ngày nào nằm ở cả hai tập."""
    train = df[df["date"] < test_start].reset_index(drop=True)
    test = df[df["date"] >= test_start].reset_index(drop=True)
    assert_no_date_overlap(train, test)
    return train, test


def assert_no_date_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Không có ngày nào xuất hiện ở cả train và test, và mọi ngày test đứng sau ngày train."""
    shared = set(train["date"]) & set(test["date"])
    assert not shared, f"Ngày xuất hiện ở cả train và test: {sorted(shared)[:3]}"
    assert train["date"].max() < test["date"].min(), "Train phải đứng hoàn toàn trước test"


class RollingOriginByDate:
    """Rolling-origin theo ngày, tương thích sklearn (có split và get_n_splits).

    Các ngày DUY NHẤT của tập train được chia thành `n_blocks` khối liên tiếp.
    Fold i (i = 1..n_blocks-1): huấn luyện trên khối 0..i-1, kiểm định trên khối i.
    Mọi dòng cùng ngày nằm cùng một phía. Với n_blocks=6 có 5 fold.
    X phải là DataFrame có cột ngày (mặc định 'date').
    """

    def __init__(self, n_blocks: int = 6, date_col: str = "date"):
        self.n_blocks = n_blocks
        self.date_col = date_col

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_blocks - 1

    def block_of_rows(self, X: pd.DataFrame) -> np.ndarray:
        """Chỉ số khối (0..n_blocks-1) của từng dòng theo ngày."""
        dates = np.sort(pd.Series(X[self.date_col]).unique())
        blocks = np.array_split(dates, self.n_blocks)
        block_of_date = {d: b for b, ds in enumerate(blocks) for d in ds}
        return pd.Series(X[self.date_col]).map(block_of_date).to_numpy()

    def split(self, X, y=None, groups=None):
        block = self.block_of_rows(X)
        for i in range(1, self.n_blocks):
            yield np.flatnonzero(block < i), np.flatnonzero(block == i)

    def fold_table(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bảng mô tả từng fold (khoảng ngày, số dòng) để vẽ sơ đồ và đưa vào báo cáo."""
        rows = []
        for f, (tr, va) in enumerate(self.split(X), start=1):
            d = pd.Series(X[self.date_col])
            rows.append({
                "fold": f,
                "train_from": d.iloc[tr].min().date(), "train_to": d.iloc[tr].max().date(),
                "val_from": d.iloc[va].min().date(), "val_to": d.iloc[va].max().date(),
                "n_train": len(tr), "n_val": len(va),
                "n_days_train": d.iloc[tr].nunique(), "n_days_val": d.iloc[va].nunique(),
            })
        return pd.DataFrame(rows)

    def __repr__(self):
        return f"RollingOriginByDate(n_blocks={self.n_blocks}, date_col='{self.date_col}')"


def assert_cv_folds_valid(cv: RollingOriginByDate, X: pd.DataFrame) -> None:
    """Mỗi fold: ngày validate đứng sau mọi ngày huấn luyện và không ngày nào xuất hiện ở cả hai phía."""
    d = pd.Series(X[cv.date_col]).reset_index(drop=True)
    for tr, va in cv.split(X):
        assert set(d.iloc[tr]).isdisjoint(set(d.iloc[va])), "Một ngày nằm ở cả train và validate của cùng fold"
        assert d.iloc[tr].max() < d.iloc[va].min(), "Validate phải đứng sau train trong thời gian"
