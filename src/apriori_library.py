"""
apriori_library.py

Trung tâm của project (giữ tên theo Lab 1/2):
- DataCleaner: làm sạch dữ liệu giao dịch
- BasketPreparer: chuyển transaction -> basket (Invoice x Item) và basket_bool
- AssociationRulesMiner: Apriori + association_rules
- FPGrowthMiner: FP-Growth + association_rules
- DataVisualizer: một vài biểu đồ minh hoạ rules

Thiết kế dựa theo mô tả cấu trúc project trong Lab 2 (FP-Growth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, Any, List

import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# -----------------------------
# Helpers
# -----------------------------

@dataclass
class RunStats:
    elapsed_seconds: float
    n_frequent_itemsets: int
    n_rules: int


def _ensure_dir(path: str) -> None:
    import os
    if not path:
        return
    os.makedirs(path, exist_ok=True)


# -----------------------------
# 1) Data Cleaning
# -----------------------------

class DataCleaner:
    """
    Làm sạch dữ liệu Online Retail (transaction-level).

    Mặc định kỳ vọng các cột:
    - InvoiceNo
    - StockCode
    - Description
    - Quantity
    - InvoiceDate
    - UnitPrice
    - CustomerID
    - Country

    Quy tắc làm sạch phổ biến:
    - Drop missing InvoiceNo/Description
    - Bỏ đơn huỷ: InvoiceNo bắt đầu bằng 'C' (nếu là string)
    - Chỉ giữ Quantity > 0 và UnitPrice > 0
    - Parse InvoiceDate sang datetime
    """

    def load_csv(self, path: str, encoding: Optional[str] = None) -> pd.DataFrame:
        if encoding:
            df = pd.read_csv(path, encoding=encoding)
        else:
            # thử vài encoding phổ biến
            for enc in ["utf-8", "ISO-8859-1", "latin1"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except Exception:
                    df = None
            if df is None:
                df = pd.read_csv(path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Basic null handling
        for col in ["InvoiceNo", "Description"]:
            if col in df.columns:
                df = df[df[col].notna()]

        # Strip description
        if "Description" in df.columns:
            df["Description"] = df["Description"].astype(str).str.strip()

        # Remove cancellations (InvoiceNo starting with 'C')
        if "InvoiceNo" in df.columns:
            inv = df["InvoiceNo"].astype(str)
            df = df[~inv.str.startswith("C")]

        # Filter positive quantity and price
        if "Quantity" in df.columns:
            df = df[df["Quantity"] > 0]
        if "UnitPrice" in df.columns:
            df = df[df["UnitPrice"] > 0]

        # Parse datetime
        if "InvoiceDate" in df.columns:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
            df = df[df["InvoiceDate"].notna()]

        # Optional: remove rows with empty Description
        if "Description" in df.columns:
            df = df[df["Description"].str.len() > 0]

        df.reset_index(drop=True, inplace=True)
        return df


# -----------------------------
# 2) Basket Preparation
# -----------------------------

class BasketPreparer:
    """
    Chuẩn bị dữ liệu basket:
    - basket: InvoiceNo x Item, giá trị là số lượng (quantity)
    - basket_bool: InvoiceNo x Item, giá trị boolean 0/1

    Bạn có thể lọc theo Country để giảm kích thước dữ liệu (nếu cần).
    """

    def create_basket(
        self,
        df: pd.DataFrame,
        invoice_col: str = "InvoiceNo",
        item_col: str = "Description",
        quantity_col: str = "Quantity",
        country: Optional[str] = None,
        min_item_len: int = 2,
    ) -> pd.DataFrame:
        _df = df.copy()
        if country is not None and "Country" in _df.columns:
            _df = _df[_df["Country"] == country]

        # Guard: items too short are usually noise
        if item_col in _df.columns:
            _df = _df[_df[item_col].astype(str).str.len() >= min_item_len]

        basket = (
            _df
            .groupby([invoice_col, item_col])[quantity_col]
            .sum()
            .unstack(fill_value=0)
        )
        return basket

    def to_bool(self, basket: pd.DataFrame) -> pd.DataFrame:
        basket_bool = basket.copy()
        basket_bool = (basket_bool > 0).astype(bool)
        return basket_bool


# -----------------------------
# 3) Association Rule Mining – Apriori
# -----------------------------

class AssociationRulesMiner:
    """
    Khai phá tập phổ biến và luật bằng Apriori (tham chiếu từ Lab 1).
    """

    def run(
        self,
        basket_bool: pd.DataFrame,
        min_support: float = 0.01,
        max_len: Optional[int] = None,
        use_colnames: bool = True,
        metric: str = "confidence",
        min_threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, RunStats]:
        start = time.perf_counter()
        frequent_itemsets = apriori(
            basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        elapsed = time.perf_counter() - start
        stats = RunStats(elapsed, len(frequent_itemsets), len(rules))
        return frequent_itemsets, rules, stats


# -----------------------------
# 4) Association Rule Mining – FP-Growth
# -----------------------------

class FPGrowthMiner:
    """
    Khai phá tập phổ biến và luật bằng FP-Growth (Lab 2).
    """

    def run(
        self,
        basket_bool: pd.DataFrame,
        min_support: float = 0.01,
        max_len: Optional[int] = None,
        use_colnames: bool = True,
        metric: str = "confidence",
        min_threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, RunStats]:
        start = time.perf_counter()
        frequent_itemsets = fpgrowth(
            basket_bool,
            min_support=min_support,
            use_colnames=use_colnames,
            max_len=max_len,
        )
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        elapsed = time.perf_counter() - start
        stats = RunStats(elapsed, len(frequent_itemsets), len(rules))
        return frequent_itemsets, rules, stats


# -----------------------------
# 5) Visualization
# -----------------------------

class DataVisualizer:
    """
    Một số biểu đồ minh hoạ rules:
    - bar chart: top rules theo lift (hoặc metric khác)
    - scatter: support vs confidence
    - network graph: quan hệ item -> item
    """

    def plot_top_rules_bar(
        self,
        rules: pd.DataFrame,
        metric: str = "lift",
        top_n: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        if rules.empty:
            raise ValueError("rules is empty")

        df = rules.sort_values(metric, ascending=False).head(top_n).copy()
        labels = df.apply(lambda r: f"{set(r['antecedents'])}→{set(r['consequents'])}", axis=1)

        plt.figure(figsize=(10, 5))
        plt.barh(range(len(df)), df[metric].values)
        plt.yticks(range(len(df)), labels)
        plt.gca().invert_yaxis()
        plt.xlabel(metric)
        plt.title(title or f"Top {top_n} rules by {metric}")

        plt.tight_layout()
        if save_path:
            _ensure_dir(str(__import__("os").path.dirname(save_path)))
            plt.savefig(save_path, dpi=200)
        plt.show()

    def plot_support_confidence_scatter(
        self,
        rules_a: pd.DataFrame,
        rules_b: Optional[pd.DataFrame] = None,
        label_a: str = "Apriori",
        label_b: str = "FP-Growth",
        save_path: Optional[str] = None,
    ) -> None:
        plt.figure(figsize=(7, 5))
        plt.scatter(rules_a["support"], rules_a["confidence"], alpha=0.5, label=label_a)
        if rules_b is not None:
            plt.scatter(rules_b["support"], rules_b["confidence"], alpha=0.5, label=label_b)

        plt.xlabel("support")
        plt.ylabel("confidence")
        plt.title("Support vs Confidence")
        plt.legend()
        plt.tight_layout()
        if save_path:
            _ensure_dir(str(__import__("os").path.dirname(save_path)))
            plt.savefig(save_path, dpi=200)
        plt.show()

    def plot_network_graph(
        self,
        rules: pd.DataFrame,
        top_n: int = 30,
        weight_col: str = "lift",
        title: str = "Association Rules Network",
        save_path: Optional[str] = None,
    ) -> None:
        if nx is None:
            raise ImportError("networkx is required for network graph. Install networkx.")

        df = rules.sort_values(weight_col, ascending=False).head(top_n).copy()

        G = nx.DiGraph()
        for _, r in df.iterrows():
            ants = list(r["antecedents"])
            cons = list(r["consequents"])
            w = float(r[weight_col])
            # add edges for all pairs antecedent -> consequent
            for a in ants:
                for c in cons:
                    G.add_edge(a, c, weight=w)

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42)

        # edge widths based on weight
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        if len(weights) == 0:
            raise ValueError("No edges to draw. Try a larger top_n or different thresholds.")
        w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        widths = 1 + 4 * w_norm

        nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, width=widths, arrows=True, alpha=0.6)

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        if save_path:
            _ensure_dir(str(__import__("os").path.dirname(save_path)))
            plt.savefig(save_path, dpi=200)
        plt.show()


# -----------------------------
# 6) Simple Weighted post-processing (optional extension)
# -----------------------------

class WeightedRulesPostProcessor:
    """
    Tuỳ chọn mở rộng theo phần gợi ý 'Weighted Association Rules' trong tài liệu Lab:
    - Tính InvoiceValue = sum(Quantity * UnitPrice) theo từng InvoiceNo
    - Với mỗi rule X=>Y, tính tổng InvoiceValue của hoá đơn chứa X∪Y (weighted_support)
    """

    def compute_invoice_value(
        self,
        df: pd.DataFrame,
        invoice_col: str = "InvoiceNo",
        quantity_col: str = "Quantity",
        price_col: str = "UnitPrice",
    ) -> pd.DataFrame:
        inv = (
            df.assign(_line_value=df[quantity_col] * df[price_col])
            .groupby(invoice_col)["_line_value"]
            .sum()
            .rename("InvoiceValue")
            .reset_index()
        )
        return inv

    def add_weighted_support(
        self,
        rules: pd.DataFrame,
        transactions: List[set],
        invoice_values: np.ndarray,
    ) -> pd.DataFrame:
        """
        transactions: list of sets, aligned with invoice_values
        invoice_values: numeric weights per transaction
        """
        if len(transactions) != len(invoice_values):
            raise ValueError("transactions and invoice_values must have same length")

        total_value = float(np.sum(invoice_values))
        out = rules.copy()

        weighted_supports = []
        for _, r in out.iterrows():
            X = set(r["antecedents"])
            Y = set(r["consequents"])
            XY = X.union(Y)

            mask = np.array([XY.issubset(t) for t in transactions], dtype=bool)
            ws = float(np.sum(invoice_values[mask])) / (total_value + 1e-12)
            weighted_supports.append(ws)

        out["weighted_support"] = weighted_supports
        return out
