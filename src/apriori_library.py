from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


@dataclass
class DataCleaner:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        if "InvoiceDate" in d.columns:
            d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="coerce")

        if "InvoiceNo" in d.columns:
            d = d[~d["InvoiceNo"].astype(str).str.startswith("C")]
        if "Quantity" in d.columns:
            d = d[pd.to_numeric(d["Quantity"], errors="coerce") > 0]
        if "UnitPrice" in d.columns:
            d = d[pd.to_numeric(d["UnitPrice"], errors="coerce") > 0]

        needed = [c for c in ["InvoiceNo", "Description"] if c in d.columns]
        if needed:
            d = d.dropna(subset=needed)

        if "Description" in d.columns:
            d["Description"] = d["Description"].astype(str).str.strip()

        return d.reset_index(drop=True)


@dataclass
class BasketPreparer:
    invoice_col: str = "InvoiceNo"
    item_col: str = "Description"
    quantity_col: str = "Quantity"

    def to_basket_bool(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[[self.invoice_col, self.item_col, self.quantity_col]].copy()
        d[self.quantity_col] = pd.to_numeric(d[self.quantity_col], errors="coerce").fillna(0)
        basket = (d.groupby([self.invoice_col, self.item_col])[self.quantity_col].sum().unstack(fill_value=0))
        return (basket > 0).astype(bool)


@dataclass
class AssociationRulesMiner:
    def run(self, basket_bool: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float):
        fi = apriori(basket_bool, min_support=min_support, use_colnames=True)
        fi["length"] = fi["itemsets"].apply(len)
        rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()
        return fi, rules


@dataclass
class FPGrowthMiner:
    def run(self, basket_bool: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float):
        fi = fpgrowth(basket_bool, min_support=min_support, use_colnames=True)
        fi["length"] = fi["itemsets"].apply(len)
        rules = association_rules(fi, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()
        return fi, rules


@dataclass
class DataVisualizer:
    def scatter_support_conf(self, rules: pd.DataFrame, title: str = "Support vs Confidence"):
        import matplotlib.pyplot as plt
        ax = rules.plot.scatter(x="support", y="confidence", figsize=(6,4), title=title)
        plt.show()

    def bar_top_lift(self, rules: pd.DataFrame, top_k: int = 15, title: str = "Top rules by lift"):
        import matplotlib.pyplot as plt
        d = rules.sort_values("lift", ascending=False).head(top_k).copy()
        d["rule"] = d["antecedents"].astype(str) + " -> " + d["consequents"].astype(str)
        ax = d.plot.barh(x="rule", y="lift", figsize=(8, max(4, top_k*0.35)), title=title)
        plt.tight_layout()
        plt.show()


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
