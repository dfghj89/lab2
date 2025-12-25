"""
run_papermill.py

Script điều phối pipeline notebook bằng papermill (theo mô tả Lab 2).
Chạy:
    python run_papermill.py --data_path data/online_retail.csv

Mặc định chạy toàn bộ pipeline:
- preprocessing_and_eda
- basket_preparation
- apriori_modelling
- fp_growth_modelling
- compare_apriori_fpgrowth
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import papermill as pm


def run_one(input_nb: str, output_nb: str, params: dict) -> None:
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)
    pm.execute_notebook(
        input_path=input_nb,
        output_path=output_nb,
        parameters=params,
        kernel_name=None,  # use default
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/online_retail.csv", help="Path to online_retail.csv")
    ap.add_argument("--country", default=None, help="Optional Country filter (e.g., 'United Kingdom')")
    ap.add_argument("--min_support", type=float, default=0.01)
    ap.add_argument("--min_confidence", type=float, default=0.5)
    ap.add_argument("--min_lift", type=float, default=1.0)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("executed_notebooks", stamp)
    os.makedirs(out_dir, exist_ok=True)

    params = dict(
        DATA_PATH=args.data_path,
        COUNTRY=args.country,
        MIN_SUPPORT=args.min_support,
        MIN_CONFIDENCE=args.min_confidence,
        MIN_LIFT=args.min_lift,
    )

    notebooks = [
        "preprocessing_and_eda.ipynb",
        "basket_preparation.ipynb",
        "apriori_modelling.ipynb",
        "fp_growth_modelling.ipynb",
        "compare_apriori_fpgrowth.ipynb",
    ]

    for nb in notebooks:
        in_path = os.path.join("notebooks", nb)
        out_path = os.path.join(out_dir, nb)
        print(f"Running {in_path} -> {out_path}")
        run_one(in_path, out_path, params)

    print(f"Done. Executed notebooks saved to: {out_dir}")


if __name__ == "__main__":
    main()
