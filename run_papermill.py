from __future__ import annotations
import datetime as dt
from pathlib import Path
import papermill as pm

RAW_CSV_PATH = "data/raw/online_retail.csv"
CLEANED_PATH = "data/processed/cleaned.parquet"
BASKET_PATH = "data/processed/basket_bool.parquet"

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.0
SUPPORT_GRID = [0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]

def run_one(in_path: Path, out_path: Path, params: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(str(in_path), str(out_path), parameters=params, log_output=True)

def main() -> None:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = Path("notebooks/runs")/ts
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_one(Path("notebooks/preprocessing_and_eda.ipynb"), runs_dir/"preprocessing_and_eda.run.ipynb",
            {"raw_csv_path": RAW_CSV_PATH, "output_cleaned_path": CLEANED_PATH})
    run_one(Path("notebooks/basket_preparation.ipynb"), runs_dir/"basket_preparation.run.ipynb",
            {"cleaned_path": CLEANED_PATH, "output_basket_path": BASKET_PATH})
    run_one(Path("notebooks/apriori_modelling.ipynb"), runs_dir/"apriori_modelling.run.ipynb",
            {"basket_path": BASKET_PATH, "min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE,
             "min_lift": MIN_LIFT, "output_rules_path": "data/processed/apriori_rules.parquet"})
    run_one(Path("notebooks/fp_growth_modelling.ipynb"), runs_dir/"fp_growth_modelling.run.ipynb",
            {"basket_path": BASKET_PATH, "min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE,
             "min_lift": MIN_LIFT, "output_rules_path": "data/processed/fpgrowth_rules.parquet"})
    run_one(Path("notebooks/compare_apriori_fpgrowth.ipynb"), runs_dir/"compare_apriori_fpgrowth.run.ipynb",
            {"basket_path": BASKET_PATH, "support_grid": SUPPORT_GRID, "min_confidence": MIN_CONFIDENCE,
             "min_lift": MIN_LIFT, "output_metrics_path": "data/processed/compare_metrics.csv"})
    print(f"Done. Outputs saved to: {runs_dir}")

if __name__ == "__main__":
    main()
