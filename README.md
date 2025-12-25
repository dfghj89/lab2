# Lab 2 — FP-Growth (Online Retail / Basket Analysis)

Repo template theo hướng dẫn Lab-2:
- Tái sử dụng pipeline Lab-1 (EDA → Basket → Apriori)
- Bổ sung FP-Growth + notebook so sánh

## 1) Data
Đặt file CSV vào:
- `data/raw/online_retail.csv`

> Gợi ý: Không commit dữ liệu lên GitHub (đã có `.gitignore`).

## 2) Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Run all notebooks with Papermill
```bash
python run_papermill.py
```

Outputs:
- `data/processed/cleaned.parquet`
- `data/processed/basket_bool.parquet`
- `data/processed/apriori_rules.parquet`
- `data/processed/fpgrowth_rules.parquet`
- `data/processed/compare_metrics.csv`
- executed notebooks: `notebooks/runs/<timestamp>/...`
