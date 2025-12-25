# Lab 2 – FP-Growth (Market Basket Analysis)

Dự án này triển khai pipeline phân tích giỏ hàng theo đúng mô tả trong **Lab 2 – FP-Growth**:
- Làm sạch dữ liệu giao dịch (Online Retail).
- Chuẩn bị dữ liệu dạng *basket* (Invoice × Item) và dạng boolean (*basket_bool*).
- Khai phá **Frequent Itemsets** và **Association Rules** bằng **FP-Growth**.
- So sánh với **Apriori** về thời gian chạy, số lượng itemsets, số lượng rules, độ dài itemset trung bình.
- Trực quan hoá rules (bar/scatter/network graph).

> **Lưu ý dataset**: repo này KHÔNG commit dữ liệu. Hãy đặt file `online_retail.csv` vào thư mục `data/`.

## 1) Cài đặt môi trường

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Chạy notebook theo thứ tự

1. `notebooks/preprocessing_and_eda.ipynb`
2. `notebooks/basket_preparation.ipynb`
3. `notebooks/apriori_modelling.ipynb` (tham chiếu – để so sánh)
4. `notebooks/fp_growth_modelling.ipynb`
5. `notebooks/compare_apriori_fpgrowth.ipynb`

## 3) Chạy pipeline tự động bằng papermill

```bash
python run_papermill.py --data_path data/online_retail.csv
```

Chạy cả Apriori và FP-Growth (mặc định). Kết quả notebook đã chạy sẽ nằm ở `executed_notebooks/`.

## 4) Cấu trúc thư mục

```
lab2-fpgrowth/
  src/apriori_library.py         # DataCleaner, BasketPreparer, AssociationRulesMiner, FPGrowthMiner, DataVisualizer
  notebooks/                     # Các notebook theo pipeline
  run_papermill.py               # Điều phối chạy notebook
  data/                          # Chứa online_retail.csv (không commit)
  figures/ reports/              # Output (không commit)
```

## 5) Gợi ý khi nộp bài

- Chụp/đưa hình các biểu đồ từ notebook (scatter/bar/network) vào blog/report.
- Nêu tối thiểu 5 insight kinh doanh từ các luật mạnh (support/confidence/lift) và đề xuất hành động.

---
Tác giả: (điền tên bạn)
