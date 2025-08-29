# An Efficient SQL Injection Detection using Hybrid CNN and RF Algorithm

This repository contains a single-file Python implementation of a hybrid **Char-CNN + RandomForest**
detector for SQL injection in raw query strings.

## Quickstart

```bash
pip install torch scikit-learn joblib pandas numpy
python hybrid_cnn_rf_sql_injection.py --data data/queries.csv --epochs 5 --batch-size 64
```

If you don't have a dataset yet, the script uses a tiny synthetic sample to illustrate the pipeline.

## Artifacts
- `artifacts/cnn_encoder.pt`
- `artifacts/rf_model.joblib`
- `artifacts/label_encoder.joblib`

## Prediction
```bash
python hybrid_cnn_rf_sql_injection.py --predict "SELECT * FROM users WHERE name='admin' OR '1'='1';"
```

---
Generated on: 2025-08-29 12:43:26
