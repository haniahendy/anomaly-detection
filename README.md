# anomaly-detection

Compares a per-feature z-score rule with Isolation Forest on 2-D data where the injected anomalies break the correlation between features (so each feature alone looks normal).

## Run
```bash
pip install -r requirements.txt
python -m pytest -q   # 4 tests
python run.py
```

## Measured result (23 Nov 2026)
1030 points, 30 injected anomalies.

| method | precision | recall |
|---|---|---|
| z-score (threshold 3) | 0.333 | 0.067 |
| Isolation Forest (contamination 0.03) | 0.452 | 0.467 |

Isolation Forest is better here, but still misses over half of the anomalies and half its flags are wrong. Results are for one synthetic dataset and one seed.

## Limits
Synthetic data. `contamination` is set from the known anomaly rate, which you would not know in practice. No threshold tuning or time-series handling.

## Provenance
Built with AI assistance as a learning/portfolio project.
