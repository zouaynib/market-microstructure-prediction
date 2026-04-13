# Market Microstructure: LOB Features & Mid-Price Prediction

**Course:** AI for Finance — M2 Paris-Saclay  
**Dataset:** FI-2010 benchmark (5 Finnish stocks, 10 trading days, 10 LOB levels)

## Objective

Explore Limit Order Book (LOB) data, engineer predictive features, and compare analytical and ML models for mid-price direction forecasting using the FI-2010 smoothed labels (horizon k=10).

## Structure

The notebook is organised into 5 parts:

| Part | Topic | Key Content |
|------|-------|-------------|
| 1 | Data Exploration | Mid-price stationarity (ADF test), bid-ask spread, LOB depth visualisation, return distribution & stylised facts |
| 2 | Feature Engineering | Order Imbalance OI(1-5), volume-weighted mid-price, spread, lagged returns (12 features total), ablation study, correlation matrix |
| 3 | Analytical Baselines | Parameter-free sign(pw-pm) predictor, Kyle (1985) linear price impact model (OLS) |
| 4 | ML Models | Logistic Regression, LightGBM (grid search), MLP (PyTorch), LSTM (PyTorch), walk-forward temporal validation |
| 5 | Synthesis | Comparison table, bar chart, key findings & limitations |

## Results

| Model | Type | Features | Accuracy |
|-------|------|----------|----------|
| sign(pw-pm) | Analytical | 2 | 32.57% |
| Kyle OLS | Analytical | 1 | 32.12% |
| Logistic Regression | ML (linear) | 12 | 45.93% |
| LightGBM | ML (tree) | 12 | 72.53% |
| MLP | ML (neural) | 52 | 47.79% |
| LSTM | ML (recurrent) | 52 | **86.32%** |

Walk-forward validation (LightGBM, 5 blocks): **66.96% +/- 5.36%**

## Files

```
TP7/
├── TD1_Market_Microstructure.ipynb   # Main notebook (fully executed)
├── Train_Dst_NoAuction_DecPre_CF_7.txt  # FI-2010 training data (149 x 254750)
├── Test_Dst_NoAuction_DecPre_CF_7.txt   # FI-2010 test data (149 x 55478)
├── fi2010_data.zip                      # Source archive
├── gen_notebook.py                      # Notebook generator script
├── build_notebook.py                    # Standalone computation script
└── README.md
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
statsmodels
scikit-learn
lightgbm
torch
```

## How to Run

Open the notebook in Jupyter:

```bash
jupyter notebook TD1_Market_Microstructure.ipynb
```

The notebook is pre-executed with all outputs (plots, tables, confusion matrices). To re-run, use **Kernel > Restart & Run All**. Full execution takes ~5 minutes (LSTM training is the bottleneck).

## Key Concepts

- **Mid-price**: pm(t) = 0.5 * (pa1 + pb1)
- **Bid-ask spread**: s(t) = pa1 - pb1
- **Order Imbalance**: OI(k) = (Vb(k) - Va(k)) / (Vb(k) + Va(k))
- **Volume-weighted mid-price**: pw = (pb1*Va1 + pa1*Vb1) / (Vb1 + Va1)
- **Kyle (1985)**: delta_pm = alpha + lambda * OI1 + epsilon (lambda > 0 = price impact)
- **FI-2010 smoothed labels**: Row 148, k=10 horizon — {1=down, 2=stable, 3=up}
