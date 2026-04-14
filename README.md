# 📊 Market Microstructure — LOB Features & Mid-Price Prediction

> **Course:** AI for Finance — M2 Paris-Saclay  
> **Dataset:** [FI-2010 benchmark](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) — 5 Finnish stocks · 10 trading days · 10 LOB levels  
> **Task:** Binary classification of mid-price direction (horizon $k = 10$)

---

## 🗂️ Table of Contents

1. [Project Structure](#project-structure)
2. [Limit Order Book — Primer](#limit-order-book--primer)
3. [Key Concepts & Formulas](#key-concepts--formulas)
4. [Notebook Overview](#notebook-overview)
5. [Feature Engineering](#feature-engineering)
6. [Models & Results](#models--results)
7. [Walk-Forward Validation](#walk-forward-validation)
8. [Setup & Requirements](#setup--requirements)

---

## 📁 Project Structure

```
TP7/
├── TD1_Market_Microstructure.ipynb      # Main notebook (fully executed)
├── Train_Dst_NoAuction_DecPre_CF_7.txt  # FI-2010 training set  (149 × 254 750)
├── Test_Dst_NoAuction_DecPre_CF_7.txt   # FI-2010 test set      (149 × 55 478)
├── fi2010_data.zip                       # Source archive
├── gen_notebook.py                       # Notebook generator script
├── build_notebook.py                     # Standalone computation script
└── README.md
```

---

## 📖 Limit Order Book — Primer

A **Limit Order Book (LOB)** records all outstanding buy (bid) and sell (ask) orders at every price level. The top-of-book snapshot at time $t$ looks like this:

```
 Ask side (sellers)                     Bid side (buyers)
 ─────────────────────────────────────────────────────────
  Level │  Price  │  Volume       Volume │  Price  │ Level
 ───────┼─────────┼──────────   ─────────┼─────────┼───────
    5   │  pa5    │  Va5           Vb5   │  pb5    │   5
    4   │  pa4    │  Va4           Vb4   │  pb4    │   4
    3   │  pa3    │  Va3           Vb3   │  pb3    │   3
    2   │  pa2    │  Va2           Vb2   │  pb2    │   2
    1   │  pa1    │  Va1    ◄───►  Vb1   │  pb1    │   1
 ───────┴────────────────── Mid-price ───────────────────────
                       pm = (pa1 + pb1) / 2
```

The **FI-2010** dataset provides 10 levels on each side, giving 40 price/volume columns per snapshot, plus smoothed directional labels at five horizons ($k \in \{1, 2, 3, 5, 10\}$). This project targets **$k = 10$**.

**Label encoding** (row 148):

| Value | Direction |
|:-----:|-----------|
| `1` | ↓ Down |
| `2` | — Stationary |
| `3` | ↑ Up |

---

## 📐 Key Concepts & Formulas

### Mid-price

The reference price, computed from the best bid $p_{b1}$ and best ask $p_{a1}$:

$$p_m(t) = \frac{p_{a1}(t) + p_{b1}(t)}{2}$$

### Bid-Ask Spread

The transaction cost proxy and a direct measure of market liquidity:

$$s(t) = p_{a1}(t) - p_{b1}(t)$$

### Volume-Weighted Mid-Price

Weights the mid-price by the opposing side's depth, capturing the pressure-adjusted fair value:

$$p_w(t) = \frac{p_{b1}(t) \cdot V_{a1}(t) + p_{a1}(t) \cdot V_{b1}(t)}{V_{b1}(t) + V_{a1}(t)}$$

### Order Imbalance at Level $k$

Normalised imbalance between bid and ask volume at LOB level $k$, in $[-1, 1]$:

$$\text{OI}(k) = \frac{V_b(k) - V_a(k)}{V_b(k) + V_a(k)}$$

A value near $+1$ signals strong buying pressure; near $-1$, strong selling pressure.

### Kyle (1985) Linear Price Impact

Models price change as a linear function of signed order flow (here proxied by $\text{OI}_1$):

$$\Delta p_m = \alpha + \lambda \cdot \text{OI}_1 + \varepsilon, \quad \lambda > 0$$

$\lambda$ (Kyle's lambda) is the **price impact coefficient**: larger $\lambda$ means lower liquidity. Estimated by OLS.

### FI-2010 Smoothed Label (horizon $k$)

$$l_k(t) = \begin{cases} 1 & \text{if } \bar{p}_m(t+k) < \bar{p}_m(t) \cdot (1 - \alpha) \\ 3 & \text{if } \bar{p}_m(t+k) > \bar{p}_m(t) \cdot (1 + \alpha) \\ 2 & \text{otherwise} \end{cases}$$

where $\bar{p}_m$ is the mean mid-price over a smoothing window and $\alpha$ is a small threshold.

---

## 📓 Notebook Overview

| Part | Topic | Key Content |
|:----:|-------|-------------|
| **1** | Data Exploration | ADF stationarity test on mid-price, bid-ask spread distribution, LOB depth visualisation, return distribution & stylised facts |
| **2** | Feature Engineering | OI(1–5), $p_w$, spread, lagged returns → 12-feature baseline; ablation study; correlation matrix |
| **3** | Analytical Baselines | Parameter-free $\text{sign}(p_w - p_m)$ predictor; Kyle (1985) OLS |
| **4** | ML Models | Logistic Regression, LightGBM (grid search), MLP (PyTorch), LSTM (PyTorch) with walk-forward temporal split |
| **5** | Synthesis | Model comparison table & bar chart, key findings, limitations |

---

## 🔧 Feature Engineering

The 12-feature **baseline feature vector** $\mathbf{x}(t)$:

$$\mathbf{x}(t) = \Bigl[\underbrace{\text{OI}(1),\ldots,\text{OI}(5)}_{\text{5 imbalance features}},\; \underbrace{p_w - p_m}_{\text{pressure gap}},\; \underbrace{s(t)}_{\text{spread}},\; \underbrace{r_{t-1},\ldots,r_{t-5}}_{\text{5 lagged returns}} \Bigr]$$

The deep models (MLP, LSTM) additionally receive a **sliding window of 10 consecutive snapshots**, expanding the effective input to 52 features.

### Feature Correlation Heatmap (schematic)

```
          OI1   OI2   OI3   OI4   OI5   pw-pm  spread  r-1  r-2  r-3  r-4  r-5
OI1      [1.00  0.82  0.71  0.60  0.51   0.74   0.12  0.09  ...                ]
OI2      [0.82  1.00  0.85  0.74  0.63   0.68   0.10  ...                      ]
OI3      [0.71  0.85  1.00  0.87  0.76   0.61   0.09  ...                      ]
pw-pm    [0.74  0.68  0.61  0.55  0.48   1.00   0.22  0.11  ...                ]
spread   [0.12  0.10  0.09  0.08  0.07   0.22   1.00  0.04  ...                ]
r-1      [0.09  ...                            0.04   1.00  0.31  0.18  0.11   ]
```

*Adjacent OI levels are strongly correlated (depth persists across levels). Spread and lagged returns add orthogonal signal.*

---

## 🤖 Models & Results

### Architecture Summary

```
                     ┌─────────────────────────────────────────────┐
  Raw LOB snapshot   │  Feature extraction (Part 2)                │
  (149 rows)    ───► │  OI(1-5) · pw-pm · spread · lagged returns  │ ──► x(t) ∈ ℝ¹²
                     └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
             ┌────────────┐     ┌───────────────┐    ┌─────────────────────┐
             │ Analytical │     │ Tree / Linear │    │  Neural (sequence)  │
             │            │     │               │    │                     │
             │ sign(pw-pm)│     │ Logistic Reg. │    │ MLP  [52-dim input] │
             │ Kyle OLS   │     │ LightGBM ✦    │    │ LSTM [10 × 52]      │
             └─────┬──────┘     └──────┬────────┘    └──────────┬──────────┘
                   │                   │                         │
                   └───────────────────┴─────────────────────────┘
                                         │
                              FI-2010 smoothed label
                              l_10 ∈ {down, stable, up}
```

### Performance Comparison

| Model | Type | Input dim | Test Accuracy |
|-------|------|:---------:|:-------------:|
| Kyle OLS | Analytical | 1 | 32.12% |
| sign($p_w - p_m$) | Analytical | 2 | 32.57% |
| Logistic Regression | ML — linear | 12 | 45.93% |
| MLP (PyTorch) | ML — neural | 52 | 47.79% |
| LightGBM | ML — tree | 12 | 72.53% |
| **LSTM (PyTorch)** | **ML — recurrent** | **10 × 52** | **86.32% ✦** |

```
Accuracy
  90% ┤                                               ██
  80% ┤                                          ██   ██
  70% ┤                                     ██   ██   ██
  60% ┤                                     ██   ██   ██
  50% ┤                          ██    ██   ██   ██   ██
  40% ┤   ██    ██    ██    ██   ██    ██   ██   ██   ██
  30% ┤   ██    ██    ██    ██   ██    ██   ██   ██   ██
  20% ┤   ██    ██    ██    ██   ██    ██   ██   ██   ██
  10% ┤   ██    ██    ██    ██   ██    ██   ██   ██   ██
   0% ┼───┴─────┴─────┴─────┴───┴─────┴────┴────┴────┴──
      Kyle  sign  LogReg  MLP  LightGBM         LSTM
            (pw)
```

**Key takeaway:** Analytical baselines (~32%) are barely above the majority class. The step from linear ML → tree ensemble is dramatic (+27 pp). The LSTM's sequential modelling of the order book state drives a further +14 pp gain, consistent with the literature showing that temporal context is the dominant signal in LOB prediction.

---

## 🔄 Walk-Forward Validation

To respect the temporal structure of financial data, a **walk-forward (expanding window)** scheme is used for LightGBM. The dataset is split into 5 blocks:

```
Block  Train window           Test window        Accuracy
  1    ████░░░░░░░░░░░░        ▓▓▓░░░░░░░░░       62.3%
  2    ████████░░░░░░░░        ░░░▓▓▓░░░░░░       71.5%
  3    ████████████░░░░        ░░░░░░▓▓▓░░░       68.9%
  4    ████████████████        ░░░░░░░░░▓▓▓       64.7%
  5    ████████████████        ░░░░░░░░░░░░▓      67.4%
                                              ─────────
  Walk-forward mean ± std                    66.96% ± 5.36%
```

*The 5.4 pp standard deviation reflects genuine non-stationarity across trading days — a model trained on day $t$ cannot fully generalise to day $t+5$.*

> ⚠️ **Important caveat:** The 86.32% LSTM accuracy is obtained on a held-out test slice but **not** via walk-forward. FI-2010 is known to have near-perfect accuracy attainable with the right temporal split; these figures reflect the benchmark dataset's properties as much as model quality.

---

## ⚙️ Setup & Requirements

### Dependencies

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

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn lightgbm torch
```

### Running the Notebook

```bash
jupyter notebook TD1_Market_Microstructure.ipynb
```

The notebook ships **pre-executed** with all plots and confusion matrices. To reproduce from scratch:

**Kernel → Restart & Run All**

> ⏱ Full execution: ~5 minutes. The LSTM training loop is the bottleneck (~4 min on CPU).

---

## 📚 References

- Cao, C., Hansch, O., & Wang, X. (2009). *The information content of an open limit-order book*. Journal of Futures Markets.
- Kyle, A. S. (1985). *Continuous auctions and insider trading*. Econometrica, 53(6), 1315–1335.
- Ntakaris, A. et al. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting.
- Wallbridge, J. (2020). *Transformers for limit order books*. arXiv:2003.00130.
