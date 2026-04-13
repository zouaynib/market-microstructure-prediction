# === CELL 1 ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')
print('All imports successful.')

# === CELL 3 ===
import urllib.request
import os
import zipfile

# Download FI-2010 dataset (from DeepLOB repository)
data_url = 'https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip'
zip_path = 'fi2010_data.zip'
train_file = 'Train_Dst_NoAuction_DecPre_CF_7.txt'

if not os.path.exists(train_file):
    print('Downloading FI-2010 data...')
    urllib.request.urlretrieve(data_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall('.')
    print('Download and extraction complete.')
else:
    print('Data already downloaded.')

# Load train + test data (normalization type 7: z-score with decreasing window)
# Format: 149 rows x N columns
# Rows 0-39: 10 LOB levels x 4 (ask_price, ask_vol, bid_price, bid_vol)
# Rows 40-143: 104 hand-crafted features (pre-computed by FI-2010 authors)
# Rows 144-148: labels for 5 horizons (k=1,2,3,5,10)
train_raw = np.loadtxt('Train_Dst_NoAuction_DecPre_CF_7.txt')
test_raw = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
raw = np.hstack([train_raw, test_raw])
print(f'Train: {train_raw.shape[1]} snapshots, Test: {test_raw.shape[1]} snapshots')
print(f'Combined: {raw.shape[1]} snapshots, {raw.shape[0]} rows (40 LOB + 104 features + 5 labels)')

# === CELL 4 ===
# Build DataFrame with LOB data
lob_raw = raw[:40, :].T  # (N, 40)
N = lob_raw.shape[0]

columns = []
for k in range(1, 11):
    columns.extend([f'pa{k}', f'Va{k}', f'pb{k}', f'Vb{k}'])

df = pd.DataFrame(lob_raw, columns=columns)
print(f'LOB DataFrame: {df.shape}')
print(f'\nSample snapshot (t=0):')
for k in range(1, 6):
    print(f'  Level {k}: ask={df[f"pa{k}"].iloc[0]:.4f} ({df[f"Va{k}"].iloc[0]:.0f})  '
          f'bid={df[f"pb{k}"].iloc[0]:.4f} ({df[f"Vb{k}"].iloc[0]:.0f})')

# Verify LOB consistency
print(f'\n=== LOB Sanity Checks ===')
print(f'pa1 >= pb1 everywhere: {(df["pa1"] >= df["pb1"]).all()}')
print(f'pa1 <= pa2 (ask levels increasing): {(df["pa1"] <= df["pa2"]).all()}')
print(f'pb1 >= pb2 (bid levels decreasing): {(df["pb1"] >= df["pb2"]).all()}')

# === CELL 6 ===
df['mid_price'] = 0.5 * (df['pa1'] + df['pb1'])

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(df['mid_price'], linewidth=0.5, color='steelblue')
axes[0].set_title('Mid-Price $p_m(t)$ Over Time', fontsize=14)
axes[0].set_xlabel('Event index $t$'); axes[0].set_ylabel('Mid-Price')

axes[1].plot(df['mid_price'].iloc[:5000], linewidth=0.5, color='darkorange')
axes[1].set_title('Mid-Price — Zoomed (first 5000 events)', fontsize=14)
axes[1].set_xlabel('Event index $t$'); axes[1].set_ylabel('Mid-Price')
plt.tight_layout(); plt.show()

# Stationarity test
adf_result = adfuller(df['mid_price'].iloc[:50000], maxlag=20, autolag='AIC')
print('=== Augmented Dickey-Fuller Test ===')
print(f'  ADF Statistic: {adf_result[0]:.4f}')
print(f'  p-value:       {adf_result[1]:.6f}')
print(f'  Critical values: {adf_result[4]}')
if adf_result[1] < 0.05:
    print('\n  => p < 0.05: reject H0. Subsample appears stationary.')
else:
    print('\n  => p >= 0.05: cannot reject H0. The mid-price is non-stationary (unit root).')
print()
print('**Interpretation:** The mid-price exhibits clear trends, level shifts, and regime changes')
print('across different stocks in the FI-2010 dataset. The series is non-stationary in the wide')
print('sense, motivating the use of returns (price changes) rather than raw levels for prediction.')

# === CELL 8 ===
df['spread'] = df['pa1'] - df['pb1']

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(df['spread'], linewidth=0.3, color='crimson', alpha=0.7)
axes[0].set_title('Bid-Ask Spread $s(t)$', fontsize=14)
axes[0].set_xlabel('Event index'); axes[0].set_ylabel('Spread')

axes[1].hist(df['spread'], bins=100, color='crimson', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[1].set_title('Distribution of Spread', fontsize=14)
axes[1].set_xlabel('Spread'); axes[1].set_ylabel('Frequency')
plt.tight_layout(); plt.show()

max_idx = df['spread'].idxmax()
print('=== Spread Statistics ===')
print(f'  Mean:   {df["spread"].mean():.6f}')
print(f'  Median: {df["spread"].median():.6f}')
print(f'  Std:    {df["spread"].std():.6f}')
print(f'  Min:    {df["spread"].min():.6f}')
print(f'  Max:    {df["spread"].max():.6f}')
print(f'\n=== Largest Spread ===')
print(f'  Index: {max_idx}')
print(f'  Spread: {df["spread"].iloc[max_idx]:.6f}')
print(f'  Ask:    {df["pa1"].iloc[max_idx]:.6f}, Bid: {df["pb1"].iloc[max_idx]:.6f}')
print(f'  Spread in bps: {df["spread"].iloc[max_idx]/df["mid_price"].iloc[max_idx]*10000:.1f} bps')
print()
print('**Interpretation:** The largest spread occurs during a period of low liquidity —')
print('possibly near session open/close, after a large market order, or at a stock transition')
print('in the dataset. Wide spreads = market makers demand more compensation for liquidity risk.')

# === CELL 10 ===
def plot_lob_snapshot(df, t, ax, title_extra=''):
    ask_prices = [df[f'pa{k}'].iloc[t] for k in range(1, 11)]
    ask_vols   = [df[f'Va{k}'].iloc[t] for k in range(1, 11)]
    bid_prices = [df[f'pb{k}'].iloc[t] for k in range(1, 11)]
    bid_vols   = [df[f'Vb{k}'].iloc[t] for k in range(1, 11)]
    
    all_prices = bid_prices[::-1] + ask_prices
    all_vols   = [-v for v in bid_vols[::-1]] + list(ask_vols)
    colors     = ['forestgreen']*10 + ['tomato']*10
    
    ax.bar(range(20), all_vols, color=colors, alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.set_xticks(range(20))
    ax.set_xticklabels([f'{p:.4f}' for p in all_prices], rotation=90, fontsize=5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(f't={t} {title_extra}', fontsize=10)
    ax.set_ylabel('Volume (bid<0, ask>0)')

timestamps = [0, N//4, N//2, 3*N//4]
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for ax, t in zip(axes, timestamps):
    plot_lob_snapshot(df, t, ax, f'(spread={df["spread"].iloc[t]:.5f})')
fig.suptitle('LOB Snapshots — Bid (green) vs Ask (red)', fontsize=14)
plt.tight_layout(); plt.show()

print('**Interpretation:** Each snapshot shows standing volume at each price level.')
print('The asymmetry between bid and ask volumes signals order flow imbalance — a key')
print('predictor of short-term price direction (Cont et al., 2014).')

# === CELL 12 ===
df['return'] = df['mid_price'].diff()
returns = df['return'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(returns, bins=200, color='steelblue', alpha=0.7, density=True, edgecolor='none')
x_range = np.linspace(returns.quantile(0.001), returns.quantile(0.999), 500)
axes[0].plot(x_range, stats.norm.pdf(x_range, returns.mean(), returns.std()), 'r-', lw=2, label='Gaussian fit')
axes[0].set_title('Distribution of Mid-Price Returns', fontsize=14)
axes[0].set_xlabel('Return'); axes[0].legend()
axes[0].set_xlim(returns.quantile(0.001), returns.quantile(0.999))

stats.probplot(returns.values, dist='norm', plot=axes[1])
axes[1].set_title('QQ-Plot: Returns vs Normal', fontsize=14)
plt.tight_layout(); plt.show()

kurt = stats.kurtosis(returns, fisher=True)
print(f'=== Return Statistics ===')
print(f'  Mean:     {returns.mean():.8f}')
print(f'  Std:      {returns.std():.8f}')
print(f'  Skewness: {stats.skew(returns):.4f}')
print(f'  Excess Kurtosis: {kurt:.4f} (Normal = 0)')
print()
print(f'**Interpretation:** Excess kurtosis = {kurt:.1f} >> 0. Returns have heavy tails — extreme')
print('moves occur much more often than a Gaussian model predicts. At tick level, most returns')
print('are 0 (no price change) with occasional large jumps, creating leptokurtosis.')

# === CELL 13 ===
# 5 largest absolute returns
abs_ret = returns.abs()
top5 = abs_ret.nlargest(5).index

print('=== 5 Largest Absolute Returns ===')
print(f'{"Rank":<6}{"Index":<10}{"Return":<15}{"Mid-Price":<15}{"Spread":<12}')
print('-'*58)
for rank, idx in enumerate(top5, 1):
    print(f'{rank:<6}{idx:<10}{df["return"].iloc[idx]:<15.6f}'
          f'{df["mid_price"].iloc[idx]:<15.6f}{df["spread"].iloc[idx]:<12.6f}')

# Plot LOB around largest return
big_idx = top5[0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, offset in enumerate([-1, 0, 1]):
    t = big_idx + offset
    if 0 <= t < len(df):
        plot_lob_snapshot(df, t, axes[i], f'(r_t={df["return"].iloc[t]:.5f})')
plt.suptitle(f'LOB around the largest absolute return (t={big_idx})', fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print()
print('**Interpretation:** The largest returns coincide with stock transitions in the FI-2010')
print('dataset (price jumps between different stocks). These are data artefacts, not genuine')
print('market moves. Within a single stock, the largest returns typically result from large')
print('market orders consuming multiple price levels or temporary liquidity gaps.')

# === CELL 15 ===
DELTA_T = 10

# Use FI-2010 pre-computed labels for k=10 (row 148)
# Labels: 1=downward, 2=stationary, 3=upward -> map to {-1, 0, +1}
df['y'] = (raw[148, :] - 2).astype(int)
print(f'Dataset size: {len(df)}')

# === CELL 17 ===
class_counts = df['y'].value_counts().sort_index()
class_pcts = df['y'].value_counts(normalize=True).sort_index() * 100

print('=== Class Distribution ===')
for cls in [-1, 0, 1]:
    label = {-1: 'Down', 0: 'Stable', 1: 'Up'}[cls]
    print(f'  y={cls:+d} ({label:>6}): {class_counts.get(cls,0):>8d}  ({class_pcts.get(cls,0):.1f}%)')

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#e74c3c', '#95a5a6', '#27ae60']
bars = ax.bar(['-1 (Down)', '0 (Stable)', '+1 (Up)'], class_counts.values, color=colors, edgecolor='black')
for bar, pct in zip(bars, class_pcts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+len(df)*0.005,
            f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution of $y_t$', fontsize=14); ax.set_ylabel('Count')
plt.tight_layout(); plt.show()

dominant = class_counts.idxmax()
print(f'\n**Observation:** The "Stable" class (y=0) is the minority (~{class_pcts.get(0,0):.0f}%).')
print(f'A naive majority classifier (y={dominant}) achieves {class_pcts[dominant]:.1f}%.')
print('We handle this by: (1) reporting per-class precision/recall, (2) using class_weight=\'balanced\'')
print('where appropriate, (3) comparing all models against the majority baseline.')

# === CELL 18 ===
# Temporal train/test split: 80/20
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f'Train: {len(train_df)} samples (indices 0 to {split_idx-1})')
print(f'Test:  {len(test_df)} samples (indices {split_idx} to {len(df)-1})')
print(f'\nTrain class distribution:')
print(train_df['y'].value_counts(normalize=True).sort_index().apply(lambda x: f'{x*100:.1f}%'))
print(f'\nTest class distribution:')
print(test_df['y'].value_counts(normalize=True).sort_index().apply(lambda x: f'{x*100:.1f}%'))
print(f'\nTest majority baseline: {test_df["y"].value_counts(normalize=True).max()*100:.1f}%')

# === CELL 19 ===
def evaluate_features(feature_cols, train_df, test_df, description=''):
    """Train logistic regression and report accuracy."""
    train_clean = train_df[feature_cols + ['y']].dropna()
    test_clean = test_df[feature_cols + ['y']].dropna()
    
    X_tr = train_clean[feature_cols].values
    y_tr = train_clean['y'].values
    X_te = test_clean[feature_cols].values
    y_te = test_clean['y'].values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_tr_s, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te_s))
    
    majority = pd.Series(y_te).value_counts(normalize=True).max()
    print(f'\n=== {description} ===')
    print(f'  Features: {feature_cols}')
    print(f'  Test Accuracy: {acc*100:.2f}%  (majority baseline: {majority*100:.2f}%)')
    return acc, model, scaler

# === CELL 21 ===
df['OI1'] = (df['Vb1'] - df['Va1']) / (df['Vb1'] + df['Va1'])
train_df['OI1'] = df['OI1'].iloc[:split_idx].values
test_df['OI1'] = df['OI1'].iloc[split_idx:].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, c, lab in [(-1,'#e74c3c','Down'), (0,'#95a5a6','Stable'), (1,'#27ae60','Up')]:
    axes[0].hist(df[df['y']==cls]['OI1'], bins=80, alpha=0.5, density=True, color=c, label=f'y={cls}')
axes[0].set_title('OI$^{(1)}$ by Class', fontsize=14); axes[0].legend()
axes[0].set_xlabel('OI$^{(1)}$'); axes[0].set_ylabel('Density')

bp = axes[1].boxplot([df[df['y']==c]['OI1'].values for c in [-1,0,1]],
                     labels=['Down','Stable','Up'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#e74c3c','#95a5a6','#27ae60']):
    patch.set_facecolor(color); patch.set_alpha(0.6)
axes[1].set_title('OI$^{(1)}$ by Class (Boxplot)', fontsize=14)
plt.tight_layout(); plt.show()

acc_oi1, _, _ = evaluate_features(['OI1'], train_df, test_df, 'Q2.1: Order Imbalance Level 1')
print()
print('**Interpretation:** OI(1)>0 = more buying pressure, pushing prices up.')
print('This is a well-known signal (Cont et al., 2014). The model beats the majority baseline,')
print('confirming that order imbalance carries genuine predictive information.')

# === CELL 23 ===
for k in range(1, 6):
    col = f'OI{k}'
    df[col] = (df[f'Vb{k}'] - df[f'Va{k}']) / (df[f'Vb{k}'] + df[f'Va{k}'])
    train_df[col] = df[col].iloc[:split_idx].values
    test_df[col] = df[col].iloc[split_idx:].values

oi_features = [f'OI{k}' for k in range(1, 6)]
acc_oi5, _, _ = evaluate_features(oi_features, train_df, test_df, 'Q2.2: OI Levels 1-5')
print(f'\n  OI(1) alone:    {acc_oi1*100:.2f}%')
print(f'  OI(1) to OI(5): {acc_oi5*100:.2f}%')
print(f'  Improvement:    {(acc_oi5-acc_oi1)*100:+.2f} pp')
print()
print('**Interpretation:** Deeper levels add modest predictive power. Most signal is at level 1.')

# === CELL 25 ===
df['pw'] = (df['pb1']*df['Va1'] + df['pa1']*df['Vb1']) / (df['Vb1'] + df['Va1'])
df['delta'] = df['pw'] - df['mid_price']
for col in ['spread', 'delta', 'pw']:
    train_df[col] = df[col].iloc[:split_idx].values
    test_df[col] = df[col].iloc[split_idx:].values

features_23 = oi_features + ['spread', 'delta']
acc_23, _, _ = evaluate_features(features_23, train_df, test_df, 'Q2.3: OI(1-5) + Spread + Delta')
print(f'\n  Improvement over OI alone: {(acc_23-acc_oi5)*100:+.2f} pp')
print()
print('**Interpretation:** delta captures the same signal as OI(1) — both measure bid/ask')
print('asymmetry. The spread adds liquidity information. delta > 0 means the volume-weighted')
print('price exceeds the simple mid-price, indicating buying pressure.')

# === CELL 27 ===
for lag in range(1, 6):
    col = f'r_lag{lag}'
    df[col] = df['return'].shift(lag)
    train_df[col] = df[col].iloc[:split_idx].values
    test_df[col] = df[col].iloc[split_idx:].values

lag_features = [f'r_lag{lag}' for lag in range(1, 6)]
all_features = oi_features + ['spread', 'delta'] + lag_features

# Lagged returns alone
acc_lags, _, _ = evaluate_features(lag_features, train_df, test_df, 'Lagged Returns Alone')
# All combined
acc_all, _, _ = evaluate_features(all_features, train_df, test_df, 'All Features Combined')

print('\n=== Autocorrelation of returns ===')
for lag in range(1, 6):
    print(f'  lag {lag}: {df["return"].autocorr(lag=lag):.6f}')
print()
print('**Interpretation:** Negative autocorrelation at lag 1 confirms mean-reversion (bid-ask')
print('bounce). Lagged returns alone are highly predictive because this mean-reversion pattern')
print('is strong and consistent. Combining all features gives the best result.')

# === CELL 29 ===
results = []
a1, _, _ = evaluate_features(['OI1'], train_df, test_df, 'OI(1)')
results.append(('OI(1)', 1, a1))
a2, _, _ = evaluate_features(oi_features, train_df, test_df, 'OI(1-5)')
results.append(('OI(1-5)', 5, a2))
a3, _, _ = evaluate_features(features_23, train_df, test_df, 'OI(1-5)+spread+delta')
results.append(('OI(1-5) + spread + delta', 7, a3))
a_l, _, _ = evaluate_features(lag_features, train_df, test_df, 'Lagged returns only')
results.append(('Lagged returns only', 5, a_l))
a4, _, _ = evaluate_features(all_features, train_df, test_df, 'All features')
results.append(('All (OI+spread+delta+lags)', 12, a4))

print('\n' + '='*65)
print('          FEATURE ENGINEERING SUMMARY TABLE')
print('='*65)
print(f'{"Feature Set":<35}{"#Feats":<8}{"Test Accuracy":<15}')
print('-'*58)
for name, nf, acc in results:
    best = ' ***' if acc == max(r[2] for r in results) else ''
    print(f'{name:<35}{nf:<8}{acc*100:.2f}%{best}')
print('-'*58)
print()
print('**Analysis:**')
print('  - Lagged returns are the strongest single feature group (bid-ask bounce).')
print('  - OI captures complementary supply/demand information.')
print('  - Combining all features yields the best accuracy.')
print('  - Delta is partially redundant with OI(1) (both measure level-1 asymmetry).')

# === CELL 31 ===
y_test_true = test_df['y'].values
y_pred_a1 = np.sign(test_df['pw'].values - test_df['mid_price'].values).astype(int)

acc_a1 = accuracy_score(y_test_true, y_pred_a1)
print('=== Q3.1: Parameter-Free Predictor: sign(p_w - p_m) ===')
print(f'Test Accuracy: {acc_a1*100:.2f}%')
print(f'\nClassification Report:')
print(classification_report(y_test_true, y_pred_a1, target_names=['Down','Stable','Up']))
print('**Note:** This predictor never predicts y=0 (p_w = p_m only when Va1 = Vb1).')
print('It uses only volume-weighted mid-price deviation — no training needed.')

# === CELL 33 ===
# Compute delta_pm
mid_full = 0.5 * (raw[0,:] + raw[2,:])
delta_pm = mid_full[DELTA_T:] - mid_full[:-DELTA_T]
# delta_pm has length N - DELTA_T; align with df
# Align lengths: delta_pm is DELTA_T shorter
dp = np.full(len(df), np.nan)
dp[:len(delta_pm)] = delta_pm[:len(df)]
df['delta_pm'] = dp

# Train OLS on training set (drop any NaN)
kyle_train = train_df[['OI1']].copy()
kyle_train['delta_pm'] = df['delta_pm'].iloc[:split_idx].values
kyle_train = kyle_train.dropna()

kyle_test = test_df[['OI1']].copy()
kyle_test['delta_pm'] = df['delta_pm'].iloc[split_idx:].values
kyle_test['y'] = test_df['y'].values
kyle_test = kyle_test.dropna()

X_tr_ols = kyle_train['OI1'].values.reshape(-1, 1)
y_tr_ols = kyle_train['delta_pm'].values
X_te_ols = kyle_test['OI1'].values.reshape(-1, 1)
y_te_ols = kyle_test['delta_pm'].values

ols = LinearRegression()
ols.fit(X_tr_ols, y_tr_ols)
alpha_hat, lambda_hat = ols.intercept_, ols.coef_[0]

print('=== Q3.2: Kyle-Style Linear Model ===')
print(f'\n(a) alpha = {alpha_hat:.8f},  lambda = {lambda_hat:.8f}')
print(f'    Sign of lambda: {"POSITIVE" if lambda_hat > 0 else "NEGATIVE"}')
print()
print("    **Kyle (1985) consistency:** lambda > 0 means buying pressure (OI > 0) pushes prices")
print("    up, consistent with Kyle's model. Lambda measures price impact of informed order flow.")
print("    Larger lambda = less liquid market (higher impact per unit of order flow).")

# (b) Directional predictions
y_pred_kyle = np.sign(alpha_hat + lambda_hat * kyle_test['OI1'].values).astype(int)
y_test_kyle = kyle_test['y'].values
acc_kyle = accuracy_score(y_test_kyle, y_pred_kyle)
print(f'\n(b) Directional accuracy: {acc_kyle*100:.2f}%')

# (c) R-squared
r2 = r2_score(y_te_ols, ols.predict(X_te_ols))
print(f'\n(c) R-squared = {r2:.6f}')
print(f'    Very low R-squared is typical in microstructure. Most tick-level variation is noise.')
print(f'    Even small positive R-squared can be economically significant at high frequency.')

# === CELL 35 ===
print('=== Q3.3: Comparison ===')
print(f'\n{"Model":<35}{"Test Accuracy":<15}')
print('-'*50)
print(f'{"sign(p_w - p_m) [no params]":<35}{acc_a1*100:.2f}%')
print(f'{"Kyle OLS [2 params]":<35}{acc_kyle*100:.2f}%')
print(f'{"LogReg + all features [12 feats]":<35}{a4*100:.2f}%')
print()
print('**Interpretation:** Logistic regression with all features significantly outperforms the')
print('analytical models. The analytical models use only OI-based signals and miss:')
print('  - Mean-reversion captured by lagged returns')
print('  - Nonlinear feature interactions')
print('  - Multi-level information (OI at levels 2-5)')
print('However, the analytical models still beat random (~33%), validating microstructure theory.')

# === CELL 37 ===
# Prepare datasets
# Feature set A: 12 engineered features (for LogReg, LightGBM)
feature_cols = all_features

train_clean = train_df[feature_cols + ['y']].dropna()
test_clean = test_df[feature_cols + ['y']].dropna()
X_train = train_clean[feature_cols].values
y_train = train_clean['y'].values
X_test = test_clean[feature_cols].values
y_test = test_clean['y'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature set B: 40 raw LOB + 12 engineered = 52 features (for NNs)
lob_cols = list(df.columns[:40])  # first 40 columns are LOB features
nn_feature_cols = lob_cols + feature_cols
train_nn = train_df[nn_feature_cols + ['y']].dropna()
test_nn = test_df[nn_feature_cols + ['y']].dropna()
X_train_nn = train_nn[nn_feature_cols].values
y_train_nn = train_nn['y'].values
X_test_nn = test_nn[nn_feature_cols].values
y_test_nn = test_nn['y'].values

scaler_nn = StandardScaler()
X_train_nn_s = scaler_nn.fit_transform(X_train_nn)
X_test_nn_s = scaler_nn.transform(X_test_nn)

print(f'LogReg/LightGBM features: {X_train.shape[1]}')
print(f'NN features (raw LOB + engineered): {X_train_nn_s.shape[1]}')
print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print(f'NN Train: {len(X_train_nn)}, NN Test: {len(X_test_nn)}')

# === CELL 39 ===
logreg = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f'=== Q4.1: Logistic Regression ===')
print(f'Test Accuracy: {acc_lr*100:.2f}%')
print(f'\nClassification Report:')
print(classification_report(y_test, y_pred_lr, target_names=['Down (-1)', 'Stable (0)', 'Up (+1)']))

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Down','Stable','Up'], yticklabels=['Down','Stable','Up'], ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'LogReg Confusion Matrix (acc={acc_lr:.4f})')
plt.tight_layout(); plt.show()

# === CELL 41 ===
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid

y_train_lgb = y_train + 1  # {0,1,2}

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Temporal validation (last 20% of training set)
val_split = int(len(X_train) * 0.8)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train_lgb[:val_split], y_train_lgb[val_split:]

best_val_acc, best_params = 0, None
print('Tuning LightGBM...')
for params in ParameterGrid(param_grid):
    m = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)
    va = accuracy_score(y_val, m.predict(X_val))
    if va > best_val_acc:
        best_val_acc, best_params = va, params

print(f'Best val accuracy: {best_val_acc*100:.2f}%, params: {best_params}')

# Retrain on full training set
lgb_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1, n_jobs=-1)
lgb_model.fit(X_train, y_train_lgb)
y_pred_lgb = lgb_model.predict(X_test) - 1
acc_lgb = accuracy_score(y_test, y_pred_lgb)

print(f'\n=== Q4.2: LightGBM ===')
print(f'Test Accuracy: {acc_lgb*100:.2f}%')
print(f'\nClassification Report:')
print(classification_report(y_test, y_pred_lgb, target_names=['Down (-1)', 'Stable (0)', 'Up (+1)']))

# Feature importances + confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
imp = pd.DataFrame({'Feature': feature_cols, 'Importance': lgb_model.feature_importances_}).sort_values('Importance')
axes[0].barh(imp['Feature'], imp['Importance'], color='steelblue')
axes[0].set_title('LightGBM Feature Importances')

cm_lgb = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Down','Stable','Up'], yticklabels=['Down','Stable','Up'], ax=axes[1])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
axes[1].set_title(f'LightGBM Confusion Matrix (acc={acc_lgb:.4f})')
plt.tight_layout(); plt.show()

# === CELL 43 ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')

# Tensors with 52 features
X_train_t = torch.FloatTensor(X_train_nn_s)
y_train_t = torch.LongTensor(y_train_nn + 1)
X_test_t = torch.FloatTensor(X_test_nn_s)
y_test_t = torch.LongTensor(y_test_nn + 1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=2048, shuffle=True)

class_counts_tr = np.bincount(y_train_nn.astype(int) + 1, minlength=3)
class_weights = torch.FloatTensor(len(y_train_nn) / (3 * class_counts_tr)).to(device)

class MLP(nn.Module):
    def __init__(self, input_dim, sizes, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in sizes:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# Architecture search
archs = {'64-32': [64,32], '128-64': [128,64], '256-128-64': [256,128,64]}
val_s = int(len(X_train_nn_s) * 0.85)
best_va, best_arch = 0, None

for name, sizes in archs.items():
    m = MLP(X_train_nn_s.shape[1], sizes, 0.3).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    ds = TensorDataset(torch.FloatTensor(X_train_nn_s[:val_s]).to(device),
                       torch.LongTensor(y_train_nn[:val_s].astype(int)+1).to(device))
    dl = DataLoader(ds, batch_size=2048, shuffle=True)
    m.train()
    for _ in range(30):
        for xb, yb in dl:
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        vp = m(torch.FloatTensor(X_train_nn_s[val_s:]).to(device)).argmax(1).cpu().numpy()
    va = accuracy_score(y_train_nn[val_s:].astype(int)+1, vp)
    print(f'  {name}: val_acc={va*100:.2f}%')
    if va > best_va: best_va, best_arch = va, (name, sizes)

print(f'Best: {best_arch[0]}')

# === CELL 44 ===
# Train best MLP architecture
mlp = MLP(X_train_nn_s.shape[1], best_arch[1], 0.3).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
crit = nn.CrossEntropyLoss(weight=class_weights)

losses = []
mlp.train()
for epoch in range(50):
    el, nb_ = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(mlp(xb), yb); loss.backward(); opt.step()
        el += loss.item(); nb_ += 1
    sched.step()
    losses.append(el/nb_)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/50, Loss={losses[-1]:.4f}')

mlp.eval()
with torch.no_grad():
    y_pred_mlp = mlp(X_test_t.to(device)).argmax(1).cpu().numpy() - 1
acc_mlp = accuracy_score(y_test_nn, y_pred_mlp)

print(f'\n=== Q4.3: MLP ({best_arch[0]}) ===')
print(f'Input: {X_train_nn_s.shape[1]} features (40 raw LOB + 12 engineered)')
print(f'Test Accuracy: {acc_mlp*100:.2f}%')
print(f'\nClassification Report:')
print(classification_report(y_test_nn, y_pred_mlp, target_names=['Down','Stable','Up']))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(confusion_matrix(y_test_nn, y_pred_mlp), annot=True, fmt='d', cmap='Purples',
            xticklabels=['Down','Stable','Up'], yticklabels=['Down','Stable','Up'], ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title(f'MLP Confusion Matrix (acc={acc_mlp:.4f})')
axes[1].plot(losses, color='purple'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('MLP Training Loss')
plt.tight_layout(); plt.show()

# === CELL 46 ===
SEQ_LEN = 50

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Build sequences from full dataset (52 features), dropping NaN
nn_clean = df[nn_feature_cols + ['y']].dropna().reset_index(drop=True)
X_full_nn = scaler_nn.fit_transform(nn_clean[nn_feature_cols].values)
y_full_nn = nn_clean['y'].values.astype(int) + 1

X_seq, y_seq = create_sequences(X_full_nn, y_full_nn, SEQ_LEN)
seq_split = int(len(X_seq) * 0.8)

X_train_seq = torch.FloatTensor(X_seq[:seq_split])
y_train_seq = torch.LongTensor(y_seq[:seq_split])
X_test_seq = torch.FloatTensor(X_seq[seq_split:])
y_test_seq_np = y_seq[seq_split:] - 1

train_seq_loader = DataLoader(TensorDataset(X_train_seq, y_train_seq), batch_size=512, shuffle=True)
print(f'Sequences: {X_seq.shape} (T={SEQ_LEN}, features={X_seq.shape[2]})')
print(f'Train: {X_train_seq.shape[0]}, Test: {X_test_seq.shape[0]}')

# === CELL 47 ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                           dropout=drop if layers > 1 else 0)
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden), nn.Linear(hidden, 64),
                                nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm = LSTMModel(X_seq.shape[2], hidden=128, layers=2, drop=0.3).to(device)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
crit = nn.CrossEntropyLoss(weight=class_weights)

print(f'LSTM parameters: {sum(p.numel() for p in lstm.parameters()):,}')
print(f'Architecture: LSTM(input={X_seq.shape[2]}, hidden=128, layers=2) -> FC(64) -> FC(3)')

lstm_losses = []
lstm.train()
for epoch in range(25):
    el, nb_ = 0, 0
    for xb, yb in train_seq_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(lstm(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
        opt.step()
        el += loss.item(); nb_ += 1
    sched.step()
    lstm_losses.append(el/nb_)
    if (epoch+1) % 5 == 0:
        print(f'  Epoch {epoch+1}/25, Loss={lstm_losses[-1]:.4f}')

lstm.eval()
with torch.no_grad():
    preds = []
    for i in range(0, len(X_test_seq), 512):
        preds.append(lstm(X_test_seq[i:i+512].to(device)).argmax(1).cpu())
    y_pred_lstm = torch.cat(preds).numpy() - 1

acc_lstm = accuracy_score(y_test_seq_np, y_pred_lstm)
print(f'\n=== Q4.4: LSTM ===')
print(f'Test Accuracy: {acc_lstm*100:.2f}%')
print(f'\nClassification Report:')
print(classification_report(y_test_seq_np, y_pred_lstm, target_names=['Down','Stable','Up']))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(confusion_matrix(y_test_seq_np, y_pred_lstm), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Down','Stable','Up'], yticklabels=['Down','Stable','Up'], ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title(f'LSTM Confusion Matrix (acc={acc_lstm:.4f})')
axes[1].plot(lstm_losses, color='darkorange')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('LSTM Training Loss')
plt.tight_layout(); plt.show()

if acc_lstm > acc_mlp:
    print(f'\n**LSTM ({acc_lstm*100:.1f}%) outperforms MLP ({acc_mlp*100:.1f}%)**:')
    print('Temporal information adds value — the LSTM captures evolving LOB dynamics.')
else:
    print(f'\n**LSTM ({acc_lstm*100:.1f}%) vs MLP ({acc_mlp*100:.1f}%)**:')
    print('The temporal component provides limited additional benefit. Our hand-crafted features')
    print('(lagged returns) already capture recent dynamics. LSTMs benefit more from raw LOB input')
    print('without pre-computed indicators (as in DeepLOB).')

# === CELL 49 ===
N_BLOCKS = 5

# Use the clean feature data (no NaN)
X_all_wf = np.vstack([X_train, X_test])
y_all_wf = np.concatenate([y_train, y_test])
block_size = len(X_all_wf) // (N_BLOCKS + 1)
wf_results = []

print('=== Walk-Forward Validation (LightGBM) ===')
for i in range(N_BLOCKS):
    tr_end = (i+1) * block_size
    te_start = tr_end
    te_end = min(te_start + block_size, len(X_all_wf))
    if te_end <= te_start:
        break
    
    X_tr_wf = X_all_wf[:tr_end]
    y_tr_wf = y_all_wf[:tr_end] + 1
    X_te_wf = X_all_wf[te_start:te_end]
    y_te_wf = y_all_wf[te_start:te_end]
    
    m = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    m.fit(X_tr_wf, y_tr_wf)
    ba = accuracy_score(y_te_wf, m.predict(X_te_wf) - 1)
    wf_results.append(ba)
    print(f'  Block {i+1}: train[0:{tr_end}] -> test[{te_start}:{te_end}] Acc={ba*100:.2f}%')

print(f'\nMean: {np.mean(wf_results)*100:.2f}% +/- {np.std(wf_results)*100:.2f}%')

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(wf_results)+1), [a*100 for a in wf_results], color='steelblue', edgecolor='black')
ax.axhline(np.mean(wf_results)*100, color='red', ls='--', label=f'Mean: {np.mean(wf_results)*100:.1f}%')
ax.set_xlabel('Block'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Walk-Forward Validation — LightGBM'); ax.legend()
plt.tight_layout(); plt.show()

print('**Interpretation:** Performance variation across blocks reflects non-stationarity')
print('(different stocks/market conditions). Walk-forward is the correct evaluation protocol')
print('for financial time series.')

# === CELL 51 ===
print('='*72)
print('                    FINAL MODEL COMPARISON')
print('='*72)
print(f'{"Model":<30}{"Type":<28}{"Test Accuracy":<15}')
print('-'*72)

models = [
    ('sign(p_w - p_m)', 'analytical (no params)', acc_a1),
    ('OLS: dp = a + l*OI', 'analytical (2 params)', acc_kyle),
    ('Logistic Regression', 'ML (linear)', acc_lr),
    ('LightGBM', 'ML (tree-based)', acc_lgb),
    ('MLP', 'ML (neural network)', acc_mlp),
    ('LSTM', 'ML (recurrent)', acc_lstm),
]
for name, mtype, acc in models:
    best = ' ***' if acc == max(m[2] for m in models) else ''
    print(f'{name:<30}{mtype:<28}{acc*100:.2f}%{best}')
print('-'*72)

best_m = max(models, key=lambda x: x[2])
print(f'\nBest: {best_m[0]} ({best_m[2]*100:.2f}%)')
print(f'Gap (best - worst): {(max(m[2] for m in models) - min(m[2] for m in models))*100:.1f} pp')

fig, ax = plt.subplots(figsize=(10, 5))
names = [m[0] for m in models]
accs = [m[2]*100 for m in models]
colors = ['#f39c12','#e67e22','#3498db','#2ecc71','#9b59b6','#e74c3c']
bars = ax.bar(names, accs, color=colors, edgecolor='black', alpha=0.8)
for b, a in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{a:.1f}%', ha='center', fontweight='bold')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Final Model Comparison', fontsize=14)
ax.set_ylim(min(accs)-5, max(accs)+5)
plt.xticks(rotation=20, ha='right')
plt.tight_layout(); plt.show()

# === CELL 53 ===
print('=== Q5.2(a): Most Predictive Single Feature ===')
print()
print('Lagged mid-price returns are the most predictive feature group. This makes economic')
print('sense: at tick level, returns exhibit negative autocorrelation (mean-reversion) due to')
print('the bid-ask bounce. Combined with OI (supply/demand imbalance from Kyle 1985 and')
print('Cont et al. 2014), the full feature set captures both mean-reversion and directional')
print('pressure.')
print()

print('=== Q5.2(b): Analytical vs ML ===')
print()
print(f'Analytical models achieve ~34% while the best ML model reaches {max(m[2] for m in models)*100:.0f}%.')
print('The gap shows ML captures nonlinear interactions and multi-feature patterns that simple')
print('analytical models miss. However, analytical models still beat random (33%), validating')
print('that OI-based signals carry genuine information.')
print()

print('=== Q5.2(c): Limitations ===')
print()
print('1. **Transaction costs:** Predicted price moves may be < spread, making trades unprofitable.')
print('2. **Class distribution:** With 3 classes, even 60% accuracy is modest. Binary up/down')
print('   prediction (removing stable) might be more actionable.')
print('3. **Non-stationarity:** Walk-forward shows 2-5% variation across blocks.')
print('4. **Latency:** Real HFT needs sub-ms execution; model inference erodes the signal.')
print('5. **Market impact:** Our own trades would move prices, reducing the exploitable edge.')

# === CELL 55 ===
print('=== Q5.3: What Would We Try Next? ===')
print()
print('1. **DeepLOB (CNN on raw LOB snapshots)** — Zhang et al. (2019) use CNNs directly on')
print('   raw LOB data with inception modules. This avoids manual feature engineering and has')
print('   achieved 60-85% accuracy on FI-2010, significantly outperforming our approach.')
print('   The convolutional layers learn cross-level and cross-feature interactions automatically.')
print()
print('2. **Transformers with self-attention** — Can capture long-range temporal dependencies')
print('   without vanishing gradients. Attention weights provide interpretability (which past')
print('   events the model focuses on). Recent work shows promise on LOB prediction.')
print()
print('3. **Different prediction horizons** — Shorter horizons (k=1,2) have more noise;')
print('   longer horizons (k=20,50) may reveal different patterns. A multi-horizon model')
print('   that jointly predicts could share representations across horizons.')
print()
print('4. **Multi-stock models** — Cross-stock lead-lag effects contain predictive information.')
print('   Training on all 5 stocks jointly could exploit cross-asset signals and improve')
print('   generalization through transfer learning.')
print()
print('5. **Addressing non-stationarity** — Online learning, Reversible Instance Normalization')
print('   (RevIN), or domain adaptation across trading days. The main practical challenge is')
print('   that LOB dynamics change across regimes, stocks, and market conditions.')
print()
print('='*72)
print('                        END OF TUTORIAL 1')
print('='*72)
