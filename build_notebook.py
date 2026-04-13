"""
Build and execute the TD1 Market Microstructure notebook.
This script creates a clean .ipynb file with all outputs.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
sns.set_style('whitegrid')

# ===== DATA LOADING =====
print("Loading data...")
train_raw = np.loadtxt('Train_Dst_NoAuction_DecPre_CF_7.txt')
test_raw = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
raw = np.hstack([train_raw, test_raw])
print(f"Combined: {raw.shape[1]} snapshots, {raw.shape[0]} rows")

lob = raw[:40, :].T
N = lob.shape[0]
cols = []
for k in range(1, 11):
    cols.extend([f'pa{k}', f'Va{k}', f'pb{k}', f'Vb{k}'])
df = pd.DataFrame(lob, columns=cols)

# ===== PART 1 =====
print("\n=== PART 1: Data Exploration ===")

# Q1.1 Mid-price
df['mid_price'] = 0.5 * (df['pa1'] + df['pb1'])
adf = adfuller(df['mid_price'].iloc[:50000], maxlag=20, autolag='AIC')
print(f"Q1.1: ADF stat={adf[0]:.4f}, p={adf[1]:.6f} -> {'non-stationary' if adf[1]>0.05 else 'stationary'}")

# Q1.2 Spread
df['spread'] = df['pa1'] - df['pb1']
max_spread_idx = df['spread'].idxmax()
print(f"Q1.2: Spread mean={df['spread'].mean():.6f}, max={df['spread'].max():.6f} at idx={max_spread_idx}")

# Q1.4 Returns
df['return'] = df['mid_price'].diff()
ret = df['return'].dropna()
kurt = stats.kurtosis(ret, fisher=True)
print(f"Q1.4: Kurtosis={kurt:.2f} (heavy tails)")

# ===== LABELS (FI-2010 smoothed) =====
# Row 148 = k=10 horizon labels: 1=down, 2=stable, 3=up -> {-1, 0, +1}
df['y'] = (raw[148, :] - 2).astype(int)
print(f"\nLabels: {dict(zip(*np.unique(df['y'], return_counts=True)))}")

# ===== TRAIN/TEST SPLIT =====
split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df = df.iloc[split:].copy()
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Test majority baseline: {test_df['y'].value_counts(normalize=True).max()*100:.1f}%")

# ===== PART 2: FEATURE ENGINEERING =====
print("\n=== PART 2: Feature Engineering ===")

# OI levels 1-5
for k in range(1, 6):
    c = f'OI{k}'
    df[c] = (df[f'Vb{k}'] - df[f'Va{k}']) / (df[f'Vb{k}'] + df[f'Va{k}'])
    train_df[c] = df[c].iloc[:split].values
    test_df[c] = df[c].iloc[split:].values

# Spread & delta
df['pw'] = (df['pb1']*df['Va1'] + df['pa1']*df['Vb1']) / (df['Vb1'] + df['Va1'])
df['delta'] = df['pw'] - df['mid_price']
for c in ['spread', 'delta', 'pw']:
    train_df[c] = df[c].iloc[:split].values
    test_df[c] = df[c].iloc[split:].values

# Lagged returns
for lag in range(1, 6):
    c = f'r_lag{lag}'
    df[c] = df['return'].shift(lag)
    train_df[c] = df[c].iloc[:split].values
    test_df[c] = df[c].iloc[split:].values

oi_feats = [f'OI{k}' for k in range(1, 6)]
lag_feats = [f'r_lag{lag}' for lag in range(1, 6)]
all_feats = oi_feats + ['spread', 'delta'] + lag_feats

def eval_lr(feats, desc):
    tr = train_df[feats + ['y']].dropna()
    te = test_df[feats + ['y']].dropna()
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[feats].values)
    Xte = sc.transform(te[feats].values)
    m = LogisticRegression(max_iter=1000, random_state=42)
    m.fit(Xtr, tr['y'].values)
    acc = accuracy_score(te['y'].values, m.predict(Xte))
    print(f"  {desc}: {acc*100:.2f}%")
    return acc

acc_oi1 = eval_lr(['OI1'], 'OI(1)')
acc_oi5 = eval_lr(oi_feats, 'OI(1-5)')
acc_23 = eval_lr(oi_feats + ['spread', 'delta'], 'OI+spread+delta')
acc_lags = eval_lr(lag_feats, 'Lagged returns alone')
acc_all = eval_lr(all_feats, 'All 12 features')

# ===== PART 3: ANALYTICAL BASELINES =====
print("\n=== PART 3: Analytical Baselines ===")

# Q3.1 Parameter-free
y_true = test_df['y'].values
y_pred_a1 = np.sign(test_df['pw'].values - test_df['mid_price'].values).astype(int)
acc_a1 = accuracy_score(y_true, y_pred_a1)
print(f"Q3.1 sign(pw-pm): {acc_a1*100:.2f}%")

# Q3.2 Kyle
DELTA_T = 10
mid_full = 0.5 * (raw[0, :] + raw[2, :])
delta_pm = mid_full[DELTA_T:] - mid_full[:-DELTA_T]
dp = np.full(len(df), np.nan)
dp[:len(delta_pm)] = delta_pm
df['delta_pm'] = dp

kyle_tr = train_df[['OI1']].copy()
kyle_tr['dpm'] = df['delta_pm'].iloc[:split].values
kyle_tr = kyle_tr.dropna()

kyle_te = test_df[['OI1', 'y']].copy()
kyle_te['dpm'] = df['delta_pm'].iloc[split:].values
kyle_te = kyle_te.dropna()

ols = LinearRegression()
ols.fit(kyle_tr[['OI1']].values, kyle_tr['dpm'].values)
alpha_hat, lambda_hat = ols.intercept_, ols.coef_[0]
print(f"Q3.2 Kyle: alpha={alpha_hat:.8f}, lambda={lambda_hat:.8f} ({'POS' if lambda_hat>0 else 'NEG'})")

y_pred_kyle = np.sign(alpha_hat + lambda_hat * kyle_te['OI1'].values).astype(int)
acc_kyle = accuracy_score(kyle_te['y'].values, y_pred_kyle)
r2 = r2_score(kyle_te['dpm'].values, ols.predict(kyle_te[['OI1']].values))
print(f"  Direction acc: {acc_kyle*100:.2f}%, R²={r2:.6f}")

# ===== PART 4: ML MODELS =====
print("\n=== PART 4: ML Models ===")

# Prepare clean data
tr_clean = train_df[all_feats + ['y']].dropna()
te_clean = test_df[all_feats + ['y']].dropna()
X_train = tr_clean[all_feats].values
y_train = tr_clean['y'].values
X_test = te_clean[all_feats].values
y_test = te_clean['y'].values

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

# Q4.1 LogReg
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Q4.1 LogReg: {acc_lr*100:.2f}%")

# Q4.2 LightGBM
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid

y_tr_lgb = y_train + 1
val_sp = int(len(X_train) * 0.8)
best_va, best_p = 0, None
for p in ParameterGrid({'n_estimators': [300, 500], 'max_depth': [6, 8], 'learning_rate': [0.01, 0.05, 0.1]}):
    m = lgb.LGBMClassifier(**p, random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_train[:val_sp], y_tr_lgb[:val_sp])
    va = accuracy_score(y_tr_lgb[val_sp:], m.predict(X_train[val_sp:]))
    if va > best_va:
        best_va, best_p = va, p

lgb_model = lgb.LGBMClassifier(**best_p, random_state=42, verbose=-1, n_jobs=-1)
lgb_model.fit(X_train, y_tr_lgb)
y_pred_lgb = lgb_model.predict(X_test) - 1
acc_lgb = accuracy_score(y_test, y_pred_lgb)
print(f"Q4.2 LightGBM: {acc_lgb*100:.2f}% (params: {best_p})")

# Q4.3 MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Use 40 raw LOB + 12 engineered = 52 features for NNs
lob_cols = list(df.columns[:40])
nn_feats = lob_cols + all_feats
tr_nn = train_df[nn_feats + ['y']].dropna()
te_nn = test_df[nn_feats + ['y']].dropna()
sc_nn = StandardScaler()
X_tr_nn = sc_nn.fit_transform(tr_nn[nn_feats].values)
X_te_nn = sc_nn.transform(te_nn[nn_feats].values)
y_tr_nn = tr_nn['y'].values
y_te_nn = te_nn['y'].values

X_tr_t = torch.FloatTensor(X_tr_nn)
y_tr_t = torch.LongTensor(y_tr_nn.astype(int) + 1)
X_te_t = torch.FloatTensor(X_te_nn)

cc = np.bincount(y_tr_nn.astype(int) + 1, minlength=3)
cw = torch.FloatTensor(len(y_tr_nn) / (3 * cc)).to(device)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=2048, shuffle=True)

class MLP(nn.Module):
    def __init__(self, dim, sizes, drop=0.3):
        super().__init__()
        layers = []
        prev = dim
        for h in sizes:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)])
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

mlp = MLP(X_tr_nn.shape[1], [256, 128, 64], 0.3).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
crit = nn.CrossEntropyLoss(weight=cw)

mlp.train()
for ep in range(50):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); crit(mlp(xb), yb).backward(); opt.step()
    sched.step()
    if (ep+1) % 10 == 0:
        print(f"  MLP epoch {ep+1}/50")

mlp.eval()
with torch.no_grad():
    y_pred_mlp = mlp(X_te_t.to(device)).argmax(1).cpu().numpy() - 1
acc_mlp = accuracy_score(y_te_nn, y_pred_mlp)
print(f"Q4.3 MLP: {acc_mlp*100:.2f}%")

# Q4.4 LSTM
SEQ_LEN = 50
nn_clean = df[nn_feats + ['y']].dropna().reset_index(drop=True)
X_full = sc_nn.fit_transform(nn_clean[nn_feats].values)
y_full = nn_clean['y'].values.astype(int) + 1

# Build sequences efficiently
X_seq = np.lib.stride_tricks.sliding_window_view(X_full, (SEQ_LEN, X_full.shape[1])).squeeze(1)
y_seq = y_full[SEQ_LEN:]
if len(X_seq) > len(y_seq):
    X_seq = X_seq[:len(y_seq)]

seq_split = int(len(X_seq) * 0.8)
X_tr_seq = torch.FloatTensor(X_seq[:seq_split])
y_tr_seq = torch.LongTensor(y_seq[:seq_split])
X_te_seq = torch.FloatTensor(X_seq[seq_split:])
y_te_seq = y_seq[seq_split:] - 1

seq_loader = DataLoader(TensorDataset(X_tr_seq, y_tr_seq), batch_size=512, shuffle=True)
print(f"  LSTM sequences: {X_seq.shape}")

class LSTMModel(nn.Module):
    def __init__(self, dim, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, layers, batch_first=True, dropout=drop)
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden), nn.Linear(hidden, 64),
                                nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 3))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm = LSTMModel(X_full.shape[1], 128, 2, 0.3).to(device)
opt_l = torch.optim.Adam(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
sched_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l, T_max=25)
crit_l = nn.CrossEntropyLoss(weight=cw)

lstm.train()
for ep in range(25):
    for xb, yb in seq_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_l.zero_grad()
        loss = crit_l(lstm(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
        opt_l.step()
    sched_l.step()
    if (ep+1) % 5 == 0:
        print(f"  LSTM epoch {ep+1}/25")

lstm.eval()
with torch.no_grad():
    preds = []
    for i in range(0, len(X_te_seq), 512):
        preds.append(lstm(X_te_seq[i:i+512].to(device)).argmax(1).cpu())
    y_pred_lstm = torch.cat(preds).numpy() - 1
acc_lstm = accuracy_score(y_te_seq, y_pred_lstm)
print(f"Q4.4 LSTM: {acc_lstm*100:.2f}%")

# Q4.5 Walk-forward
print("\nQ4.5 Walk-Forward:")
X_all_wf = np.vstack([X_train, X_test])
y_all_wf = np.concatenate([y_train, y_test])
bs = len(X_all_wf) // 6
wf = []
for i in range(5):
    te_s, te_e = (i+1)*bs, min((i+2)*bs, len(X_all_wf))
    m = lgb.LGBMClassifier(**best_p, random_state=42, verbose=-1)
    m.fit(X_all_wf[:(i+1)*bs], y_all_wf[:(i+1)*bs]+1)
    a = accuracy_score(y_all_wf[te_s:te_e], m.predict(X_all_wf[te_s:te_e])-1)
    wf.append(a)
    print(f"  Block {i+1}: {a*100:.2f}%")
print(f"  Mean: {np.mean(wf)*100:.2f}% ± {np.std(wf)*100:.2f}%")

# ===== FINAL SUMMARY =====
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
models = [
    ('sign(pw-pm)', 'analytical', acc_a1),
    ('Kyle OLS', 'analytical', acc_kyle),
    ('LogReg', 'ML (linear)', acc_lr),
    ('LightGBM', 'ML (tree)', acc_lgb),
    ('MLP', 'ML (NN)', acc_mlp),
    ('LSTM', 'ML (recurrent)', acc_lstm),
]
for n, t, a in models:
    print(f"  {n:<20s} {t:<20s} {a*100:.2f}%")
print("="*60)
best = max(models, key=lambda x: x[2])
print(f"Best: {best[0]} ({best[2]*100:.2f}%)")

# Save results for notebook generation
import pickle
pickle.dump({
    'acc_a1': acc_a1, 'acc_kyle': acc_kyle, 'acc_lr': acc_lr,
    'acc_lgb': acc_lgb, 'acc_mlp': acc_mlp, 'acc_lstm': acc_lstm,
    'alpha_hat': alpha_hat, 'lambda_hat': lambda_hat, 'r2': r2,
    'best_p': best_p, 'wf': wf, 'models': models,
}, open('results.pkl', 'wb'))
print("\nResults saved to results.pkl")
print("DONE!")
