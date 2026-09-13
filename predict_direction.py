import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

TICKER = "SPY"
START = "2015-01-01"
END = "2025-12-31"
TEST_FRAC = 0.2


def load_data():
    try:
        import yfinance as yf

        df = yf.download(
            TICKER, start=START, end=END, auto_adjust=False, progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) > 500:
            print(
                f"Downloaded {TICKER}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()}"
            )
            return df
        raise ValueError("download too short")
    except Exception as e:
        print(
            f"yfinance failed ({e}), using synthetic OHLCV fallback (random-walk, same pipeline)."
        )
        rng = np.random.default_rng(42)
        idx = pd.bdate_range(START, END)
        ret = rng.normal(0.0004, 0.01, len(idx))
        close = 200 * np.exp(np.cumsum(ret))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, len(idx))))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, len(idx))))
        vol = rng.integers(50_000_000, 120_000_000, len(idx))
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=idx,
        )


df = load_data()
df["Close_next"] = df["Close"].shift(-1)
df["Target"] = (df["Close_next"] > df["Close"]).astype(int)
df["Today_dir"] = (df["Close"] > df["Close"].shift(1)).astype(int)
close = df["Close"]
delta = close.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["RSI_14"] = 100 - (100 / (1 + rs))
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
sma20 = close.rolling(20).mean()
std20 = close.rolling(20).std()
df["SMA20_dist"] = close / sma20 - 1
df["PctB_20"] = (close - (sma20 - 2 * std20)) / (
    (sma20 + 2 * std20) - (sma20 - 2 * std20)
)
df["Ret_1"] = close.pct_change(1)
df["Ret_5"] = close.pct_change(5)
df["Vol_20"] = df["Ret_1"].rolling(20).std()
df["HL_range"] = (df["High"] - df["Low"]) / df["Close"]
df["OC_range"] = (df["Close"] - df["Open"]) / df["Open"]
df["Vol_chg"] = df["Volume"].pct_change(1)
df = df.dropna().copy()
print(f"Rows after indicators/warmup: {len(df)}")
print("\n--- CLASS BALANCE (1=up next day, 0=down) ---")
print(df["Target"].value_counts(normalize=True).round(4).to_string())
print(df["Target"].value_counts().to_string())
RAW = ["Close", "Open", "High", "Low", "Volume"]
for lag in range(1, 6):
    df[f"Close_lag{lag}"] = close.shift(lag)
    df[f"Ret_lag{lag}"] = df["Ret_1"].shift(lag - 1)
df = df.dropna()
RAW_FEATS = [f"Close_lag{i}" for i in range(1, 6)] + ["Volume"]
ENG_FEATS = [
    "Ret_1",
    "Ret_5",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA20_dist",
    "PctB_20",
    "Vol_20",
    "HL_range",
    "OC_range",
    "Vol_chg",
]
n = len(df)
cut = int(n * (1 - TEST_FRAC))
train, test = df.iloc[:cut], df.iloc[cut:]
print(
    f"\nTime split: train {train.index[0].date()}->{train.index[-1].date()} ({len(train)}), test {test.index[0].date()}->{test.index[-1].date()} ({len(test)})"
)
print("Train balance:", train["Target"].value_counts(normalize=True).round(4).to_dict())
print("Test  balance:", test["Target"].value_counts(normalize=True).round(4).to_dict())
scaler = StandardScaler()
Xtr_eng = scaler.fit_transform(train[ENG_FEATS])
Xte_eng = scaler.transform(test[ENG_FEATS])
ytr, yte = train["Target"].values, test["Target"].values
scaler_raw = StandardScaler()
Xtr_raw = scaler_raw.fit_transform(train[RAW_FEATS])
Xte_raw = scaler_raw.transform(test[RAW_FEATS])


def scores(y_true, y_pred, name):
    return {
        "model": name,
        "acc": round(accuracy_score(y_true, y_pred), 4),
        "prec": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "rec": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


results = []
maj = int(train["Target"].mode()[0])
results.append(
    scores(yte, np.full_like(yte, maj), f"Baseline: always-{maj} (train majority)")
)
results.append(
    scores(yte, test["Today_dir"].values, "Baseline: persistence (predict today dir)")
)
lr_raw = LogisticRegression(max_iter=1000)
lr_raw.fit(Xtr_raw, ytr)
results.append(scores(yte, lr_raw.predict(Xte_raw), "LogReg on RAW prices"))
lr_eng = LogisticRegression(max_iter=1000)
lr_eng.fit(Xtr_eng, ytr)
results.append(scores(yte, lr_eng.predict(Xte_eng), "LogReg on ENGINEERED"))
rf_eng = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_eng.fit(Xtr_eng, ytr)
rf_pred = rf_eng.predict(Xte_eng)
results.append(scores(yte, rf_pred, "RandomForest on ENGINEERED"))
res = pd.DataFrame(results).set_index("model")
print("\n--- TEST COMPARISON ---")
print(res.to_string())
print("\nConfusion (RF engineered):\n", confusion_matrix(yte, rf_pred))
res.to_csv("results_comparison.csv")
print("Saved results_comparison.csv")
best = res["acc"].idxmax()
print(
    f"\nInterpretation: best test acc = {res['acc'].max():.4f} ({best}). Majority baseline = {res.iloc[0]['acc']:.4f}. If ML barely beats baselines, direction is near-unpredictable from price history alone."
)
plt.figure(figsize=(12, 4))
t = test.index
plt.plot(t, test["Close"], color="black", lw=1, label="Close (test)")
up_mask = rf_pred == 1
dn_mask = ~up_mask
plt.scatter(t[up_mask], test["Close"][up_mask], s=8, label="Pred UP (RF)", marker="^")
plt.scatter(t[dn_mask], test["Close"][dn_mask], s=8, label="Pred DOWN (RF)", marker="v")
plt.title(
    f"{TICKER} test: predicted vs actual next-day direction (RF engineered, acc={res.loc['RandomForest on ENGINEERED', 'acc']:.3f})"
)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=120)
print("Saved pred_vs_actual.png")
plt.figure(figsize=(12, 3))
s = slice(0, 120)
plt.step(range(120), yte[s], where="mid", label="Actual (1=up)")
plt.step(range(120), rf_pred[s], where="mid", label="Predicted RF", linestyle="--")
plt.yticks([0, 1], ["DOWN", "UP"])
plt.title("Actual vs predicted direction - first 120 test days")
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_actual_steps.png", dpi=120)
print("Saved pred_vs_actual_steps.png")
