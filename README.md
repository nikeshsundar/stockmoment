[README.md](https://github.com/user-attachments/files/32156236/README.md)
# Next-Day Price Direction Prediction

Predicts next-day direction (up/down) for SPY from daily OHLCV plus hand-built indicators in pure pandas. No pandas-ta.

## Run
```
/c/Users/nikesh/AppData/Local/Programs/Python/Python310/python.exe predict_direction.py
```
If missing libs:
```
/c/Users/nikesh/AppData/Local/Programs/Python/Python310/python.exe -m pip install yfinance scikit-learn matplotlib pandas
```
Do not use default `python` (hermes venv lacks deps).

## Outputs
- `results_comparison.csv` — test metrics table
- `pred_vs_actual.png` — test Close with Pred UP/DOWN markers
- `pred_vs_actual_steps.png` — actual vs predicted 0/1, first 120 test days

## Method
- Data: SPY 2015-2025 via yfinance, synthetic random-walk fallback
- Label: `df["Close_next"] = df["Close"].shift(-1)` then `Target = (Close_next > Close)`. Features use t or earlier only.
- Indicators by hand: RSI-14 (Wilder ewm), MACD 12/26/9, SMA20 distance, Bollinger %B 20/2, Ret 1/5, Vol 20, HL/OC ranges, Vol change
- Split: chronological 80/20. `StandardScaler` fit on train only.
- Compared: Baseline always-train-majority, Baseline persistence (today dir), LogReg on RAW lags, LogReg on ENGINEERED, RandomForest on ENGINEERED

## Results (real SPY)
Class balance 54.7% up. Test: majority 0.589, persistence 0.522, raw LogReg 0.542, engineered LogReg 0.579, engineered RF 0.506. Engineered beats raw/persistence but not majority. Direction is near-unpredictable from price history alone.
