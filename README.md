# bitcoin-sentiment-trader-performance
“Analyzing Bitcoin sentiment vs Hyperliquid trader performance with data cleaning, EDA, and ML.”


# ==========================================
# Bitcoin Sentiment vs Hyperliquid Trader Performance
# End-to-End Project Script (Google Colab)
# ==========================================

# Step 0: Install dependencies
!pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost shap gdown

# ------------------------------
# Step 1: Import libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For statistics & ML
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------
# Step 2: Download datasets from Google Drive
# ------------------------------
# Replace with your file IDs if different
!gdown --id 1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs -O historical_data.csv
!gdown --id 1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf -O fear_greed.csv

trades = pd.read_csv("historical_data.csv", low_memory=False)
sent = pd.read_csv("fear_greed.csv", low_memory=False)

print("Trades columns:", trades.columns.tolist())
print("Sentiment columns:", sent.columns.tolist())

# ------------------------------
# Step 3: Data Cleaning
# ------------------------------
# Standardize column names
trades.columns = [c.strip().lower().replace(" ", "_") for c in trades.columns]
sent.columns = [c.strip().lower().replace(" ", "_") for c in sent.columns]

# --- Handle datetime in trades ---
if "time" in trades.columns:
    trades["time"] = pd.to_datetime(trades["time"], errors="coerce", utc=True)
    trades["date"] = trades["time"].dt.date
elif "timestamp" in trades.columns:
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce", utc=True)
    trades["date"] = trades["timestamp"].dt.date
elif "date" in trades.columns:
    trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.date
else:
    trades["date"] = pd.date_range(start="2020-01-01", periods=len(trades)).date

# --- Handle datetime in sentiment ---
if "date" in sent.columns:
    sent["date"] = pd.to_datetime(sent["date"], errors="coerce").dt.date

# Numeric conversions (only if column exists)
for col in ["execution_price", "size", "closedpnl", "leverage"]:
    if col in trades.columns:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

# Clean side column if available
if "side" in trades.columns:
    trades["side"] = trades["side"].astype(str).str.upper().str.strip()
    trades["side"] = trades["side"].replace({"LONG": "BUY", "SHORT": "SELL"})
else:
    trades["side"] = "BUY"  # default placeholder

# Drop invalid rows
if "size" in trades.columns and "execution_price" in trades.columns:
    trades = trades[(trades["size"] != 0) & trades["execution_price"].notna()]

# Derived features
if "size" in trades.columns and "execution_price" in trades.columns:
    sign_map = {"BUY": 1, "SELL": -1}
    trades["signed_size"] = trades["size"] * trades["side"].map(sign_map)
    trades["notional"] = trades["size"] * trades["execution_price"]

# ------------------------------
# Step 4: Merge with sentiment
# ------------------------------
if "classification" in sent.columns:
    sent = sent.rename(columns={"classification": "sent_label"})

if "sent_label" in sent.columns:
    trades = trades.merge(sent[["date", "sent_label"]], on="date", how="left")
    trades["is_fearful"] = trades["sent_label"].str.contains("Fear", case=False, na=False)
    trades["is_greedy"] = trades["sent_label"].str.contains("Greed", case=False, na=False)
else:
    trades["sent_label"] = "Neutral"
    trades["is_fearful"] = False
    trades["is_greedy"] = False

print(trades.head())

# ------------------------------
# Step 5: Exploratory Data Analysis
# ------------------------------
# Daily PnL aggregate
if "closedpnl" in trades.columns:
    daily = trades.groupby("date").agg(daily_pnl=("closedpnl", "sum"))

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily["daily_pnl"])
    plt.title("Daily PnL Over Time")
    plt.show()

    # Boxplot by sentiment
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="sent_label", y="closedpnl", data=trades)
    plt.yscale("symlog")
    plt.title("PnL by Market Sentiment")
    plt.show()

# ------------------------------
# Step 6: Statistical Tests
# ------------------------------
if "closedpnl" in trades.columns:
    fear_pnl = trades.loc[trades["is_fearful"], "closedpnl"].dropna()
    greed_pnl = trades.loc[trades["is_greedy"], "closedpnl"].dropna()

    if len(fear_pnl) > 0 and len(greed_pnl) > 0:
        print("t-test:", ttest_ind(fear_pnl, greed_pnl, equal_var=False))
        print("Mann-Whitney U:", mannwhitneyu(fear_pnl, greed_pnl))

# ------------------------------
# Step 7: Regression Analysis
# ------------------------------
if "closedpnl" in trades.columns:
    try:
        formula = "closedpnl ~ C(sent_label) + leverage + notional + C(side)"
        model = smf.ols(formula, data=trades).fit()
        print(model.summary())
    except Exception as e:
        print("Regression error:", e)

# ------------------------------
# Step 8: Clustering Traders
# ------------------------------
if "account" in trades.columns and "closedpnl" in trades.columns:
    acct = trades.groupby("account").agg(
        total_trades=("closedpnl", "count"),
        wins=("closedpnl", lambda x: (x > 0).sum()),
        avg_pnl=("closedpnl", "mean"),
        avg_leverage=("leverage", "mean") if "leverage" in trades.columns else ("closedpnl", "mean")
    )
    acct["win_rate"] = acct["wins"] / acct["total_trades"]

    X = acct[["win_rate", "avg_pnl", "avg_leverage", "total_trades"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    acct["cluster"] = km.fit_predict(X_scaled)
    print(acct.groupby("cluster").mean())

# ------------------------------
# Step 9: Predictive Modeling
# ------------------------------
if "closedpnl" in trades.columns:
    X = trades[["leverage", "notional"]].fillna(0)
    y = (trades["closedpnl"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

# ------------------------------
# Step 10: Simple Rule Backtest
# ------------------------------
def simulate_rule(df, leverage_cutoff=5):
    if "closedpnl" not in df.columns:
        return None
    base_pnl = df["closedpnl"].sum()
    if "leverage" in df.columns:
        filtered = df[~((df["is_fearful"]) & (df["leverage"] > leverage_cutoff))]
    else:
        filtered = df
    rule_pnl = filtered["closedpnl"].sum()
    return base_pnl, rule_pnl, rule_pnl - base_pnl

print("Rule backtest:", simulate_rule(trades, leverage_cutoff=5))

# ------------------------------
# Step 11: Save Cleaned Data
# ------------------------------
trades.to_csv("trades_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as trades_cleaned.csv")

