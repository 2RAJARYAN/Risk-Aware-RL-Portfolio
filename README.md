# Risk-Aware Reinforcement Learning for Portfolio Optimization

##  Overview

This project builds a **risk-aware portfolio optimization system** using Reinforcement Learning (RL).
The agent learns to allocate capital across multiple stocks by balancing **returns and risk factors** such as volatility and market uncertainty.

We construct a full pipeline:

* Data collection & feature engineering
* Custom trading environment (Gym-style)
* RL training (PPO or other algorithms)
* Backtesting & performance evaluation

---

##  Objectives

* Learn **dynamic portfolio allocation** using RL
* Incorporate **risk-awareness** via features like volatility and VIX
* Compare RL performance with traditional strategies (e.g., equal-weight portfolio)

---

##  Methodology

### 1. Data Pipeline

* Source: Yahoo Finance
* Assets: Dow Jones 30 stocks
* Features:

  * Daily returns
  * Rolling volatility
  * Momentum
  * SMA ratio
  * VIX (market risk indicator)

### 2. State Space

Each timestep consists of:

```
[returns, volatility, momentum, sma_ratio, vix_z] × 30 assets
```

---

### 3. Environment

Custom Gym environment:

* **State** → market features
* **Action** → portfolio weights
* **Reward** → return − transaction cost

---

### 4. RL Algorithm

* Primary: PPO (Proximal Policy Optimization)
* Alternatives: SAC, A2C

---

### 5. Evaluation

* Portfolio growth over time
* Comparison with baseline (equal-weight)
* Metrics:

  * Sharpe Ratio
  * Maximum Drawdown
  * Volatility

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data_pipeline.py
│   ├── env.py
│   ├── train.py
│   └── utils.py
│
├── notebooks/
│   └── eda.ipynb
│
├── results/
│   ├── plots/
│   └── metrics/
│
└── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib yfinance gym stable-baselines3 torch
```

---

## 🚀 Usage

### 1. Run Data Pipeline

```bash
python src/data_pipeline.py
```

### 2. Train RL Agent

```bash
python src/train.py
```

### 3. View Results

* Plots saved in `results/plots/`
* Metrics saved in `results/metrics/`

---

## Sample Results


---

## Research Contributions

* Risk-aware feature design using volatility & VIX
* Custom RL environment for portfolio allocation
* Evaluation under transaction costs

---

## Limitations

* No market impact modeling
* Limited to daily frequency data
* Assumes perfect liquidity

---

## Future Work

* Add CVaR-based risk penalty
* Use alternative RL algorithms (SAC, TD3)
* Incorporate macroeconomic indicators
* Extend to Indian markets (e.g., Nifty 50, Sensex)

---

##  Acknowledgements

* Yahoo Finance API
* Stable-Baselines3
* OpenAI Gym

---

## If you find this useful, consider giving it a star!
