import numpy as np
import matplotlib.pyplot as plt

def compute_return(portfolio_values):
    return np.diff(portfolio_values)/portfolio_values[:-1]

def sharpe_ratio(returns):
    return np.mean(returns)/(np.std(returns)+1e-8)

def max_drawdown(profolio_values):
    values=np.array(profolio_values)
    peak=np.maximum.accumulate(values)
    drawdown=(values-peak)/peak
    return drawdown.min()

def volatility(returns):
    return np.std(returns)

##plot
def plot_profolio(portfolio_values,title="protfolio growth"):
    plt.figure(figsize=(10,5))
    plt.plot(portfolio_values,label="portfolio")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("portfolio value")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_comparison(rl_values,baseline_values):
    plt.figure(figsize=(10,5))
    plt.plot(rl_values,label="portfolio")
    plt.title("RL vs Baseline")
    plt.xlabel("time")
    plt.ylabel("portfolio value")
    plt.grid(True)
    plt.legend()
    plt.show()


def print_metrics(portfolio_values,name="Model"):
    returns=compute_return(portfolio_values)
    print(f"\n-----{name} performance-----")
    print(f"shape ratio:{sharpe_ratio(returns):.4f}")
    print(f"volatitlity: {volatility(returns):.4f}")
    print(f"max drawdown: {max_drawdown(portfolio_values):.4f}")