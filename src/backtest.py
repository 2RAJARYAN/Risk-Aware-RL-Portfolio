import os
# Fix for the OpenMP Matplotlib/PyTorch crash on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from utils import *
import pandas as pd
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from env import PortfolioEnv

##here we write the test_env same as train.py
#load the data (df_final)
BASE_DIR=Path(__file__).resolve().parent.parent
csv_path=BASE_DIR/"data"/"processed_data"/"df_final.csv"
df_final=pd.read_csv(csv_path,parse_dates=['date'])

df_final = df_final.dropna()
#so we split data dynamically
sort_dates=df_final['date'].sort_values().unique()

train_cutoff=int(len(sort_dates)*0.8)
dev_cutoff=int(len(sort_dates)*0.9)

train_dates=sort_dates[:train_cutoff]
dev_dates=sort_dates[train_cutoff:dev_cutoff]
test_dates=sort_dates[dev_cutoff:]

train_df = df_final[df_final['date'].isin(train_dates)]
dev_df = df_final[df_final['date'].isin(dev_dates)]
test_df = df_final[df_final['date'].isin(test_dates)]
print(f"Train shape: {train_df.shape} | Dev shape: {dev_df.shape} | Test shape: {test_df.shape}")

test_env=DummyVecEnv([lambda:PortfolioEnv(test_df)])


#########################################
# Load the trained model
model = PPO.load('./logs/best_model/best_model.zip')

# 1. Extract environment attributes safely through the VecEnv wrapper
num_days = len(test_env.envs[0].dates)
num_assets = test_env.envs[0].num_assets

print("Running RL Agent Backtest...")
obs = test_env.reset()
portfolio_values = [1.0]

for _ in range(num_days - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)

    # Use [0] to extract the value from the VecEnv array
    new_value = portfolio_values[-1] * (1 + reward[0])
    portfolio_values.append(new_value)

    if done[0]:
        break

print("Running Equal-Weight Baseline...")
obs = test_env.reset()
baseline_values = [1.0]

# VecEnvs expect a "batch" of actions, so we wrap the array in a list
# Baseline logic: divide capital evenly among all 30 stocks
equal_weights = [np.ones(num_assets) / num_assets]

for _ in range(num_days - 1):
    obs, reward, done, info = test_env.step(equal_weights)

    new_value = baseline_values[-1] * (1 + reward[0])
    baseline_values.append(new_value)

    if done[0]:
        break

#save the result
## save metrics to csv
os.makedirs("result/metrics",exist_ok=True)

result_df=pd.DataFrame({
    "rl":portfolio_values,
    "baseline":baseline_values
})
result_df.to_csv("result/metrics/performance.csv",index=False)
print("csv saved succeed")

# Print the final metrics using your utils.py functions
print("---final performance metrics---")
print_metrics(portfolio_values,"Rl agent PPO")
print_metrics(baseline_values,"equal weight baseline")

plot_comparison(portfolio_values,baseline_values)