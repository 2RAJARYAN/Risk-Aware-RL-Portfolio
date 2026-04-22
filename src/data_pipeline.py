import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import PortfolioEnv

#load the data (df_final)
BASE_DIR=Path(__file__).resolve().parent.parent
csv_path=BASE_DIR/"data"/"processed_data"/"df_final.csv"
df_final=pd.read_csv(csv_path,parse_dates=['date'])

#train/dev/test split
'''
train_df=df_final[df_final['date']<"2018-01-01"]
dev_df=df_final[(df_final['date']>="2018-01-01")& (df_final['date']<"2020-01-01")]
test_df=df_final[df_final['date']>="2020-01-01"]
''' #this lead to shape to (0,9)??

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

##environment setup
def make_env():
    return PortfolioEnv(train_df)

env=DummyVecEnv([make_env])  ##ppo require it

## create model
model=PPO("MlpPolicy",
          env,
          learning_rate=3e-4,
          n_steps=2048,
          batch_size=64,
          gamma=0.99,
          verbose=1)

##train
print("Start training")
model.learn(total_timesteps=100_000)

##save model
model.save("ppo_portfolio")
print("Model saved")
##evaluate
print("Evaluting ")
test_env=PortfolioEnv(test_df)

# obs=test_env.reset()
obs_=test_env.reset()   #Gymnasium reset() returns TWO values (obs and info)

portfolio_values=[1]

for _ in range(len(test_env.dates)-1):
    # deterministic=True for testing. We want the agent's best guess, not random exploration!
    action,_=model.predict(obs,deterministic=True)

    obs,reward,terminated,truncated,_=test_env.step(action)

    portfolio_values.append(portfolio_values[-1]*(1+reward))

    if terminated or truncated:
        break

#plot
plt.plot(portfolio_values)
plt.title("Portfolio growth (ppo)")
plt.grid()
plt.show()

