import os
# Fix for the OpenMP Matplotlib/PyTorch crash on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
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

##environment setup
def make_env():
    return PortfolioEnv(train_df)

env=DummyVecEnv([make_env])  ##ppo require it

#create an enviornment for our dev set 
eval_env=DummyVecEnv([lambda:PortfolioEnv(dev_df)])

##call back and model initialization
#log directory
os.makedirs('./logs/best_model/',exist_ok=True)

eval_callback=EvalCallback(eval_env,
                           best_model_save_path='./logs/best_model/',
                           log_path='./logs/',
                           eval_freq=5000,
                           deterministic=True,
                           render=False)

## create model
model=PPO("MlpPolicy",
          env,
          learning_rate=3e-4,
          n_steps=2048,
          batch_size=64,
          gamma=0.99,
          verbose=1,
          device="cpu"
        )
print("model device use for traning")
print(model.device)

##train
print("Start training with dev set evaluation")
model.learn(total_timesteps=100_000,callback=eval_callback)

print("loading best model for final evaluation..")
best_model=PPO.load('./logs/best_model/best_model.zip')


##evaluate
print("Evaluting on test data")
# test_env=PortfolioEnv(test_df) #need to wrap the test env in dummyvecEnv
test_env=DummyVecEnv([lambda :PortfolioEnv(test_df)])  

# obs=test_env.reset()
obs=test_env.reset()   #Gymnasium reset() returns TWO values (obs and info)

portfolio_values=[1.0]

#get the number of days directly from the test dataframe
num_days=len(test_df['date'].unique())

for _ in range(num_days-1):
    # deterministic=True for testing. We want the agent's best guess, not random exploration!
    action,_=best_model.predict(obs,deterministic=True)
    #vecEnv step() return exactly 4 variable.
    obs,reward,done,info = test_env.step(action)

    #VecEnv return arrays ,we need first item in reward array
    portfolio_values.append(portfolio_values[-1]*(1+reward[0]))

    if done:
        break

#plot
plt.plot(portfolio_values)
plt.title("Portfolio growth (ppo)")
plt.xlabel("Trading Days")
plt.ylabel("Cumulative Value (1.0 = Initial Capital)")
plt.grid()
plt.show()

