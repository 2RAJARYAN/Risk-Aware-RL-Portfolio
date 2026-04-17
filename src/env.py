# import gym       #gym is discontinue by openai, so we use gymnasium
import gymnasium as gym
import numpy as np 
import pandas as pd
from pathlib import Path


#defining the portfolio env
class PortfolioEnv(gym.Env):
    def __init__(self,df):
        super().__init__()
        
        self.df=df
        self.dates=sorted(df['date'].unique())
        self.tickers=sorted(df['ticker'].unique())

        self.num_assets=len(self.tickers)
        # return, volatility, momentum, sma_ratio , vix_z
        self.num_features=5 

        #action : portfolio weights
        self.action_space=gym.spaces.Box(low=0,
                                         high=1,
                                         shape=(self.num_assets,),
                                         dtype=np.float32
                                        )
        #observation-> flattened state
        self.observation_space=gym.spaces.Box(
            low=-np.inf,high=np.inf,
            shape=(self.num_assets* self.num_features,),
            dtype=np.float32
        )
        self.current_step=None
        self.prev_weights=np.ones(self.num_assets)/self.num_assets

    #reset
    def reset(self):
        self.current_step=0
        self.prev_weights=np.ones(self.num_assets)/self.num_assets
        # return self.__get_state()     #when using gym 
        return self._get_state(),{}               #as we change the gym lib->gymnasium
        #now must return two things: state, info (a dictionary).

    #getting states
    def _get_state(self):
        current_date=self.dates[self.current_step]
        data=self.df[self.df['date']==current_date]

        #ensure correct order of tickes
        data=data.set_index('ticker').loc[self.tickers]

        features=data[['return','volatility','momentum','sma_ratio','vix_z']].values
        return features.flatten().astype(np.float32)

    def step(self,action):
        
        #normalize weights(sum=1)
        weights=action/(np.sum(action)+1e-8)

        current_date=self.dates[self.current_step]
        data=self.df[self.df['date']==current_date]
        data=data.set_index('ticker').loc[self.tickers]


        returns=data['return'].values

        #portfolio return 
        portfolio_return=np.dot(weights,returns)

        #transaction cost (based on change in weights)
        cost=0.001*np.sum(np.abs(weights-self.prev_weights))

        reward=portfolio_return -cost

        #move forward
        self.current_step+=1
        # done=self.current_step>=len(self.dates)-1
        terminated=self.current_step>=len(self.dates)-1
        truncated=False

        self.prev_weights=weights

        next_state=self._get_state()

        # return next_state,reward,done,{}      
        return next_state,float(reward),terminated,truncated,{}      
    # now must return five things: next_state, reward, terminated, truncated, info. 
    # (`done` was split into terminated for naturally finishing, and truncated for hitting a time limit).


#getting you df_final  form the folder
# 1. Dynamically find the folder this script (env.py) is sitting in
# __file__ refers to env.py. parent goes up to 'src'. parent again goes up to 'Risk-Aware-RL-Portfolio'
BASE_DIR=Path(__file__).resolve().parent.parent
# 2. Build the exact path to the CSV by joining the folders
csv_path=BASE_DIR/"data"/"processed_data"/"df_final.csv"
# 3. Load the data
df_final=pd.read_csv(csv_path)


env=PortfolioEnv(df_final)
state=env.reset()

for _ in range(10):
    action=np.random.rand(env.num_assets)
    next_state,reward,terminated,truncated,info=env.step(action)
    print("Reward : ",reward)

    #reset if the episode finishes during the loop  
    if terminated or truncated:
        state, info = env.reset()

    '''
    Concept behind the if statement
    * in rl ,environment is operate in `Episodes`.
    * terminated =true then end naturally (on more trades)
    * truncated =ture  then it stopped artificially (like you set limit on trades)
    * If the agent just hit the final day of the stock market data, wipe the board clean and 
      restart the timeline from the beginning (env.reset()) so the loop can safely take its next step
    * this logic can help ppo in running millions of step without crashing. 
    '''

    

