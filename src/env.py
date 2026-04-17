# import gym       #gym is discontinue by openai, so we use gymnasium
import gymnasium as gym
import numpy as np 
import pandas as pd
# from data import processed_data 
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

        return next_state,float(reward),terminated,truncated,{}
    # now must return five things: next_state, reward, terminated, truncated, info. 
    # (done was split into terminated for naturally finishing, and truncated for hitting a time limit).


# df_final=pd.read_csv('../data/processed_data/df_final.csv')
# df_final=pd.read_csv('../df_final.csv')
df_final = pd.read_csv("C://Users//Raj Aryan//Documents//amdpro//MinorS//data//processed_data//df_final.csv")

env=PortfolioEnv(df_final)
state=env.reset()

for _ in range(10):
    action=np.random.rand(env.num_assets)
    next_state,reward,done,_=env.step(action)
    print("Reward : ",reward)

