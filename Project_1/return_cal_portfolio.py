# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pandas as pd
import numpy as np
import scipy.optimize as sco
cwd = os.getcwd()
pwd = cwd


# load return
ret_stock = pd.read_csv(pwd+'/portfolio/ret_stock.csv').set_index('date')
capital = pd.read_csv(pwd+'/data/market_cap_3.csv').set_index('date')

def sharpe(w, mean, var):
    #@ w: weights
    #@ mean: array
    #@ var: 2d array
    #@ output: sharpe_ratio
    return -np.sum((w*mean))/np.sqrt(np.dot(w.T,np.dot(var,w)))

def Markov(ret_stock, days = 252):
    #@ output: pandas dataframe markovwitz portfolio weights of each stock
    mean = ret_stock.rolling(days).mean().dropna()
    var = ret_stock.rolling(days).cov().dropna()
    bnds = tuple((0,1) for x in range (ret_stock.shape[1]))
    cons=({'type':'eq','fun':lambda x : np.sum(x)-1})
    wei = []
    for i in list(mean.index):
        def min_func(w):
            return sharpe(w, np.array(mean.loc[i]), np.array(var.loc[i]))
        opts=sco.minimize(min_func, ret_stock.shape[1]*[1./ret_stock.shape[1],], \
                          method='SLSQP',bounds=bnds,constraints=cons)
        wei.append(opts['x'])
    return pd.DataFrame(wei, index = mean.index, columns = mean.columns)

def Cap_Weighted(capital):
    return capital.div(capital.sum(), axis=1)

markov_p = Markov(ret_stock).shift().dropna()
ret_stock = ret_stock.reindex(markov_p.index)
markov_p.to_csv(pwd+'/portfolio/markov_p.csv')
ret_stock .to_csv(pwd+'/portfolio/ret_stock.csv')

#@ calculate everyday return of each portfolio
reward = {}
reward['equal'] = ret_stock.mean(axis=1)
reward['markov'] = (ret_stock*markov_p).sum(axis=1)
reward['capital_w'] = (ret_stock*(Cap_Weighted(capital).shift()).reindex(ret_stock.index)).sum(axis=1)

reward = pd.DataFrame(reward)
reward.to_csv(pwd+'/portfolio/reward.csv')
# '''
# def re(stockp,days):
#     def Cap_Weighted(stockp,days=30):
#         days=30
#         cap_temp=capital.loc[stockp.index,stockp.columns]
#         ret_temp=ret.loc[stockp.index,stockp.columns]
        
#         cap_t=np.array(cap_temp.iloc[30,:])
#         cap_t[np.isnan(cap_t)]=0
#         # cap_t.dropna(fillna=0)
#         wi1=cap_t/sum(cap_t)
            
#         final_ret=sum(wi1*ret_temp.iloc[30,:])
#         wei=np.array(wi1)
#         return wei,final_ret
    
#         ### stockp 是 选取的股票的dataframe index为日期
#         ##capital是total_assets那个total assets 的dataframe
#     def Markov(stockp,days=30):
#         ret_temp=ret.loc[stockp.index,stockp.columns]
        
#         ret1=ret_temp.iloc[0:30,:]
#         ret1.fillna(0)
#         wei=stockp.shape[1]*[1./stockp.shape[1],]
#         def cal(wei):
#             wei=np.array(wei)
#             pre=np.sum(ret1.mean()*wei)
#             pvol=np.sqrt(np.dot(wei.T,np.dot(ret1.cov(),wei)))
#             return np.array([pre,pvol,pre/pvol])
#         def min_func(wei):
#             return -cal(wei)[2]
#         bnds = tuple((0,1) for x in range (stockp.shape[1]))
#         cons=({'type':'eq','fun':lambda x : np.sum(x)-1})
#         opts=sco.minimize(min_func,stockp.shape[1]*[1./stockp.shape[1],]\
#                           ,method='SLSQP',bounds=bnds,constraints=cons)
#         wei=opts['x']
#         final_ret=sum(ret_temp.iloc[30,:]*wei)
#         return wei,final_ret
    
#     def Equal_W(stockp,days=30):
#         ret_temp=ret.loc[stockp.index,stockp.columns]  
#         # ret1=ret_temp.iloc[0:30,:]-1
#         s_temp=stockp.iloc[30,:]
#         s_temp=np.array(s_temp)
#         wei=np.zeros(s_temp.shape)
#         j=0
#         for i in range(s_temp.shape[0]):
            
#             if np.isnan(s_temp[i])==False:
#                 wei[i]=1
#                 j=j+1
#             else:
#                 wei[i]=0
#         if j!=0:
#             wei=wei*(1/j)
#             final_ret=sum(ret_temp.iloc[30,:]*wei)
#         else:
#             final_ret=0
#         return wei,final_ret
    
#     CapR=Cap_Weighted(stockp,days=30)[1]
#     MarkR=Markov(stockp,days=30)[1]
#     EqR=Equal_W(stockp,days=30)[1]
#     return CapR,MarkR,EqR


# m=ret.shape[0]
# Return_df=pd.DataFrame(columns=['Cap','Mark','EqR'])
# days=30
# for r in range(0,4):
#     stockp=ret.iloc[r:r+days+1]
#     idx=stockp.index[-1]
#     Return_df.loc[idx,:]=re(stockp,days)
# Return_df.to_csv('test.csv')
# '''
# class Bandit:
#     # @k_arm: # of arms
#     # @epsilon: probability for exploration in epsilon-greedy algorithm
#     # @initial: initial estimation for each action
#     # @step_size: constant step size for updating estimations
#     # @sample_averages: if True, use sample averages to update estimations instead of constant step size
#     # @UCB_param: if not None, use UCB algorithm to select action
#     # @gradient: if True, use gradient based bandit algorithm
#     # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
#     # @true_reward: pandas series of k arms
#     def __init__(self, k_arm, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
#                  gradient=False, gradient_baseline=False):
#         self.k = k_arm
#         self.step_size = step_size
#         self.sample_averages = sample_averages
#         self.indices = np.arange(self.k)
#         self.time = 0
#         self.UCB_param = UCB_param
#         self.gradient = gradient
#         self.gradient_baseline = gradient_baseline
#         self.average_reward = 0
#         self.epsilon = epsilon
#         self.initial = initial

#     def reset(self):

#         # estimation for each action
#         self.q_estimation = np.zeros(self.k) + self.initial

#         # # of chosen times for each action
#         self.action_count = np.zeros(self.k)

#         self.time = 0

#     # get an action for this bandit
#     def act(self):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.indices)

#         if self.UCB_param is not None:
#             UCB_estimation = self.q_estimation + \
#                 self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
#             q_best = np.max(UCB_estimation)
#             return np.random.choice(np.where(UCB_estimation == q_best)[0])

#         if self.gradient:
#             exp_est = np.exp(self.q_estimation)
#             self.action_prob = exp_est / np.sum(exp_est)
#             return np.random.choice(self.indices, p=self.action_prob)

#         q_best = np.max(self.q_estimation)
#         return np.random.choice(np.where(self.q_estimation == q_best)[0])

#     # take an action, update estimation for this action
#     def step(self, action, true_reward):
#         # use true reward to update
#         reward = true_reward.iloc[action]
#         self.time += 1
#         self.action_count[action] += 1
#         self.average_reward += (reward - self.average_reward) / self.time

#         if self.sample_averages:
#             # update estimation using sample averages
#             self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
#         elif self.gradient:
#             one_hot = np.zeros(self.k)
#             one_hot[action] = 1
#             if self.gradient_baseline:
#                 baseline = self.average_reward
#             else:
#                 baseline = 0
#             self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
#         else:
#             # update estimation with constant step size
#             self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
#         return reward


# def train(k_arm, reward, epsilon, step_size):
#     # @k_arm: int
#     # @reward: pandas.dataframe
#     res = []
#     actions = []
#     test = Bandit(k_arm, epsilon, step_size)
#     test.reset()
#     for t in range(len(reward)):
#         action = test.act()
#         re = test.step(action, reward.iloc[t,:])*100
#         actions.append(action)
#         res.append(re/100)
#     return {'return':res, 'actions':actions}

# result = train(3, reward, 0.01, 0.3)    
            


