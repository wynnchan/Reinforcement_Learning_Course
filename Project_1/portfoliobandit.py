# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 21:51:24 2022

@author: 123
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



cwd = os.getcwd()
pwd = cwd+'/portfolio/'

reward = {}

ret_stock=pd.read_csv(pwd+'ret_stock.csv').set_index('date')
markov_p=pd.read_csv(pwd+'markov_p.csv').set_index('date')

ret_stock = ret_stock.reindex(markov_p.index)
capital=pd.read_csv(cwd+'/data/market_cap_3.csv').set_index('date')
# def Cap_Weighted(capital):
#     return capital.div(capital.sum(), axis=1)

# reward['equal'] = ret_stock.mean(axis=1)
# reward['markov'] = (ret_stock*markov_p).sum(axis=1)
# reward['capital_w'] = (ret_stock*(Cap_Weighted(capital).shift()).reindex(ret_stock.index)).sum(axis=1)

# reward = pd.DataFrame(reward)

#reward.to_csv('reward.csv')

reward=pd.read_csv(pwd+'reward.csv').set_index('date').dropna()

ret_index = pd.read_csv(cwd+'/data/ret_index.csv').set_index('date')
ret_index = ret_index.rename(columns = {'close':'CSI 300'})
ret_index = ret_index.reindex(ret_stock.index)
ret_index.index = pd.to_datetime(ret_index.index)

#os.makedirs(cwd+'/portfolio_figures/')

reward.index = pd.to_datetime(reward.index)
plt.figure(figsize=(20,10))
sns.lineplot(data=(reward.join(ret_index)+1).cumprod())
plt.savefig('portfolio.jpg')


class Bandit:
    # @karms: # of arms
    # @epi: probability for exploration in epi-greedy algorithm
    # @initial: initial estimation for each action
    # @stepsize: constant step size for updating estimations
    # @sample_avr: if True, use sample s to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @grad: if True, use grad based bandit algorithm
    # @grad_baseline: if True, use  reward as baseline for grad based bandit algorithm
    # @true_reward: pandas series of k arms
    def __init__(self, karms, epi=0., initial=0., stepsize=0.1, sample_avr=False, UCB_param=None,
                 grad=False, grad_baseline=False):
        self.k = karms
        self.stepsize = stepsize
        self.sample_avr = sample_avr
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.grad = grad
        self.grad_baseline = grad_baseline
        self._reward = 0
        self.epi = epi
        self.initial = initial

    def reset(self):

        # estimation for each action
        self.q_esti = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.act_count = np.zeros(self.k)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epi:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_esti + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.act_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.grad:
            exp_est = np.exp(self.q_esti)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_esti)
        return np.random.choice(np.where(self.q_esti == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action, true_reward):
        # use true reward to update
        reward = true_reward.iloc[action]
        self.time += 1
        self.act_count[action] += 1
        self._reward += (reward - self._reward) / self.time

        if self.sample_avr:
            # update estimation using sample
            self.q_esti[action] += (reward - self.q_esti[action]) / self.act_count[action]
        elif self.grad:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.grad_baseline:
                baseline = self._reward
            else:
                baseline = 0
            self.q_esti += self.stepsize * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_esti[action] += self.stepsize * (reward - self.q_esti[action])
        return reward
def train(karm, reward, eps, stepsize, UCB_param=None, grad=False, grad_baseline=False):
    # @k_arm: int
    # @reward: pandas.dataframe
    res = []
    actions = []
    test = Bandit(karm, epi = eps, initial = 0, stepsize = stepsize, UCB_param=None, grad=False, grad_baseline=False)
    test.reset()
    for t in range(len(reward)):
        action = test.act()
        re = test.step(action, reward.iloc[t,:])
        actions.append(action)
        res.append(re)
    return {'return':res, 'actions':actions}



k=3
# idx=[]

step_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
simulation = 100


epsilon_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
simulation = 100

test_mean = []
test_std = []
test_mean_UCB = []
test_std_UCB = []
s=[]
e=[]

for step in step_range:
    for epsilon in epsilon_range:
        i = 0
        temp = []
        temp2 = []
        while i < 100:
            i = i+1
            temp.append((np.array(train(k, reward*100, epsilon, step)['return'])/100+1).prod())
            temp2.append((np.array(train(k, reward*100, epsilon, step, UCB_param=2)['return'])/100+1).prod())

        s.append(step)
        e.append(epsilon)
        test_mean.append(np.array(temp).mean())
        test_std.append(np.array(temp).std())
        test_mean_UCB.append(np.array(temp2).mean())
        test_std_UCB.append(np.array(temp2).std())
        

df1=pd.DataFrame()
df1['step']=s
df1['epsilon']=e
df1['mean']=test_mean   
df1['std']=test_std
df1["Coe_of_vol"]=np.array(test_mean)/np.array(test_std)

l1=np.array(test_mean)/np.array(test_std)

df2=pd.DataFrame()
df2['step']=s
df2['epsilon']=e
df2['mean']=test_mean_UCB
df2['std']=test_std_UCB
df2['Coe_of_vol']=np.array(test_mean_UCB)/np.array(test_std_UCB)

l2=np.array(test_mean_UCB)/np.array(test_std_UCB)

if (max(l1)>max(l2)):
    idx=np.argmax(l1)
    flag=1
    step_=s[idx]
    epsilon_=e[idx]
else:
    idx=np.argmax(l2)
    flag=2
    step_=s[idx]
    epsilon_=e[idx]
    
result = {}
actions = {}
i = 0
if flag==1:
    while i < 10:
        temp = train(k, reward*100, epsilon_, step_)
        result[i] = pd.Series(temp['return'])/100
        actions[i] = pd.Series(temp['actions'])
        i = i+1
else:
    while i < 10:
        temp = train(k, reward*100, epsilon_, step_, UCB_param=2)
        result[i] = pd.Series(temp['return'])/100
        actions[i] = pd.Series(temp['actions'])
        i = i+1
result = pd.DataFrame(result)
actions = pd.DataFrame(actions)

plt.figure(figsize=(20,10))
sns.lineplot(data=(result+1).cumprod())
plt.savefig(cwd+'/portfolio_figure/'+'returns.jpg')

plt.figure(figsize=(20,10))
sns.histplot(data = actions.unstack().reset_index(), x = 'level_0', hue = 0, multiple = 'stack')
plt.savefig(cwd+'/portfolio_figure/'+'actions.jpg')

























