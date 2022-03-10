# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

# cwd = '/Users/chenwynn/Documents/courses_21spring/reinforcement learning/Reinforcement_Learning_Course/Project_1'
cwd = os.getcwd()
pwd = cwd+'/factors/'

# load return
ret_stock = pd.read_csv(pwd+'ret_stock.csv').set_index('date')

# load factors, and saved in a dicionary
factor_list = ['alpha','beta', 'momentum', 'size', 'earnyild', 'resvol', 'growth', 'bp', 'leverage', 'liquidty']
factors = {}
for factor in factor_list:
    factors[factor] = pd.read_csv(pwd+factor+'.csv').set_index('date')

# calculate returns of each factor portfolio

return_factors = {}
for factor in factor_list:
    tmp = factors[factor].stack().groupby('date').nlargest(19)
    tmp.loc[:,:] = 1
    tmp = tmp.unstack()
    tmp.index = ret_stock.index
    tmp2 = factors[factor].stack().groupby('date').nsmallest(19)
    tmp2.loc[:,:] = -1
    tmp2 = tmp2.unstack()
    tmp2.index = ret_stock.index
    return_factors[factor] = ((tmp.shift()*ret_stock).mean(axis=1)+(tmp2.shift()*ret_stock).mean(axis=1))/2

reward = pd.DataFrame(return_factors)
reward.dropna(inplace=True)

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

ret_index = pd.read_csv(pwd+'ret_index.csv').set_index('date')
ret_index = ret_index.rename(columns = {'close':'CSI 300'})
ret_index.index = pd.to_datetime(ret_index.index)

#os.makedirs(cwd+'/figures/')

reward.index = pd.to_datetime(reward.index)
plt.figure(figsize=(20,10))
sns.lineplot(data=(reward.join(ret_index)+1).cumprod())
plt.savefig(cwd+'/factors_figures/'+'factors.jpg')

k=10
# idx=[]

step_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
simulation = 100


epsilon_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]

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
        # lb="step = "+str(step)+", "+"epi = "+str(epsilon)
        # idx.append(lb)
        s.append(step)
        e.append(epsilon)
        test_mean.append(np.array(temp).mean())
        test_std.append(np.array(temp).std())
        test_mean_UCB.append(np.array(temp2).mean())
        test_std_UCB.append(np.array(temp2).std())
        

#@ the best parameters are stepsize = 0.3, eps = 0.02, UCB_param = 2
#@ draw figures of the same bandit
result = {}
actions = {}
i = 0
while i < 10:
    temp = train(10, reward*100, 0.3, 0.02)
    result[i] = pd.Series(temp['return'])/100
    actions[i] = pd.Series(temp['actions'])
    i = i+1
result = pd.DataFrame(result)
actions = pd.DataFrame(actions)

plt.figure(figsize=(20,10))
sns.lineplot(data=(result+1).cumprod())
plt.savefig(cwd+'/factors_figures/'+'returns.jpg')

plt.figure(figsize=(20,10))
sns.histplot(data = actions.unstack().reset_index(), x = 'level_0', hue = 0, multiple = 'stack')
plt.savefig(cwd+'/factors_figures/'+'actions.jpg')




    













