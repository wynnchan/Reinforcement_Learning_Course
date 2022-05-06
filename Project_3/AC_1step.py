#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:46:37 2022

@author: chenwynn
"""


import numpy as np
from math import factorial

class Actor():
  def __init__(self):
    # first layer
    self.w1 = 0.1
    self.w2 = 0.1
    self.b1 = 0.1
    self.w3 = 0.1
    self.w4 = 0.1
    self.b2 = 0.1
    # second layer
    self.w5 = 0.1
    self.w6 = 0.1
    self.b3 = 0.1

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
    h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
    o = self.w5 * h1 + self.w6 * h2 + self.b3
    return o

  def train(self, x, sigma, learn_rate):

    # --- Do a feedforward (we'll need these values later)
    h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1

    h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2

    d_w1 = self.w5 * x[0]
    d_w2 = self.w5 * x[1]
    d_w3 = self.w6 * x[0]
    d_w4 = self.w6 * x[1]
    d_w5 = h1
    d_w6 = h2
    
    d_b1 = self.w5
    d_b2 = self.w6
    d_b3 = 1
    

        # --- Update weights and biases
        # Neuron h1
    self.w1 += learn_rate * sigma * d_w1
    self.w2 += learn_rate * sigma * d_w2
    self.b1 += learn_rate * sigma * d_b1

        # Neuron h2
    self.w3 += learn_rate * sigma * d_w3
    self.w4 += learn_rate * sigma * d_w4
    self.b2 += learn_rate * sigma * d_b2

        # Neuron o1
    self.w5 += learn_rate * sigma * d_w5
    self.w6 += learn_rate * sigma * d_w6
    self.b3 += learn_rate * sigma * d_b3


def bitree(father):
    if np.random.uniform() < 0.5:
        return father*1.05
    else:
        return father/1.05
    
def OptionValue(state, strike_price):
    price = state[0]
    T = int(state[1])
    if T == 0:
        return max(0, price-strike_price)
    else:
        C = 0
        for i in range(T+1):
            C += factorial(T)/(factorial(i)*factorial(T-i))* (1/2)**T * max(0, 1.05**i*((1/1.05)**(T-i))*price-strike_price)
        return C

T = 500000
alpha = 1e-8
beta = 1e-8
gamma = 1
epsilon = 0.05
strike_price = 100
actor = Actor()
critic = Actor()
rewards = []
actions = []
reward = -20
for i in range(T):
    S0 = np.array([100, 2.5], dtype=np.float32)
    a = actor.feedforward(S0)
    S1 = np.array([0,0], dtype=np.float32)
    S1[0] = bitree(S0[0])
    S1[1] = max(S1[0]-100,0)
    R = -np.abs((S1[0]-S0[0])*a - (S1[1]-2.5))
    V = critic.feedforward(S0)
    V1 = critic.feedforward(S1)
    sigma = R+V1-V
    critic.train(S0, sigma, alpha)
    actor.train(S0, sigma/(a+0.01), beta)
    alpha = alpha*gamma
    beta = beta*gamma
    if i % 10 == 0:
        rewards.append(reward/10)
        actions.append(a)
        reward = R
    else:
        reward += R
    '''
    if np.random.uniform() > epsilon:
        a = actor.feedforward(S1)
    else:
        a = np.random.uniform()
    S2 = np.array([0,0], dtype=np.float32)
    S2[0] = bitree(S1[0])
    S2[1] = S1[1]-1
    R = -np.abs((S2[0]-S1[0])*a - (OptionValue(S2, strike_price)-OptionValue(S1, strike_price)))
    V = critic.feedforward(S1)
    sigma = R-V
    print(a)
    actor.train(S1, sigma, alpha)
    critic.train(S1, sigma/(a+0.01), beta)
    alpha = alpha*gamma
    beta = beta*gamma
    reward += R
    rewards.append(reward)
    '''
import matplotlib.pyplot as plt

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title('Reward Convergence')
plt.plot(rewards[1:])
plt.subplot(1,2,2)
plt.title('Action Convergence')
plt.plot(actions)