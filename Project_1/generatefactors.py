#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 09:26:42 2022

@author: chenwynn
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
import math
import os

# exponential weighting
def exp_w(t, h):
    #@ t: time range
    #@ h: banshuaiqi
    alpha = 1-math.exp(math.log(1/2)/h)
    weights = [(1-alpha)**t]
    for i in range(t):
        weights.append((1-alpha)**(t-i-1)*alpha)
    
    return weights



# pwd = '/Users/chenwynn/Documents/courses_21spring/reinforcement learning/project1/data/'
pwd = os.getcwd() + '/data/'

stock_list = list(pd.read_csv(pwd+'stock_list.csv').iloc[:,1])

stock_price = pd.read_csv(pwd+'stock_price.csv')
stock_price = stock_price.rename(columns={'time':'date'})
# calculate stock return
close = stock_price.set_index(['date', 'code'])[['close']].unstack()
ret_stock = np.exp(np.log(close).diff())-1
ret_stock.dropna(inplace=True)
ret_stock.columns = ret_stock.columns.droplevel(0)
close.columns = close.columns.droplevel(0)

#@ calculate beta, fit stock returns against hushen 300 index

index_300 = pd.read_csv(pwd+'index_300.csv')
index_300 = index_300.rename(columns={'Unnamed: 0':'date'})
ret_index = np.exp(np.log(index_300.set_index('date')[['close']]).diff())-1
ret_index.dropna(inplace=True)
#regression rs = alpha + beta*r + e
beta = {}
r_square = {}
alpha = {}
for stock in stock_list:
    t = 252
    tmp = []
    tmp2 = []
    tmp3 = []
    while t < 2430:
        t = t+1
        model = LinearRegression().fit(ret_index.iloc[(t-252):t], ret_stock[stock].iloc[(t-252):t], sample_weight=exp_w(251, 63))
        tmp.append(model.coef_[0])
        tmp2.append(model.score(ret_index.iloc[(t-252):t], ret_stock[stock].iloc[(t-252):t], sample_weight=exp_w(251, 63)))
        tmp3.append(model.intercept_)
    beta[stock] = tmp
    r_square[stock] = tmp2
    alpha[stock] = tmp3

beta = pd.DataFrame(beta)
r_square = pd.DataFrame(r_square)
alpha = pd.DataFrame(alpha)

#@ calculate momentum 126-5 h=31

momentum = {}
for stock in stock_list:
    t = 126
    tmp = []
    while t < 2430:
        t = t+1
        tmp.append(sum(np.log(ret_stock[stock].iloc[(t-126):t]+1)*exp_w(125, 31)))
    momentum[stock] = tmp
    
#@ size = ln market_cap

market_cap = pd.read_csv(pwd+'market_cap_3.csv').set_index('date')

size = np.log(market_cap.dropna())
    
#@ calculate earningyild = 0.45 · ETOP + 0.55 · CETOP

etop = pd.read_csv(pwd+'ep_ratio_ttm.csv').set_index('date')
cetop = pd.read_csv(pwd+'operating_cash_flow_per_share_ttm.csv').set_index('date')/close  
earnyild = 0.45*etop+0.55*cetop    

#@ resvol = 0.74 · DASTD + 0.16 · CMRA + 0.10 · HSIGMA

dastd = {}
for stock in stock_list:
    t = 252
    tmp = []
    while t < 2430:
        t = t+1
        tmp.append(sum((ret_stock[stock].iloc[(t-252):t]-ret_stock[stock].iloc[(t-252):t].mean())**2 *exp_w(251, 63)))
    dastd[stock] = tmp
dastd = pd.DataFrame(dastd)
    
cmra = (ret_stock.rolling(252).max()-ret_stock.rolling(252).min()).shift().dropna()

r_square.index = cmra.index
beta.index = cmra.index
dastd.index = cmra.index
# SSE = (1-R^2)*SST
hsigma = (1-r_square)*dastd

resvol = np.sqrt(dastd)*0.74 + cmra*0.16 + np.sqrt(hsigma)*0.1

#@ growth =  0.61 · SGRO + 0.39 · EGRO

sgro = pd.read_csv(pwd+'basic_earnings_per_share.csv').set_index('date')
egro = pd.read_csv(pwd+'adjusted_earnings_per_share_ttm.csv').set_index('date')

growth = 0.61*sgro + 0.39*egro

#@ bp: book to price

bp = pd.read_csv(pwd+'book_to_market_ratio_ttm.csv').set_index('date')

#@ leverage = 0.38 · MLEV + 0.35 · DTOA + 0.27 · BLEV

# since long term payable, loans, and bond data have too many missing values, we can't calculate MLEV

dtoa = pd.read_csv(pwd+'debt_to_asset_ratio_ttm.csv').set_index('date').dropna() 
blev = pd.read_csv(pwd+'book_to_market_ratio_ttm.csv').set_index('date').dropna()
leverage = 0.54*dtoa + 0.46*blev

#@ liquidty = 0.35 · STOM + 0.35 · STOQ + 0.30 · STOA

turnover = pd.read_csv(pwd+'turnover.csv')
turnover = turnover.rename(columns = {'tradedate':'date'}).set_index(['date', 'order_book_id'])
liquidty = (0.35*turnover['week'] + 0.35*turnover['month'] + 0.3*turnover['year']).unstack()


#@ maintain time consistency and data output
os.mkdir(pwd+'factors/')

time_index = beta.index

ret_stock.reindex(time_index).to_csv(pwd+'factors/ret_stock.csv')
ret_index.reindex(time_index).to_csv(pwd+'factors/ret_index.csv')

alpha.reindex(time_index).to_csv(pwd+'factors/alpha.csv')
beta.reindex(time_index).to_csv(pwd+'factors/beta.csv')

momentum = pd.DataFrame(momentum)
momentum.index = ret_stock.iloc[(2430-2304):2430,:].index
momentum.reindex(time_index).to_csv(pwd+'factors/momentum.csv')

size.reindex(time_index).to_csv(pwd+'factors/size.csv')

earnyild.reindex(time_index).to_csv(pwd+'factors/earnyild.csv')

resvol.to_csv(pwd+'factors/resvol.csv')

growth.reindex(time_index).to_csv(pwd+'factors/growth.csv')

bp.reindex(time_index).to_csv(pwd+'factors/bp.csv')

leverage.reindex(time_index).to_csv(pwd+'factors/leverage.csv')

liquidty.reindex(time_index).to_csv(pwd+'factors/liquidty.csv')























    
    
    