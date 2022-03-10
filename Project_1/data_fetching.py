#!/usr/bin/env python
# coding: utf-8

import os

cwd = os.getcwd()
pwd = cwd+'/data/'

## initialize API
import rqdatac
rqdatac.init()


# In[2]:


import pandas as pd


# In[4]:

## data which should be download
dataL=["return_on_equity_lyr",
"return_on_equity_ttm",
"return_on_asset_lyr",
"return_on_asset_ttm",
"total_asset_turnover_lyr",
"total_asset_turnover_ttm",
"adjusted_profit_to_total_profit_lyr",
"adjusted_profit_to_total_profit_ttm"
"basic_earnings_per_share",
"book_to_matket_ratio_ttm",
"ep_ratio_ttm",
"surplus_cash_protection_multiples_lyr",
"surplus_cash_protection_multiples_ttm",
"gross_profit_margin_lyr","gross_profit_margin_ttm",
"operating_revenue_growth_ratio_lyr",
"operating_revenue_growth_ratio_ttm",
"net_profit_growth_ratio_lyr",
"net_profit_growth_ratio_ttm",
"operating_cash_flow_per_share_ttm",
"ocf_to_debt_lyr",
"cash_flow_ratio_ttm","debt_to_asset_ratio_ttm","current_ratio_ttm",
"total_assets"]


stockL=pd.read_csv("StockList.csv")
stockL=list(stockL.iloc[:,1])



# In[ ]:



for d in dataL:
    
    
    for i in range(0,len(stockL)):
            s=stockL[i]
            df = rqdatac.get_factor(s,d,
                   start_date='20070101',end_date='20220101')
     
            temp=pd.DataFrame(df.loc[s,:])
            temp.columns=[s]
            if i==0:
                ss=temp
            else:
                ss=pd.concat([ss, temp], axis=1)
        
    doc_n=pwd+d+str(".csv") ## save factor as its name
    ss.to_csv(doc_n)
