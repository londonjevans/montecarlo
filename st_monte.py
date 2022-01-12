#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:13:36 2022

@author: josevans
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
import streamlit as st
import requests
import base64

st.write("""
         
# Montecarlo Simulator

Enter the required fields

""")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    
    return href
    
@st.cache(ttl=86400)
def get_spot(asset):
    url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency=USD&apikey=W7Y4ZQP4R37OFPRO'.format(asset)
    r = requests.get(url)
    
    df = pd.DataFrame(r.json()).T
    
    price = pd.to_numeric(df.iloc[0, 4])

    return price

@st.cache(ttl=86400)
def get_cbs(asset):
    now = datetime.now()
    ticker = asset+'-USD'
    start = str(now.date()-timedelta(days=1))
    gran = 60 # in seconds, {60, 300, 900, 3600, 21600, 86400} , 1 second --> 1 day
    r = f"https://api.pro.coinbase.com/products/{ticker}/candles?start={start}T00:00:00&granularity={gran}"
    j = requests.get(r).json()

    df = pd.DataFrame(j)
    price = df.iloc[0, 4]
    return price

def reset_price():
    price=0
    return price

def gen_df(days, rfr, coupon, start_price, vol, num_tokens, principal, bc_share):
    
    df = pd.DataFrame(index=range(days))
    
    default = False
    loss = 0
    
    df['discount_factor'] = 1-(rfr*df.index/365)
    df['z_1'] = stats.norm.rvs(loc=0,scale=1, size=days, random_state = None)
    df['z_2'] = stats.norm.rvs(loc=0,scale=1,size=days,  random_state = None)
    df['tokens_received'] = 0
    df['token_price'] = start_price
    df['principal_repayment'] = 0
    df['principal'] = principal
    df['tokens_sold'] = num_tokens/days
    df.loc[df.index == 0, 'tokens_sold'] = 0
    df['daily_change'] = 0
    
    for i in range(0, days, 30):
        df.loc[df.index == i, 'tokens_received'] = num_tokens/24
        
    for i in range(1, days):
        
        df.loc[df.index == i, 'token_price'] = df['token_price'][i-1]+df['z_1'][i]*vol*np.sqrt(1/365)*df['token_price'][i-1] # +rfr*1/365*df['token_price'][i-1]

    for i in range(1, len(df)):    
        
        df.loc[df.index == i, 'daily_change'] = df.token_price[i]/df.token_price[i-1]-1
    
    
    
    for i in range(0, days, 30):
        if i > 0:
            try:
                df.loc[df.index == i, 'principal_repayment'] = num_tokens/24 * np.mean(df[i-29:i+1].token_price)
            except Exception as e:
                print(e)
            
    for i in range(1, days):
        
        df.loc[df.index == i, 'principal'] = max(df.iloc[i-1].principal+(1/365*coupon*df.iloc[i-1].principal)-df.iloc[i].principal_repayment*.99, 0)
        
    df['funding_cost'] = df.principal.shift(1)*rfr*1/365
    df['accrued_interest'] = df.principal.shift(1)*coupon*1/365
    
    loan_maturity = df.loc[df.principal == 0].index.min()
    if loan_maturity > 0:
        loan_df = df.loc[df.index <= loan_maturity]
    else:
        loan_maturity = days
        loan_df = df
    
    funding_cost = np.nansum(loan_df.funding_cost)
    return30d = round((df.iloc[30].token_price - start_price)/start_price*100, 2)
    total_sales = np.dot(df.token_price, df.tokens_sold)
    loan_df_sales = np.dot(loan_df.token_price, loan_df.tokens_sold)
    sales_diff = total_sales-loan_df_sales
    interest = np.nansum(loan_df.accrued_interest)
    
    if len(loan_df) < days:
        rebate = loan_df.iloc[-1].principal_repayment - loan_df.iloc[-2].principal
    else:
        rebate = 0 
    
   
    profit = loan_df_sales+sales_diff*bc_share-funding_cost-principal+interest+rebate*bc_share-rebate
    if profit < 0:
        default = True
        loss = profit
    
    asset_coverage = total_sales / principal
    average_price = np.mean(df.token_price)
    min_price = df.token_price.min()
    max_price = df.token_price.max()
    realised_vol = np.std(df.daily_change, ddof=1)*np.sqrt(365)
    
    mc_df = pd.DataFrame({'Expected_Vol':vol,'Actual_vol':realised_vol,'Days':days, 'Risk_Free_Rate':rfr, 'Coupon':coupon,'Start_Price':start_price,  'Number_Of_Tokens':num_tokens, 'Principal':principal, 'BC_Share_Of_Profits':bc_share,  'Funding_Cost':funding_cost, 'Return_30d':return30d, 'Loan_Maturity_Days':loan_maturity, 'Total_sales':total_sales, 'Sales_During_Loan':loan_df_sales,'Interest':interest, 'Profit':profit, 'Asset_Coverage':asset_coverage, 'Default':default, 'Loss_Given_Default':loss, 'Average_Token_Price':average_price, 'Min_Price':min_price, 'Max_Price':max_price, 'Residual':sales_diff, 'Rebate':rebate}, index = [0])
    
    return [pd.DataFrame(df.token_price).T, mc_df, df]

tickers = list(pd.read_csv('digital_currency_list.csv')['currency code'])

asset = st.sidebar.selectbox("Asset - e.g. ETH", options=tickers, index=tickers.index('DOT'), on_change=reset_price)



days = st.sidebar.number_input('Life of Loan in Days', value=730)

rfr = st.sidebar.number_input('Funding cost, e.g. for 10% input 10', value=10)/100

coupon = st.sidebar.number_input('Loan Coupon, e.g. for 10% input 10', value=10)/100

start_price = st.sidebar.number_input('Starting Price of Asset', value=get_spot(asset))

volatility = st.sidebar.number_input('Volatility, e.g. for 100% input 100', value=100)/100

num_tokens = st.sidebar.number_input('Number of tokens to model', value=650000)

principal = st.sidebar.number_input('Loan Principal Amount', value=10000000)

bc_share = st.sidebar.number_input('Blockchain Share of Residual Profits', value=30)/100

num_sims = st.sidebar.number_input('Number of Simulations', value=5)

starter = st.button('Click to Generate')



if starter:
    
    montecarlo = pd.DataFrame()
    price_df = pd.DataFrame()
    
    for i in range(num_sims):
        if i%50 == 0:

            st.write('{} sims done'.format(i))
            st.write(datetime.now())
        
        list_dfs = gen_df(days, rfr, coupon, start_price, volatility, num_tokens, principal, bc_share)   
        
        montecarlo = montecarlo.append(list_dfs[1])
        
        price_df = price_df.append(list_dfs[0])
    
    lgd = montecarlo.loc[montecarlo.Loss_Given_Default < 0].Loss_Given_Default.mean()
        
    temp = pd.DataFrame(montecarlo.groupby(['Principal', 'Start_Price', 'Expected_Vol', 'Coupon']).Profit.mean())
    temp['Residual'] = pd.DataFrame(montecarlo.groupby(['Principal', 'Start_Price', 'Expected_Vol', 'Coupon']).Residual.mean())
    temp['LGD'] = lgd
    temp['Loan_Average_Life'] = pd.DataFrame(montecarlo.groupby(['Principal', 'Start_Price', 'Expected_Vol', 'Coupon']).Loan_Maturity_Days.mean())
    temp['Percent_Default'] = montecarlo.groupby(['Principal','Start_Price', 'Expected_Vol', 'Coupon']).Default.sum()/montecarlo.groupby(['Principal','Start_Price', 'Expected_Vol', 'Coupon']).Default.count()
    
    st.write(temp)
    
    percentile = pd.DataFrame(montecarlo.groupby(['Principal','Start_Price', 'Expected_Vol', 'Coupon']).describe(percentiles=[.05,.95])).drop(columns=['BC_Share_Of_Profits', 'Return_30d', 'Actual_vol', 'Days', 'Risk_Free_Rate', 'Number_Of_Tokens', 'Profit'])
    
    st.write('Click to Download Analysis')
        
    st.markdown(get_table_download_link(temp), unsafe_allow_html=True)
    
    st.write(percentile)
    
    st.write('Click to Download Percentile Analysis')
        
    st.markdown(get_table_download_link(percentile), unsafe_allow_html=True)     
    
    st.write('Click to Download Raw Results')
        
    st.markdown(get_table_download_link(montecarlo), unsafe_allow_html=True)
        
    st.write('Click to Download Raw Price Simulations')
        
    st.markdown(get_table_download_link(price_df), unsafe_allow_html=True)








