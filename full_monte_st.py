#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:23:32 2022

@author: josevans
"""

# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime, timedelta
import base64
import streamlit as st
import requests

st.write("""
         
# Montecarlo Simulator

Enter the required fields

Note: When downloading CSV files you must append .csv to filename after downloading in order to open

""")

days=730
rfr=.1
coupons=[.05,.1]
start_prices=[27,28]
volatility=[1]
num_tokens=650000
principals=[5000000]
bc_share=0
num_sims=5

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
        
        df.loc[df.index == i, 'principal'] = max(df.iloc[i-1].principal+(1/365*coupon*df.iloc[i-1].principal)-df.iloc[i].principal_repayment, 0)
        
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
    
    mc_df = pd.DataFrame({'Expected_Vol':vol,'Actual_vol':realised_vol,'Days':days, 'Risk_Free_Rate':rfr, 'Coupon':coupon,'Start_Price':start_price,  'Number_Of_Tokens':num_tokens, 'Principal':principal, 'BC_Share_Of_Profits':bc_share,  'Funding_Cost':funding_cost,'Interest':interest, 'Return_30d':return30d, 'Loan_Maturity_Days':loan_maturity, 'Total_sales':total_sales, 'Sales_During_Loan':loan_df_sales, 'Profit':profit, 'Asset_Coverage':asset_coverage, 'Default':default, 'Loss_Given_Default':loss, 'Average_Token_Price':average_price, 'Min_Price':min_price, 'Max_Price':max_price, 'Residual':sales_diff, 'Rebate':rebate}, index = [0])
    
    return [pd.DataFrame(df.token_price).T, mc_df, df]
  

list_dfs = gen_df(730, .12, .12, 42, 1, 650000, 10000000, 30)    

montecarlo = list_dfs[1]
price_df = list_dfs[0]

def get_result(result):
    global results
    results.append(result)

def full_monty(num, mc_df, price_df):
    
    for i in range(num):
        if i%50 == 0:
            print(i)
        list_dfs = gen_df(730, .12, .12, 42, 1, 650000, 10000000, .3)  
        mc_df = mc_df.append(list_dfs[2])
        price_df = price_df.append(pd.DataFrame(list_dfs[0].token_price).T)
    
    return mc_df, price_df

# print(datetime.now())

# montecarlo, price_df = full_monty(1000, montecarlo, price_df)

# print(datetime.now())



def full_monty_async(num, mc_df, price_df, args):
    
    pool = mp.Pool(mp.cpu_count())

    for i in range(num):
        if i%1000 == 0:

            print('Working {}')
        pool.apply_async(gen_df, args=args, callback=get_result)
        
    pool.close()
    pool.join()
    
tickers = list(pd.read_csv('digital_currency_list.csv')['currency code'])

asset = st.sidebar.selectbox("Asset - e.g. ETH", options=tickers, index=tickers.index('DOT'), on_change=reset_price)

days = st.sidebar.number_input('Life of Loan in Days', value=730)

rfr = st.sidebar.number_input('Funding cost, e.g. for 10% input 10', value=10)/100

coupons = st.sidebar.multiselect('Loan Coupon(s) - Chose multiple if desired', options=list(np.round(np.arange(.01, .2, .01), 2)))

current_price=get_spot(asset)

start_prices = st.sidebar.multiselect('Starting Price(s) of Asset', options=list(np.round(np.arange(0, current_price*3, .1))))

volatility = st.sidebar.multiselect('Volatility(ies)', options=list(np.round(np.arange(.1, 3, .1), 2)))

num_tokens = st.sidebar.number_input('Number of tokens to model', value=650000)

principals = st.sidebar.multiselect('Loan Principal Amount(s)', options=list(range(100000, 30000000, 100000)))

bc_share = st.sidebar.multiselect('Blockchain Share of Residual Profits', options=list(np.round(np.arange(0.01, .99, .01))))

num_sims = st.sidebar.number_input('Number of Simulations', value=5)    

start = st.button('Compute Scenario(s)')
    
if start:   
    st.write(datetime.now())
    all_results = pd.DataFrame()
    price_df = pd.DataFrame()
    analyses = {}
        
    
        
     
    for principal in principals:    
      for vol in volatility:
        for spot in start_prices:
          for coupon in coupons:
            new = pd.DataFrame()  
            args = (days, rfr, coupon, spot, vol, num_tokens, principal, 0)
            for i in range(num_sims):
                       
       
                results = gen_df(days, rfr, coupon, spot, vol, num_tokens, principal, 0)

                new = new.append(results[1])

                price_df = price_df.append(results[0])
                
            all_results = all_results.append(new)
            analysis = pd.DataFrame.from_dict({'Loan_Maturity_Days':[new['Loan_Maturity_Days'].mean(), new['Loan_Maturity_Days'].min(),new['Loan_Maturity_Days'].max(), np.percentile(new['Loan_Maturity_Days'],.05), np.percentile(new['Loan_Maturity_Days'],.95), new['Loan_Maturity_Days'].std()], 
                                       'Residual_Post_Waterfall':[new['Residual'].mean(), new['Residual'].min(),new['Residual'].max(), np.percentile(new['Residual'],.05), np.percentile(new['Residual'],.95), new['Residual'].std()], 
                                       'Interest':[new['Interest'].mean(), new['Interest'].min(),new['Interest'].max(), np.percentile(new['Interest'],.05), np.percentile(new['Interest'],.95), new['Interest'].std()], 
                                       'Asset_Coverage':[new['Asset_Coverage'].mean(), new['Asset_Coverage'].min(),new['Asset_Coverage'].max(), np.percentile(new['Asset_Coverage'],.05), np.percentile(new['Asset_Coverage'],.95), new['Asset_Coverage'].std()],
                                       'Probability_Of_Default':[new.Default.sum()/len(new)],
                                       'Loss_Given_Default':[new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'].mean(), new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'].min(),new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'].max(), np.percentile(new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'],.05), np.percentile(new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'],.95), new.loc[new['Loss_Given_Default'] < 0]['Loss_Given_Default'].std()],
                                       '{}_Average_Price'.format(asset):[new['Average_Token_Price'].mean(), new['Average_Token_Price'].min(),new['Average_Token_Price'].max(), np.percentile(new['Average_Token_Price'],.05), np.percentile(new['Average_Token_Price'],.95), new['Average_Token_Price'].std()],
                                       '{}_Min_Price'.format(asset):[new['Min_Price'].mean(), new['Min_Price'].min(),new['Min_Price'].max(), np.percentile(new['Min_Price'],.05), np.percentile(new['Min_Price'],.95), new['Min_Price'].std()], 
                                       '{}_Max_Price'.format(asset):[new['Max_Price'].mean(), new['Max_Price'].min(),new['Max_Price'].max(), np.percentile(new['Max_Price'],.05), np.percentile(new['Max_Price'],.95), new['Max_Price'].std()],
                                       'Funding_Cost':[new['Interest'].mean(), new['Interest'].min(),new['Interest'].max(), np.percentile(new['Interest'],.05), np.percentile(new['Interest'],.95), new['Interest'].std()], 
                                       'Profit':[new['Profit'].mean(), new['Profit'].min(),new['Profit'].max(), np.percentile(new['Profit'],.05), np.percentile(new['Profit'],.95), new['Profit'].std()]},
                                        orient='index', columns=['Average', 'Min', 'Max', '5th_p', '95th_p', 'Std'])
            analyses[args] = analysis
            st.write(args)
            st.write('(days, rfr, coupon, start price, vol, number of tokens, BC share of resid)')
            st.write(analysis)

            
    copy = all_results
    
    all_results = all_results.groupby(['Principal', 'Start_Price', 'Expected_Vol', 'Coupon'])

    defaulted =  copy.loc[copy.Loss_Given_Default < 0].groupby(['Principal', 'Start_Price', 'Expected_Vol', 'Coupon'])
    
    temp = pd.DataFrame(all_results.Profit.mean())
    temp['Residual'] = pd.DataFrame(all_results.Residual.mean())
    temp['LGD'] = defaulted.Loss_Given_Default.mean()
    temp['Loan_Average_Life'] = pd.DataFrame(all_results.Loan_Maturity_Days.mean())
    temp['Percent_Default'] = all_results.Default.sum()/all_results.Default.count()
    
    st.write(temp)

    st.write('Click to Download Analysis')
    
    st.markdown(get_table_download_link(temp.reset_index()), unsafe_allow_html=True)

    st.write('Click to Download All Results in Full')
      
    st.markdown(get_table_download_link(copy), unsafe_allow_html=True)
    
    st.write(datetime.now())

    


