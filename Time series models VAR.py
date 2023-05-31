#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Time series analysis


# In[54]:


pip install arch


# In[55]:


pip install pmdarima


# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch.unitroot import ADF
from arch.unitroot import PhillipsPerron
from arch.unitroot import KPSS

import pmdarima
from pmdarima.arima import auto_arima
from arch.unitroot import engle_granger


# In[57]:


df1 = pd.read_csv("cpidata.csv")


# In[58]:


df1.head()


# In[59]:


df1.hist()


# In[60]:


#Plotting the data

df1.plot()

#Stationary


# In[61]:


df1.isnull().any()

#There are no missing values 


# In[62]:


### Unit root testing


# In[141]:


#ADF test

adf = ADF(df1["CPI"])
print(adf.summary().as_text())
# adf
# You can also run adf command to get data in a table, but printing it as text gives more information; eg. Tells H0 and H1.


# Inference: As p-value is greater than 0.05, we cannot reject the null hypothesis of stati


# In[64]:


#Philips Pherron test

pp = PhillipsPerron(df1["CPI"])
print(pp.summary().as_text())


# In[65]:


#KPSS test

kpss = KPSS(df1["CPI"])
print(kpss.summary().as_text())

#The results of KPSS-test are different because H0 and H1 are interchanged; H0 = Non-stationarity


# In[66]:


#Adding a new variable "d_CPI" to the dataframe that tells the first difference

df1["d_CPI"] = df1["CPI"].diff()


# In[68]:


#Checking the newly created dataframe
df1
#There is a missing value, so we cannot run ADF test! We will have to drop this value 


# In[94]:


#Dropping the null values 

df2 = df1.dropna()


# In[95]:


#Checking the dataframe

df2


# In[96]:


#Testing stationarity on this newly created variable

adf2 = ADF(df2["d_CPI"], lags = 12)
print(adf2.summary().as_text())

#OR in one line
#adf2 = ADF(df1["CPI"].diff().dropna(), lags = 12)

#We use 12 lags because CPI is monthly data


# In[97]:


pp2 = PhillipsPerron(df2["d_CPI"], lags = 12)
print(pp2.summary().as_text())


# In[98]:


kpss2 = KPSS(df2["d_CPI"], lags = 12)
print(kpss2.summary().as_text())


# In[99]:


#As the data is not stationary at the first difference also, we need to test unit root on the 2nd difference


# In[100]:


#Making a new variable at the 2nd difference

df2["ld_CPI"] = np.log(df2["CPI"]).diff()


# In[101]:


#Checking the newly created dataframe
df2

#There is a missing value, which we will remove


# In[102]:


#Removing the missing values

df3 = df2.dropna()
df3


# In[87]:


### Stationary tests on the 2nd difference


# In[103]:


#ADF test

adf3 = ADF(df3["ld_CPI"], lags = 12)
print(adf3.summary().as_text())


# In[106]:


#Phillips Perron test

pp3 = PhillipsPerron(df3["ld_CPI"], lags = 12)
print(pp3.summary().as_text())


# In[109]:


#KPSS test

kpss3 = KPSS(df3["ld_CPI"], lags = 12)
print(kpss3.summary().as_text())


# In[111]:


from pmdarima.arima.utils import ndiffs


# In[113]:


#We can go for ndiffs function to automatically give the number of differences

ndiffs(df3["CPI"], test = "adf")


# In[114]:


ndiffs(df3["CPI"], test = "pp")


# In[116]:


ndiffs(df3["CPI"], test = "kpss")


# In[ ]:


### The ndiffs function tells us that the variables are stationary at the first difference, so we need to redo ADF test etc.
# We do this by testing for 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[117]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[120]:


plot_acf(df3["CPI"])

#ACF is for MA - Moving average


# In[121]:


plot_pacf(df3["CPI"])


# In[125]:


#Fitting ARIMA model - auto arima

arimamodel = pmdarima.auto_arima(df3["ld_CPI"])
arimamodel.summary()


#!#! How to interpret?
#!#! See online how to forecast values 


# In[126]:


#!#!#! Homework


#Run a Seasonal ARIMA model
#Run all post estimation commands
#Run a forecast 


# In[127]:


#### VAR model


# In[128]:


import statsmodels.api as sm
from statsmodels.tsa.api import VAR


# In[132]:


df4 = pd.read_excel("VAR.xlsx", index_col = 0, parse_dates = True)
#index_col = 0 --> First column in my excel sheet should be used as the index

df4.head()
#df4 = df4.set_index("Quarter")


# In[134]:


df4.hist()


# In[137]:


df4.plot()

#GAP is non-stationary; other variables seem to be stationary


# In[139]:


#Checking for null values
df4.isnull().any()

#There are no null values


# In[140]:


### Stationarity testing


# In[145]:


#ADF test

adf = ADF(df4["CAL"], lags = 12)
adf


# In[149]:


adf = ADF(df4["CAL"], trend = "c", lags = 12)
adf


# In[151]:


adf = ADF(df4["CAL"], trend = "ct", lags = 12)
adf


# In[153]:


adf = ADF(df4["CAL"], trend = "ctt", lags = 12)
adf

#CAL is stationary with a trend and drift


# In[155]:


pp = PhillipsPerron(df4["GAP"], trend = "c", lags = 12)
pp


# In[157]:


pp = PhillipsPerron(df4["GAP"], trend = "ct", lags = 12)
pp


# In[158]:


pp = PhillipsPerron(df4["GAP"], trend = "ctt", lags = 12)
pp


# In[159]:


#Use arch package for modelling volatility
#There is no paxkage that automaticlaly fits


# In[168]:


#Grander causality test
from statsmodels.tsa.stattools import grangercausalitytests

grangercausalitytests(df4[["CAL", "GAP"]], maxlag=[3])


# In[169]:


grangercausalitytests(df4[["GAP", "CAL"]], maxlag=[3])


# In[170]:


grangercausalitytests(df4[["GAP", "INF"]], maxlag=[3])


# In[171]:


### Disadvantage of time series models in Python: Compared to STATA, R, Eviews etc you dont get customized results,
#and you need to run too many codes 


# In[177]:


#Lag langth p of var
#https://www.statsmodel.ord/devel/vector_ar.html
model = VAR(df4) #recall that rawData is wihtout difference operation

#We choose the lag order with the lowest AIC

for i in [1,2,3,4,5]:
    result = model.fit(i)
    try:
        print("Lag order =", i)
        print("AIC : ", result.aic)
        print("BIC : ", result.bic)
        print("FPE : ", result.fpe)
        print("HQIC : ", result.hqic, "\n")
    except:
        continue
        
# Lag order 2 has the lowest value of AIC, so we choose this.

#We can use more lags, but this is quarterly data and so 4 lags is fine [Choose up to 1 year];
#Daily data - 365 lags
#Monthly data - 12 lags
#Weekly data - 52 lags
#Yearly data


# In[179]:


model = VAR(df4)
model_fitted = model.fit(2)
#Choosing 2 lags
model_fitted.summary()


# In[180]:


#Code for auto var?
#Where did sir get the codes from?


# In[181]:


model1 = arch_model()


# In[183]:


pip install yfinance


# In[205]:


#How to import data from Yahoo Finance

import yfinance as yf
# data = yf.download(["GC = F"], "2021-01-01", "2023-05-16")
data = yf.download(["INFY.BO"], "2021-01-01", "2023-05-16")

#Data from Bombay stock exchange

#GC = F is for gold prices
data.tail()


# In[202]:


#Data from National Stock exchange

data1 = yf.download(["INFY.NS"], "2021-01-01", "2023-05-16")

#Data from National stock exchange

#GC = F is for gold prices
data1.tail()


# In[ ]:


#Do a log transformation to make the data stationary

#data["Return"]=100*(data["Adj"]) ~~~~~~~ Something like this.........

