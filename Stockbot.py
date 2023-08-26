#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from datetime import date,timedelta


# In[5]:


from pandas.plotting import register_matplotlib_converters


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import matplotlib.dates as mdates


# In[10]:


from sklearn.preprocessing import MinMaxScaler


# In[11]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[12]:


from keras.models import Sequential


# In[13]:


from keras.layers import LSTM ,Dense


# In[14]:


today=date.today()


# In[15]:


date_today=today.strftime("%Y-%m-%d")


# In[16]:


date_start='2013-01-01'


# In[17]:


#get stock


# In[60]:


stock=input("Enter Stock name:")


# In[61]:


stockname=stock


# In[62]:


symbol='</>'


# In[63]:


import pandas_datareader as webreader


# In[64]:


#df = webreader.DataReader(symbol,start=date_start,end=date_today,data_source="yahoo")


# In[65]:


import yfinance as yf


# In[66]:


df=yf.download(symbol,start=date_start,end=date_today)


# In[67]:


print (df.shape)


# In[68]:


df.head(5)


# In[69]:


register_matplotlib_converters()


# In[70]:


years=mdates.YearLocator()


# In[71]:


fig ,ax1=plt.subplots(figsize=(16,6))


# In[72]:


ax1.xaxis.set_major_locator(years)


# In[73]:


x=df.index


# In[74]:


y=df['Close']


# In[75]:


ax1.fill_between(x, 0,y ,color='#b9e1fa')


# In[76]:


ax1.legend([stockname], fontsize=12)


# In[77]:


plt.title (stockname+' from '+ date_start +' to ' +date_today)


# In[78]:


plt.plot(y, color= '#039dfc',label=stockname, linewidth=1.0)


# In[79]:


plt.ylabel('Stocks',fontsize=12)


# In[80]:


train_df=df.filter(['Close'])


# In[81]:


data_unscaled=train_df.values


# In[82]:


train_data_length=math.ceil(len(data_unscaled)*0.8)


# In[83]:


mmscalar = MinMaxScaler(feature_range=(0,1))


# In[84]:


np_data=mmscalar.fit_transform(data_unscaled)


# In[85]:


sequence_length=50


# In[86]:


#prediction Index


# In[87]:


index_Close = train_df.columns.get_loc("Close")


# In[88]:


print(index_Close)


# In[89]:


train_data_len = math.ceil(np_data.shape[0]*0.8)


# In[90]:


train_data=np_data[0:train_data_len, :]


# In[91]:


test_data = np_data[train_data_len - sequence_length:, :]


# In[92]:


def partition_dataset(sequence_length , train_df):
    x,y =[] , []
    data_len = train_df.shape[0];
    for i in range(sequence_length,data_len):
        x.append(train_df[i-sequence_length: i, :])
        y.append(train_df[i,index_Close])
        


# In[93]:


x=np.array(x)
y=np.array(y)


# In[94]:


x_train, y_train=partition_dataset()
x_test, y_test=partition_dataset()


# In[95]:


plus='+';minus=''


# In[96]:


print(f'the close price'{predicated price})


# In[ ]:





# In[ ]:




