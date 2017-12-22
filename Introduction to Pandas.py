
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[22]:


df = pd.read_csv('train.csv',header=0)
df.to_csv('train_2.csv',index=False)


# In[3]:


df.head()


# In[23]:


df.iloc[5:10,0:5]


# In[5]:


df[['Survived','Pclass']]


# In[2]:


# Create DataFrame from numpy array
df = pd.DataFrame(np.zeros([3,2]), columns=['date','isSunny'])
df


# In[4]:


# Create numpy array from DataFrame
np_array = df.values
print(type(np_array))


# In[8]:


# handling date and time
df['date']=['2018-01-01','2018-01-02','2018-01-03']
df['isSunny']=[True,False,True]
date = pd.to_datetime(df['date'],infer_datetime_format=True)
date


# In[9]:


date.dt.month


# In[10]:


# get date from date-time object
date.dt.day


# In[11]:


# get day of the week from date-time object
date.dt.dayofweek


# In[12]:


# series to DataFrame
series_to_dict = {'year':date.dt.year,'month':date.dt.month,'date':date.dt.day,'day':date.dt.dayofweek}
df_new = pd.DataFrame(series_to_dict)
df_new


# In[13]:


# Create DataFrame with NaN
df = pd.DataFrame(pd.np.empty([3,2])*pd.np.nan, columns=['date','isSunny'])
df


# In[14]:


# use of fillna
df['isSunny'] = df['isSunny'].fillna(0)
df


# In[16]:


# Replace elements
df['isSunny'].replace([0.0,1.0],[1.0,2.0], inplace=True)
df


# In[17]:


# concat and reference issue
df_1 = df
df_2 = df
df_2.iloc[1][1]=234
df_new = pd.concat([df_1,df_2],axis=1)
df_new


# In[18]:


# data churning
df = pd.DataFrame(np.arange(50),columns=['val'])
df_max = df.nlargest(5,'val')
df = df.drop(df_max.index.get_level_values(0))
df


# In[24]:


# One-hot-encoding
df = pd.DataFrame({'Day':['Monday','Tuesday','Wednesday'],'Date':[1,2,3]})
df


# In[25]:


df_oh = pd.get_dummies(df)
df_oh

