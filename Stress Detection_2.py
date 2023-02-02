#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv("C:/Users/prati/OneDrive/Desktop/Talent Battle/Machine Learning/Projet 1/stress.csv")
df.head()


# In[6]:


df.describe()


# In[8]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[ ]:




