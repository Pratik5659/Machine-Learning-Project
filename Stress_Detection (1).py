#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


df=pd.read_csv("C:/Users/prati/OneDrive/Desktop/Talent Battle/Machine Learning/Projet 1/spam.csv",encoding="latin-1")


# In[7]:


#Visualizing dataset
df.head(n=10)


# In[9]:


df.shape


# In[11]:


#To Check whether target attribute is binary or not
np.unique(df['class'])


# In[13]:


np.unique(df['message'])


# In[15]:


#Creating Sparse Matrix
x=df["message"].values
y=df["class"].values

#create count Vectorizer Object
cv=CountVectorizer()

x=cv.fit_transform(x)
v=x.toarray()

print(v)


# In[19]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[21]:


#Spliting train + Test 3:1
train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[23]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)

y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[25]:


#Training Score
print(bnb.score(train_x,train_y)*100)

#Testing Score
print(bnb.score(test_x,test_y)*100)


# In[30]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[31]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[ ]:




