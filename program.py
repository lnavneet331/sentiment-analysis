
# In[9]:


df.isnull().sum()


# In[10]:


#df.info()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[12]:


import warnings
warnings.filterwarnings('ignore')


# In[13]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')


# In[3]:


#df.head()


# In[4]:


#df.tail()


# In[5]:


