
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


#df.shape


# In[6]:


#df.columns


# In[7]:


df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()


# In[9]:


df.isnull().sum()


# In[10]:


