
# In[9]:


df.isnull().sum()


# In[10]:




df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()

