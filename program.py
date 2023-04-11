
# In[9]:


df.isnull().sum()


# In[10]:


#df.info()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns





df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()

