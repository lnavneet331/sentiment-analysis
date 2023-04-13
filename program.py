
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


#df.info()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


import warnings
warnings.filterwarnings('ignore')


# In[13]:


df['Liked'].unique()


# In[14]:


df['Liked'].value_counts()


# In[15]:


plt.figure(figsize=(15,6))
sns.countplot(df['Liked'], data = df, palette = 'hls')
plt.show()


# In[16]:


balance_counts = df.groupby('Liked')['Liked'].agg('count').values
#balance_counts


# In[17]:


from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


# In[18]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x = [0],
    y=[balance_counts[0]],
    name='Like',
    text=[balance_counts[0]],
    textposition='auto',
    marker_color= 'blue'
))
fig.add_trace(go.Bar(
    x= [1],
    y=[balance_counts[1]],
    name='Dislike',
    text=[balance_counts[1]],
    textposition='auto',
    marker_color= 'red'
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by Likes</span>'
)
fig.show()


# In[19]:


df['Review_Length'] = df['Review'].apply(lambda x: len(x.split(' ')))


# In[20]:


plt.figure(figsize=(15,6))
sns.histplot(df['Review_Length'], bins = 20, kde = True, palette = 'hls')
plt.show()


# In[21]:


like_df = df[df['Liked'] == 0]['Review_Length'].value_counts().sort_index()
dislike_df = df[df['Liked'] == 1]['Review_Length'].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=like_df.index,
    y=like_df.values,
    name= 0,
    fill='tozeroy',
    marker_color= 'blue',
))
fig.add_trace(go.Scatter(
    x=dislike_df.index,
    y=dislike_df.values,
    name=1,
    fill='tozeroy',
    marker_color= 'red',
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Data Distribution in Different Fields</span>'
)
fig.update_xaxes(range=[0, 70])
fig.show()


# In[22]:


#df


# In[23]:


df.Review_Length.describe()


# In[24]:


df_new = df.copy()


# In[25]:

