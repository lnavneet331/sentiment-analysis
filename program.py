import pandas as pd
df = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

df.duplicated().sum()

df = df.drop_duplicates()

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df['Liked'].unique()

df['Liked'].value_counts()

plt.figure(figsize=(15,6))
sns.countplot(df['Liked'], data = df, palette = 'hls')
plt.show()

balance_counts = df.groupby('Liked')['Liked'].agg('count').values

from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

fig = go.Figure()
fig.add_trace(go.Bar(
    x = [0],
    y=[balance_counts[0]],
    name='Like',
