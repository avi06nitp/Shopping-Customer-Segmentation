#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C:/Users/Gaelim/Desktop/Datasets/Segmentation Data/Mall_Customers.csv")


# In[3]:


df.head()


# # Univariate Analysis

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)']);


# In[6]:


df.columns


# In[7]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[8]:


sns.kdeplot(df['Annual Income (k$)'],shade=True,hue=df['Gender']);


# In[9]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.kdeplot(df[i],shade=True,hue=df['Gender'])


# In[10]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[11]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[12]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )


# In[13]:


#df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[14]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[15]:


df.corr()


# In[16]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate

# In[33]:


clustering1 = KMeans(n_clusters=3)


# In[34]:


clustering1.fit(df[['Annual Income (k$)']])


# In[35]:


clustering1.labels_


# In[36]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[37]:


df['Income Cluster'].value_counts()


# In[38]:


clustering1.inertia_


# In[39]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[40]:


intertia_scores


# In[41]:


plt.plot(range(1,11),intertia_scores)


# In[42]:


df.columns


# In[43]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[ ]:


#Bivariate Clustering


# In[46]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[47]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[55]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[77]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[58]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[59]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[60]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[61]:


scale = StandardScaler()


# In[62]:


df.head()


# In[64]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[65]:


dff.columns


# In[68]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[70]:


dff = scale.fit_transform(dff)


# In[74]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[75]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[76]:


df


# In[78]:


df.to_csv('Clustering.csv')


# In[ ]:




