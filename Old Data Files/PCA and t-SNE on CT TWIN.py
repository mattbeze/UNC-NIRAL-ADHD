#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd


# In[2]:


master = pd.ExcelFile("CONTE_TWIN_CT_SA_tlaplace_Master.xlsx") 


# In[3]:


master.sheet_names


# In[4]:


ct1_2y = master.parse('1_2_TWIN_CT_tlaplace_dROI')
#ct1_2_twin = master.parse('1_2_TWIN_CT_tlaplace_dROI')
ct4_6y = master.parse('4_6_TWIN_CT_tlaplace_dROI')
#ct4_6_twin = master.parse('4_6_TWIN_CT_tlaplace_dROI')


# In[5]:


ct1_2y.rename(columns={'ROI':'SubjectId'},inplace=True)
ct4_6y.rename(columns={'ROI':'SubjectId'},inplace=True)

# There is probably a better way to add the age column to each record based off the recordId
ct1y = ct1_2y.loc[ct1_2y['SubjectId'].str.contains("1yr"),:]
ones = pd.DataFrame(data=[1] * ct1y.shape[0], columns=["Age"])
ct1y = pd.concat([ct1y, ones] ,axis=1).dropna()
     
ct2y = ct1_2y.loc[ct1_2y['SubjectId'].str.contains("2yr"),:]
twos = pd.DataFrame(data=[2] * ct2y.shape[0], columns=["Age"])
ct2y = pd.concat([ct2y, twos] ,axis=1).dropna()

ct4y = ct4_6y.loc[ct4_6y['SubjectId'].str.contains("4yr"),:]
fours = pd.DataFrame(data=[4] * ct4y.shape[0], columns=["Age"])
ct4y = pd.concat([ct4y, fours] ,axis=1).dropna()

ct6y = ct4_6y.loc[ct4_6y['SubjectId'].str.contains("6yr"),:]
sixes = pd.DataFrame(data=[6] * ct6y.shape[0], columns=["Age"])
ct6y = pd.concat([ct6y, sixes] ,axis=1).dropna()

# Combine dataframe
ctdf = pd.concat([ct1y, ct2y, ct4y, ct6y], axis=0)
ctdf.head()


# In[6]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[7]:


# standardize values for PCA
x = ctdf.loc[:, ~ctdf.columns.isin(['SubjectId', 'Age'])]
x = StandardScaler().fit_transform(x)
x.shape

features = ctdf.drop(['SubjectId', 'Age'], axis=1)


# In[8]:


pca = PCA(n_components=150)
pca.fit(x)

pca


# In[9]:


pcs = []
for i in range(1,151):
    pcs.append('PC' + str(i))
principalComponentDf = pd.DataFrame(data = np.transpose(pca.components_), columns = pcs)
columnNames = pd.DataFrame(data = ctdf.columns.values.tolist()[1:], columns = ['ROI'])
principalComponentDf = pd.concat([columnNames, principalComponentDf], axis=1)
# why does the dataframe have 151 rows?
principalComponentDf.head()


# In[10]:


cumVar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumVar)
plt.ylabel("Cumulative Variance Explained")
plt.xlabel("Principal Component")


# In[11]:


# Keep only the principal components that cumulatively explain at 90% of the variance
imptPCs = pcs[:len(cumVar[cumVar <= .9])]
principalComponentDf = principalComponentDf.loc[:, imptPCs]
#principalComponentDf.head()


# In[12]:


# For visualization keep 2 principal components
print(pca.explained_variance_ratio_[0:2]) #variance explained by first two PCs

#x = ctdf.loc[:, ~ctdf.columns.isin(['SubjectId', 'Age'])]
firstTwoPCs = pd.DataFrame(data = pca.components_[:,:2], columns = ['PC1', 'PC2'])
pcScores = pd.DataFrame(data = np.dot(x, firstTwoPCs), columns = ['PC1', 'PC2'])
ages = ctdf.loc[:,['Age']].reset_index()
pcScores = pd.concat([pcScores,ages['Age']], axis = 1)
pcScores.head()


# In[13]:


from ggplot import * 
#from ggplot import scale_fill_brewer

chart = ggplot( pcScores, aes(x='PC1', y='PC2', color='Age') )         + geom_point(size=75,alpha=0.8)         + ggtitle("First and Second Principal Components colored by digit")
chart


# In[14]:


import time

from sklearn.manifold import TSNE

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(features.values)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# In[15]:


df_tsne = ctdf.copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='Age') )         + geom_point(size=70,alpha=1.0,)         + ggtitle("tSNE dimensions colored by digit")
chart


# In[ ]:




