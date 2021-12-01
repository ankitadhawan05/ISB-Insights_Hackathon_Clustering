# ISB-Insights_Hackathon

#This project was a part of ISB@Insights Hackathon 2021 whose problem statement was to create a 
#multi-dimensional poverty and deprivation Index based on the Mission Antayodya Survey 2019 (a household survey done to examine the state of infrastructure #facilities in terms of access to healthcare, education, banking and infrastructure facilities)

The dataset was divided into 4 parts to create sub-indices on education, banking, healthcare and infra
The following code is the backend work done to cluster rural districts of India for education and healthcare. K-Means clustering was used to cluster the districts. Similar variables were dropped to remove the multi-collinearit and transformations on the raw data set was done before performing clustering.

For data transformation- based on literature scans, a threshold value of 30% was chosen to define deprivation i.e. the districts that had less than 30% of facilities were considered deprived and assigned a value of 1. rest of the values were assigned values of 0.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
pd.set_option('display.max_columns', 500)
!ls ../input/isb-hackathon-dataset/Hackathon_education.xlsx

!ls ../input/isb-hackathon-dataset/Hackathon_education.xlsx

df=pd.read_excel(r'../input/isb-hackathon-dataset/Hackathon_education.xlsx',sheet_name='%Data')
df.head()

df_dep=pd.read_excel(r'../input/isb-hackathon-dataset/Hackathon_education.xlsx',sheet_name='Dep_Data')
df_dep.head()
df.columns

X=df[['prim_school','middle_school', 'high_school','high_second_school','degree_clg','public_library','indoor_sports','outdoor_sports','both_sports','vocational']].values
X

X_dep=df_dep[['prim_school','middle_school','high_school','high_second_school','degree_clg','public_library','indoor_sports','outdoor_sports','both_sports','vocational']].values
X_dep

from sklearn.decomposition import PCA

pca = PCA().fit(X_dep)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')

plt.show()

Select Number of PCA compnents based on the above graph for both deprivation data and raw dataset
Deprivation dataset was chosen since the transformed dataset defined the scope of the study

pca_dep = PCA(n_components = 3).fit(X_dep).transform(X_dep)
pca_perc = PCA(n_components = 4).fit(X).transform(X)

from sklearn import preprocessing

inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(pca_dep)
    kmeanModel.fit(pca_dep)
    inertia.append(kmeanModel.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()

  kmeans = KMeans(n_clusters=3, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))
  kmeans = KMeans(n_clusters=4, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))
  kmeans = KMeans(n_clusters=5, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))
 
#Choose number of clusters
kmeans = KMeans(n_clusters=3, random_state=1).fit(pca_dep)

#Attach cluster to main dataset
labelsKM_PCA = pd.DataFrame(kmeans.labels_)
labelled_KM_PCA = pd.concat((df_dep,labelsKM_PCA),axis=1)
labelled_KM_PCA = labelled_KM_PCA.rename({0:'labels'},axis=1)
labelled_KM_PCA['labels'].value_counts()

labelled_KM_PCA.head()

eduinfra_df = pd.merge(df,labelled_KM_PCA[['District_code','labels','Dep_Index']],on='District_code').sort_values(by='Dep_Index')
eduinfra_df.head()

sns.lmplot(x='high_school',y='outdoor_sports',data=eduinfra_df,hue='labels',fit_reg=False)

## The same code and steps can be used to cluster districts in access to healthcare.For Healthcare similar varibales were dropped before performing K-Means clustering

!ls ../input/isb-hackathon-dataset/Hackathon_health.xlsx

df_health=pd.read_excel(r'../input/isb-hackathon-dataset/Hackathon_health.xlsx',sheet_name='%data')
df_health.head()

df_dep_health=pd.read_excel(r'../input/isb-hackathon-dataset/Hackathon_health.xlsx',sheet_name='Deprivation_data')
df_dep_health.head()

df_health.columns

X_health=df_health[['chc', 'phc', 'subcentre', 'subcentre_l1km', 'subcentre_g10km',
       'closed_drain', 'pucca_slab_drain', 'pucca_uncovered_drain',
       'kuchha_drain']].values
X_health

df_dep_health.columns

X_dep_health=df_dep_health[['chc', 'phc', 'subcentre', 'subcentre_l1km', 'subcentre_g10km',
       'closed_drain', 'pucca_slab_drain', 'pucca_uncovered_drain',
       'kuchha_drain']].values
X_dep_health

from sklearn.decomposition import PCA

pca = PCA().fit(X_dep_health)

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')

plt.show()

## Select Number of PCA compnents based on the above graph
pca_dep_health = PCA(n_components = 3).fit(X_dep_health).transform(X_dep_health)
pca_perc_health = PCA(n_components = 4).fit(X_health).transform(X_health)


inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(pca_dep_health)
    kmeanModel.fit(pca_dep_health)
    inertia.append(kmeanModel.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


  kmeans = KMeans(n_clusters=2, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))
  kmeans = KMeans(n_clusters=3, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))
  kmeans = KMeans(n_clusters=4, random_state=10).fit(pca_dep)
  print('Silhouette Score: ',metrics.silhouette_score(pca_dep,kmeans.labels_))

#Choose number of clusters
kmeans = KMeans(n_clusters=3, random_state=1).fit(pca_dep_health)

#Attach cluster to main dataset
labelsKM_PCA = pd.DataFrame(kmeans.labels_)
labelled_KM_PCA = pd.concat((df_dep_health,labelsKM_PCA),axis=1)
labelled_KM_PCA = labelled_KM_PCA.rename({0:'labels'},axis=1)
labelled_KM_PCA['labels'].value_counts()


labelled_KM_PCA.head()


healthinfra_df = pd.merge(df_health,labelled_KM_PCA[['District_code','labels','Dep_Index']],on='District_code').sort_values(by='Dep_Index')
healthinfra_df.head()


sns.lmplot(x='pucca_slab_drain',y='kuchha_drain',data=healthinfra_df,hue='labels',fit_reg=False)
