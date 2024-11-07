from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
labels = kmeans.labels_
data_with_labels = pd.DataFrame(data=np.c_[x,labels],columns=iris.feature_names+['clusterLabel'])
plt.figure(figsize=(10,10))
plt.scatter(x[labels==0,0],x[labels==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(x[labels==1,0],x[labels==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(x[labels==2,0],x[labels==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.xlabel('kmeans clustering')
plt.legend()
plt.show()