#import matplotlib.pyplot as plt

#fig,ax=plt.subplots()
#ax.plot([1,2,3,4],[1,4,2,5])
#plt.ylabel('some numbers')
#plt.savefig('myplot.png')

from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset, classes=make_blobs(n_samples=200,n_features=2,centers=4, cluster_std=0.5, random_state=0)
df=pd.DataFrame(dataset, columns=['var1','var2'])
print(df.head(2))

inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(df).inertia_ for k in range(1,12)]
plt.plot(range(1,12), inertias, 'bo-')
plt.savefig('elbow.png') 

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0).fit(df)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
from collections import Counter
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))
