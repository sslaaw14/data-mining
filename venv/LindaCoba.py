import pandas as pd
from sklearn.cluster import KMeans

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/transaction.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',100)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
country = dataSet.pivot_table(index=['Country'], aggfunc='size')
print(country)

print("\nNomor 3")
transaksi = dataSet.groupby(['Country', 'InvoiceNo']).sum()[['Qty']].reset_index()
print(transaksi)
transaksi = transaksi.groupby('Country').mean()[['Qty']]
print(transaksi)

print("\nNomor 4")
cluster_i = []
cluster_val = []
for i in range(10) :
    kmeans = KMeans(n_clusters=3, init='random',n_init=1).fit(transaksi)
    label = kmeans.labels_
    cluster_i.append(label)
    cluster_val.append(kmeans.inertia_ / len(transaksi))
print("\n=====HASIL CLUSTERING=====")
print(cluster_i)
print("\n===== HASIL CLUSTER ANALYSIS (SSE)=====")
print(cluster_val)

print("\nNomor 5")
min_index = cluster_val.index(min(cluster_val))
cluster = cluster_i[min_index]
print("Minimal Value Cluster_Val ", min(cluster_val) , " pada index ", min_index)
print(cluster)

print("\nNomor 6")
from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(transaksi, cluster)
centroid = clf.centroids_
print(centroid)
transaksi['label'] = cluster
transaksi = transaksi.reset_index()
for (i,item) in transaksi.iterrows() :
    if item['label'] == 0 :
        transaksi.loc[i,'centroid'] = centroid[0]
    if item['label'] == 1 :
        transaksi.loc[i,'centroid'] = centroid[1]
    if item['label'] == 2 :
        transaksi.loc[i,'centroid'] = centroid[2]
print(transaksi)

print("\nNomor 7")
import numpy as np
sort = centroid[np.argsort(centroid[:,0])]
sort_transaksi = transaksi.sort_values(['centroid'])
print(sort)
print(sort_transaksi)

print("\nNomor 8")
