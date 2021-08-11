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
transaksi['label'] = cluster
transaksi = transaksi.reset_index()
centroid = transaksi.groupby('label').mean()
print(centroid)
for (i,item) in transaksi.iterrows() :
    if item['label'] == 0 :
        transaksi.loc[i,'centroid'] = centroid.loc[0,'Qty']
    if item['label'] == 1 :
        transaksi.loc[i,'centroid'] = centroid.loc[1,'Qty']
    if item['label'] == 2 :
        transaksi.loc[i,'centroid'] = centroid.loc[2,'Qty']
#print(transaksi)

print("\nNomor 7")
sorted = centroid.sort_values(['Qty'])
print(sorted)
sorted_transaksi = transaksi.sort_values(['centroid'])
#print(sorted_transaksi)

print("\nNomor 8")
import numpy as np
print("=====Transaksi Rendah=====")
print(transaksi['Country'].loc[np.where(transaksi['label'] == 1)])
print("\n=====Transaksi Sedang=====")
print(transaksi['Country'].loc[np.where(transaksi['label'] == 0)])
print("\n=====Transaksi Tinggi=====")
print(transaksi['Country'].loc[np.where(transaksi['label'] == 2)])

print("\nNomor 9")
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
low = mpatches.Patch(color='purple', label='Low')
medium = mpatches.Patch(color='red', label='Medium')
high = mpatches.Patch(color='aquamarine', label='High')
plt.xlabel('Country')
plt.scatter(transaksi['Country'], transaksi['Qty'], c=transaksi['label'], cmap='rainbow')
plt.ylabel('Transaction')
plt.legend(handles=[low,medium,high])
plt.xticks(rotation=30, horizontalalignment='right')
plt.show()

