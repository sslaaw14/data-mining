import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/transaction.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
country = dataSet.pivot_table(index=['Country'], aggfunc='size')
print(country)

print("\nNomor 3")
dataSetCopy = dataSet.loc[:,['InvoiceNo','StockCode','Qty','InvoiceDate','CustomerID','Country']]
duplicate = dataSetCopy.drop_duplicates(subset=['Country','InvoiceNo'])
transaksi = duplicate.groupby('Country').count()[['InvoiceNo']]
print(transaksi)

print("\nNomor 4")
cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average")
label = cluster.fit_predict(transaksi)
print(label)

print("\nNomor 5")
transaksi['label'] = label
transaksi = transaksi.reset_index()
centroid = transaksi.groupby('label').mean()
print(centroid)
for (i,item) in transaksi.iterrows() :
    if item['label'] == 0 :
        transaksi.loc[i,'centroid'] = centroid.loc[0,'InvoiceNo']
    if item['label'] == 1 :
        transaksi.loc[i,'centroid'] = centroid.loc[1,'InvoiceNo']
    if item['label'] == 2 :
        transaksi.loc[i,'centroid'] = centroid.loc[2,'InvoiceNo']
print(transaksi)

print("\nNomor 6")
sorted = transaksi.sort_values(['centroid'])
sorted2 = centroid.sort_values(['InvoiceNo'])
print(sorted)

print("\nNomor 7")
import numpy as np

print("=====Transaksi Rendah=====")
print(sorted['Country'].loc[np.where(sorted['label'] == 0)])
print("\n=====Transaksi Sedang=====")
print(sorted['Country'].loc[np.where(sorted['label'] == 1)])
print("\n=====Transaksi Tinggi=====")
print(sorted['Country'].loc[np.where(sorted['label'] == 2)])

print("\nNomor 8")
import matplotlib.patches as mpatches
low = mpatches.Patch(color='purple', label='Low')
medium = mpatches.Patch(color='red', label='Medium')
high = mpatches.Patch(color='aquamarine', label='High')
plt.xlabel('Country')
plt.scatter(transaksi['Country'], transaksi['InvoiceNo'], c=transaksi['label'], cmap='rainbow')
plt.ylabel('Transaction')
plt.legend(handles=[low,medium,high])
plt.show()