import pandas as pd

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/transaction.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',100)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
dataSet.InvoiceDate = dataSet.InvoiceDate.apply(pd.to_datetime)
dataSet['year'] = pd.DatetimeIndex(dataSet['InvoiceDate']).year
dataSet['month'] = pd.DatetimeIndex(dataSet['InvoiceDate']).month
dataSet['date'] = pd.DatetimeIndex(dataSet['InvoiceDate']).day
data = dataSet.loc[dataSet['year'] == 2011, ['InvoiceNo', 'Country', 'month']].reset_index()
data = data.drop(['index'], axis=1)
print(data)

print("\nNomor 3 - 8")
from sklearn.cluster import AgglomerativeClustering
i = 0
transaksi_tinggi = []
for j in range(1,13) :
    print("\nBULAN ", j)
    data_bulan = data.loc[data['month'] == j, ['InvoiceNo','Country']]
    print(data_bulan)
    duplicate = data_bulan.drop_duplicates(subset=['Country', 'InvoiceNo'])
    transaksi = duplicate.groupby('Country').count()[['InvoiceNo']]
    print("\nJUMLAH TRANSAKSI")
    print(transaksi)
    print("\nCLUSTERING")
    clustering = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average")
    cluster = clustering.fit_predict(transaksi)
    print(cluster)
    print("\nCENTROID")
    transaksi['label'] = cluster
    transaksi = transaksi.reset_index()
    centroid = transaksi.groupby('label').mean()
    print(centroid)
    print("\nSORTED")
    sorting = centroid.sort_values(['InvoiceNo']).reset_index()
    print(sorting)
    print("\nTRANSAKSI TINGGI")
    transaksi_tinggi.append(sorting.loc[2, 'InvoiceNo'])
    print("Centroid Negara Transaksi Tinggi : ", transaksi_tinggi[i])
    i = i+1

print("\nNomor 9")
import seaborn as sns
from matplotlib import pyplot as plt
x = [1,2,3,4,5,6,7,8,9,10,11,12]
ax = sns.stripplot(x, transaksi_tinggi)
ax.set(xlabel='Month', ylabel='Transaksi Tinggi')
plt.show()

print("\nNomor 10")
from sklearn import linear_model 
bulan = pd.DataFrame(x)
transaksi_tinggi = pd.DataFrame(transaksi_tinggi)
x = bulan.values.reshape(-1,1)
y = transaksi_tinggi.values.reshape(-1,1)
regresi = linear_model.LinearRegression()
regresi.fit(x,y)
print("Nilai a\t: ", regresi.intercept_)
print("Nilai b\t: ", regresi.coef_)
a = regresi.intercept_
b = regresi.coef_
predictVal = a + (b * 13)
print("Hasil Prediksi\t: ", predictVal)