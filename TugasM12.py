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
data = dataSet.loc[dataSet['Country'] == 'Germany',['Qty', 'month', 'year']].reset_index()
data = data.loc[data['year'] == 2011, ['Qty','month','year']]
print(data)

print("\nNomor 3")
totalQty = data.groupby('month').sum()[['Qty']].reset_index()
print(totalQty)

print("\nNomor 4")
import seaborn as sns
from matplotlib import pyplot as plt
ax = sns.stripplot(totalQty['month'], totalQty['Qty'])
ax.set(xlabel='Month', ylabel='Total Qty')
plt.show()

print("\nNomor 5")
from sklearn import linear_model
x = totalQty['month'].values.reshape(-1,1)
y = totalQty['Qty'].values.reshape(-1,1)
regresi = linear_model.LinearRegression()
regresi.fit(x,y)
print("Nilai a\t: ", regresi.intercept_)
print("Nilai b\t: ", regresi.coef_)
a = regresi.intercept_
b = regresi.coef_
prediksi = a + (b * 13)
print("Hasil Prediksi\t: ", prediksi)

