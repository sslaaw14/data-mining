import pandas as pd
import math
import numpy as np

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
dataFrame = pd.DataFrame(dataSet)
print("Jumlah baris : ", dataFrame.shape[0])
print("Jumlah kolom : ", len(dataFrame.columns))

print("\nNomor 3")
data = dataSet.loc[:,['Age','Fare']]
print(data)

print("\nNomor 4")
classurvived = dataSet.loc[:,['Survived']]
print(classurvived)

print("\nNomor 5")
mean = dataSet[['Age','Survived']].groupby(['Survived']).mean()
print(mean)
for (x,item) in dataSet.iterrows():
    if item['Survived'] == 0 and math.isnan(item['Age']):
        data.loc[x,'Age'] = mean.loc[0,'Age']

for (x,item) in dataSet.iterrows():
    if item['Survived'] == 1 and math.isnan(item['Age']):
        data.loc[x,'Age'] = mean.loc[1,'Age']
print(data)

print("\nNomor 10 - Z-Score")
meanAge = np.mean(data['Age'])
meanFare = np.mean(data['Fare'])
print("Mean Age : ",meanAge)
print("Mean  Fare : ",meanFare)
for (y,zscore) in data.iterrows() :
    data.loc[x,'Age'] = zscore.loc[x,'Age']