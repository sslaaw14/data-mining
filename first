import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier

#print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
#print(dataSet)

#print("Nomor 2")
test_dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_test.csv")
#print(test_dataSet)

#print("Nomor 3")
train_data = dataSet.loc[:,['Sex','Age','Pclass','Fare','Survived']]
train_data = train_data.replace({'Sex' : {"female":1, "male":0}})
mean = dataSet[['Age','Survived']].groupby(['Survived']).mean()
#print(mean)
for (x,item) in train_data.iterrows():
    if item['Survived'] == 0 and math.isnan(item['Age']):
        train_data.loc[x,'Age'] = mean.loc[0,'Age']

for (x,item) in train_data.iterrows():
    if item['Survived'] == 1 and math.isnan(item['Age']):
        train_data.loc[x,'Age'] = mean.loc[1,'Age']
train_data = train_data.drop(['Survived'], axis=1)
#print(train_data)

#print("Nomor 4")
test_data = test_dataSet.loc[:,['Sex','Age','Pclass','Fare']]
emptyAge = test_data[test_data['Age'].isnull()].index.tolist()
emptyFare = test_data[test_data['Fare'].isnull()].index.tolist()
test_data = test_data.drop(emptyAge)
test_data = test_data.drop(emptyFare)
test_data = test_data.replace({'Sex' : {"female": 1, "male": 0}})
#print(test_data)

#print("Nomor 5")
train_label = dataSet.loc[:, ['Survived']]
#print(train_label)

#print("Nomor 6")
test_label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_testlabel.csv")
test_label = test_label.drop(['PassengerId'], axis=1)
test_label = test_label.drop(emptyAge)
test_label = test_label.drop(emptyFare)
#print(test_label)

#print("Nomor 7")
newMin = 0
newMax = 1
minValue = train_data.min()
maxValue = train_data.max()
train_data = (train_data-train_data.min())*(newMax-newMin)/(train_data.max()-train_data.min())+newMin
#print(train_data)

print("Nomor 8")
test_data = (test_data-minValue)*(newMax-newMin)/(maxValue-minValue)+newMin
print(test_data)

print("Nomor 9")
for x in range(1,11):
    kNN = KNeighborsClassifier(n_neighbors=x, weights='distance')
    kNN.fit(train_data, train_label.values.ravel())
    error = kNN.score(test_data, test_label)
    print("Error K-",x," = ",1 - error)

print("Error terkecil adalah K dengan Nilai 10")
