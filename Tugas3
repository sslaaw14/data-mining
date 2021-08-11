import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
print(dataSet)

print("Nomor 2")
test_dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_test.csv")
print(test_dataSet)

print("Nomor 3")
train_data = dataSet.loc[:,['Age','Fare']]
emptyTrainAge = train_data[train_data['Age'].isnull()].index.tolist()
train_data = train_data.drop(emptyTrainAge)
print(train_data)

print("Nomor 4")
test_data = test_dataSet.loc[:,['Age','Fare']]
emptyAge = test_data[test_data['Age'].isnull()].index.tolist()
emptyFare = test_data[test_data['Fare'].isnull()].index.tolist()
test_data = test_data.drop(emptyAge)
test_data = test_data.drop(emptyFare)
print(test_data)

print("Nomor 5")
train_label = dataSet.loc[:, ['Survived']]
train_label = train_label.drop(emptyTrainAge)
print(train_label)

print("Nomor 6")
test_label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_testlabel.csv")
test_label = test_label.drop(['PassengerId'], axis=1)
test_label = test_label.drop(emptyAge)
test_label = test_label.drop(emptyFare)
print(test_label)

print("Nomor 7")
minValue = train_data.min()
maxValue = train_data.max()
newMax = 1
newMin = 0
print(minValue,"\n",maxValue)
train_data = (train_data-train_data.min())*(newMax-newMin)/(train_data.max()-train_data.min())+newMin
print(train_data)

print("Nomor 8")
test_data = (test_data-minValue)*(newMax-newMin)/(maxValue-minValue)+newMin
print(test_data)

print("Nomor 9")
kNN = KNeighborsClassifier(n_neighbors=3,weights='distance')
kNN.fit(train_data,train_label)
class_result = kNN.predict(train_data)
print(class_result)

print("Nomor 10 & 11")
error = kNN.score(test_data,test_label)
print("Error : ", (1-error)*100)