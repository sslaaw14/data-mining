import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import math

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
data = dataSet.loc[:,['Sex','Age','Pclass','Fare','Survived']]
print(data)

print("\nNomor 3")
df = pd.DataFrame(data)
print("Jumlah NaN Value di Atribut Age : ", df['Age'].isnull().sum())
train_data = data.loc[data['Age'].notnull(),['Sex','Pclass','Fare','Survived']]
train_data = train_data.replace({'Sex' : {"female":1, "male":0}})
print(train_data)

print("\nNomor 4")
train_label = data.loc[data['Age'].notnull(),['Age']]
train_label = train_label.astype(int);
print(train_label)

print("\nNomor 5")
test_data = data.loc[data['Age'].isnull(),['Sex','Pclass','Fare','Survived']]
test_data = test_data.replace({'Sex' : {"female":1, "male":0}})
print(test_data)

print("\nNomor 6")
minValue = train_data.min()
maxValue = train_data.max()
newMax = 1
newMin = 0
print("\nMinimal Train Data :\n",minValue,"\nMaximal Train Data :\n",maxValue)
train_data = (train_data-train_data.min())*(newMax-newMin)/(train_data.max()-train_data.min())+newMin
print(train_data)

print("\nNomor 7")
test_data = (test_data-minValue)*(newMax-newMin)/(maxValue-minValue)+newMin
print(test_data)

print("\nNomor 8")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
#lab_enc = preprocessing.LabelEncoder()
#training_scores_encoded = lab_enc.fit_transform(train_label)
#print(training_scores_encoded)
knn.fit(train_data, train_label)
class_result = knn.predict(test_data)
print(class_result)
print(class_result.shape[0])

print("\nNomor 9")
listnull = data[data['Age'].isnull()].index.tolist()
i = 0;
for x in class_result :
    if data['Age'].isnull().any() :
        data.loc[data.Age.isnull(), 'Age'] = class_result
print(data)
#print(listnull)

print("\nNomor 10")
test_dataset = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_test.csv")
print(test_dataset)

print("\nNomor 11")
train_data = data.loc[:,['Sex','Age','Pclass','Fare']]
train_data = train_data.replace({'Sex' : {"female":1, "male":0}})
print(train_data)

print("\nNomor 12")
train_label = data.loc[:,['Survived']]
print(train_label)

print("\nNomor 13")
test_data = test_dataset.loc[:,['Sex','Age','Pclass','Fare']]
test_data = test_data.replace({'Sex' : {"female":1, "male":0}})
emptyAge = test_data[test_data['Age'].isnull()].index.tolist()
emptyFare = test_data[test_data['Fare'].isnull()].index.tolist()
test_data = test_data.drop(emptyAge)
test_data = test_data.drop(emptyFare)
print(test_data)

print("\nNomor 14")
test_label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_testlabel.csv")
test_label = test_label.drop(emptyAge)
test_label = test_label.drop(emptyFare)
test_label = test_label.drop(['PassengerId'], axis=1)
print(test_label)

print("\nNomor 15")
minBaru = train_data.min()
maxBaru = train_data.max()
print("\nMinimal Value : \n",minBaru,"Max Value : \n",maxBaru)
train_data = (train_data-train_data.min())*(newMax-newMin)/(train_data.max()-train_data.min())+newMin
print(train_data)

print("\nNomor 16")
test_data = (test_data-minBaru)*(newMax-newMin)/(maxBaru-minBaru)+newMin
print(test_data)

print("\nNomor 17")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(train_data, train_label)
class_result2 = knn.predict(test_data)
print(class_result2)

print("\nNomor 18 & 19")
error = knn.score(test_data, test_label)
print("Error : ", (1-error)*100)