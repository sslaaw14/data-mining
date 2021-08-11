import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

print("Nomor 1")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
#print(dataSet)

print("Nomor 2")
test_dataset = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_test.csv")
#print(test_dataset)

print("Nomor 3")
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
print(train_data)

print("\nNomor 4")
test_data = test_dataset.loc[:,['Sex','Age','Pclass','Fare']]
emptyAge = test_data[test_data['Age'].isnull()].index.tolist()
emptyFare = test_data[test_data['Fare'].isnull()].index.tolist()
test_data = test_data.drop(emptyAge)
test_data = test_data.drop(emptyFare)
test_data = test_data.replace({'Sex' : {"female": 1, "male": 0}})
print(test_data)

print("\nNomor 5")
train_label = dataSet.loc[:,['Survived']]
print(train_label)

print("\nNomor 6")
test_label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_testlabel.csv")
test_label = test_label.drop(['PassengerId'], axis=1)
test_label = test_label.drop(emptyAge)
test_label = test_label.drop(emptyFare)
print(test_label)

print("\nNomor 7")
print("========== Kategorikal Train Data ==========")
train_data['Ageint'] = train_data['Age'].astype(int)
train_data['Fareint'] = train_data['Fare'].astype(int)
for (x,item) in train_data.iterrows():
    if item['Ageint'] < 15 :
        train_data.loc[x, 'Age'] = 0
    if item['Ageint'] >= 15 and item['Ageint'] < 40 :
        train_data.loc[x, 'Age'] = 1
    if item['Ageint'] >= 40 and item['Ageint'] < 60 :
        train_data.loc[x, 'Age'] = 2
    if item['Ageint'] >= 60 :
        train_data.loc[x, 'Age'] = 3
for (x,item) in train_data.iterrows() :
    if item['Fareint'] <= 20 :
        train_data.loc[x, 'Fare'] = 0
    if item['Fareint'] > 20 and item['Fareint'] <=60 :
        train_data.loc[x, 'Fare'] = 1
    if item['Fareint'] > 60 and item['Fareint'] <=90 :
        train_data.loc[x, 'Fare'] = 2
    if item['Fareint'] > 90  :
        train_data.loc[x, 'Fare'] = 3
train_data = train_data.drop(['Ageint'], axis=1)
train_data = train_data.drop(['Fareint'], axis=1)
print(train_data)
print("========== Kategorikal Test Data ==========")
test_data['Ageint'] = test_data['Age'].astype(int)
test_data['Fareint'] = test_data['Fare'].astype(int)
for(x,itemtest) in test_data.iterrows() :
    if itemtest['Ageint'] < 15 :
        test_data.loc[x,'Age'] = 0
    if itemtest['Ageint'] >= 15 and itemtest['Ageint'] < 40 :
        test_data.loc[x, 'Age'] = 1
    if itemtest['Ageint'] >= 40 and itemtest['Ageint'] < 60 :
        test_data.loc[x, 'Age'] = 2
    if itemtest['Ageint'] >= 60 :
        test_data.loc[x, 'Age'] = 3
for(x,itemtest) in test_data.iterrows() :
    if itemtest['Fareint'] <= 20 :
        test_data.loc[x,'Fare'] = 0
    if itemtest['Fareint'] > 20 and itemtest['Fareint'] <= 60 :
        test_data.loc[x, 'Fare'] = 1
    if itemtest['Fareint'] > 60 and itemtest['Fareint'] <= 90 :
        test_data.loc[x, 'Fare'] = 2
    if itemtest['Fareint'] > 90 :
        test_data.loc[x, 'Fare'] = 3
test_data = test_data.drop(['Ageint'], axis = 1)
test_data = test_data.drop(['Fareint'], axis = 1)
print(test_data)

print("\nNomor 8")
#train_data['Sex'],_ = pd.factorize(train_data['Sex'])
#train_data['Age'],_ = pd.factorize(train_data['Age'])
#train_data['Pclass'],_ = pd.factorize(train_data['Pclass'])
#train_data['Fare'],_ = pd.factorize(train_data['Fare'])
#test_data['Sex'],_ = pd.factorize(test_data['Sex'])
#test_data['Age'],_ = pd.factorize(test_data['Age'])
#test_data['Pclass'],_ = pd.factorize(test_data['Pclass'])
#test_data['Fare'],_ = pd.factorize(test_data['Fare'])
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(train_data,train_label)
y_pred = clf.predict(test_data)
y_pred = y_pred.reshape(331,1)
count_misclassified = (test_label != y_pred).sum()
print("Jumlah Error : ",count_misclassified)
print("Error : ",count_misclassified/331 * 100)

print("\nNomor 9")
dot_data = StringIO()
export_graphviz(clf, out_file= dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names =['Sex','Age','Pclass','Fare'],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

