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
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic.csv")
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows',10)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
test_dataset = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_test.csv")
print(test_dataset)

print("\nNomor 3")
train_data = dataSet.loc[:,['Sex','Age','Pclass','Fare','Survived']]
mean = dataSet[['Age','Survived']].groupby(['Survived']).mean()
print("Nilai Mean :\n",mean)
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
print(test_data)

print("\nNomor 5")
train_label = dataSet.loc[:,['Survived']]
print(train_label)

print("\nNomor 6")
test_label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/titanic_testlabel.csv")
test_label = test_label.drop(['PassengerId'], axis=1)
print(test_label)

print("\nNomor 7")
train_data['Sex'],_ = pd.factorize(train_data['Sex'])
train_data['Age'],_ = pd.factorize(train_data['Age'])
train_data['Pclass'],_ = pd.factorize(train_data['Pclass'])
train_data['Fare'],_ = pd.factorize(train_data['Fare'])
test_data['Sex'],_ = pd.factorize(test_data['Sex'])
test_data['Age'],_ = pd.factorize(test_data['Age'])
test_data['Pclass'],_ = pd.factorize(test_data['Pclass'])
test_data['Fare'],_ = pd.factorize(test_data['Fare'])
print("=====Train Data=====")
print(train_data)
print("=====Test Data=====")
print(test_data)
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf = clf.fit(train_data,train_label)
y_pred = clf.predict(test_data)
y_pred = y_pred.reshape(418,1)
count_misclassified = (test_label != y_pred).sum()
print("Jumlah Error : ",count_misclassified)
print("Error : ",count_misclassified/418 * 100)

print("\nNomor 9")
dot_data = StringIO()
export_graphviz(clf, out_file= dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names =['Sex','Age','Pclass','Fare'],class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes2.png')
Image(graph.create_png())