import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
data = dataSet.loc[:,['Name', 'Sex','Age','Pclass','Fare']]
print(data)

print("\nNomor 4")
classurvived = dataSet.loc[:,['Survived']]
print(classurvived)

print("\nNomor 5")
data['Realtives'] = dataSet['SibSp'] + dataSet['Parch']
print(data)

print("\nNomor 6")
pclass1=0
pclass2=0
pclass3=0
for(x,pclass) in data.iterrows() :
    if pclass['Pclass'] == 1 :
        pclass1 = pclass1 +1
    if pclass['Pclass'] == 2 :
        pclass2 = pclass2+1
    if pclass['Pclass'] == 3 :
        pclass3 = pclass3 + 1
print("Jumlah Pclass 1 : ",pclass1,"\nJumlah Pclass 2 : ",pclass2,"\nJumlah Pclass 3 : ",pclass3)

print("\nNomor 7")
male = 0
female = 0
for(y,sexsum) in data.iterrows() :
    if sexsum['Sex'] == "female" :
        female = female + 1
    if sexsum['Sex'] == "male" :
        male = male + 1
print("Jumlah Female : ",female, "\nJumlah Male :",male)

print("\nNomor 8")
jumlahSurvived = dataFrame.groupby(['Survived', 'Pclass'])['Pclass'].count()
print(jumlahSurvived)

print("\nNomor 9")
sexsurvived = dataFrame.groupby(['Survived', 'Sex'])['Survived'].count()
# create plot
fig, ax = plt.subplots()
index = np.arange(2)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, sexsurvived[0], bar_width,
alpha=opacity,
color='b',
label='Dead')
rects2 = plt.bar(index + bar_width, sexsurvived[1], bar_width,
alpha=opacity,
color='g',
label='Survived')
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by person')
plt.xticks(index , ('Female', 'Male'))
plt.legend()

plt.tight_layout()
plt.show()

print("\nNomor 10")
agesurvived = dataFrame.groupby(['Survived', 'Age'])['Age'].count()
print(agesurvived)


