import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

print("Nomor 1")
dataSet = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/transaction.csv")
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows',12)
pd.set_option('display.width', 1000)
print(dataSet)

print("\nNomor 2")
data = dataSet.loc[dataSet['Country'] == 'Portugal', ['InvoiceNo','StockCode','Qty','InvoiceDate','CustomerID','Country']]
print(data)

print("\nNomor 3")
transaksi = data.groupby(['InvoiceNo', 'StockCode'])['Qty'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
def encode_units(x) :
    if x <= 0 :
        return 0
    if x >= 1 :
        return 1
transaksi = transaksi.applymap(encode_units)
print(transaksi)

print("\nNomor 4")
#ALGORITMA APRIOR#
frequent_itemsets = apriori(transaksi, min_support=0.2, use_colnames=True)
print(frequent_itemsets)
#ASSOCIATION RULE
associationrule = association_rules(frequent_itemsets, metric="confidence")
hasil = associationrule[associationrule['confidence'] >= 0.7]
print(hasil)

