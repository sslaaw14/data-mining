import os
import io
import nltk
import string
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

fileStopList = open("F:/Kuliah/Semester 6/Data Mining/textmining/stoplist.txt")
line = fileStopList.read()
more_words = line.split()
stop_factory = StopWordRemoverFactory().get_stop_words()
stoplist = stop_factory + more_words
factory = StemmerFactory()
stemmer = factory.create_stemmer()

print("Nomor 1")
data = []
work_dir = "F:/Kuliah/Semester 6/Data Mining/textmining/news_dataset"
for index in range(1, 51):
    name = "data{index}.txt".format(index=index)
    path = os.path.join(work_dir, name)
    with io.open(path, mode="r", errors='ignore') as fd:
        content = fd.read()
        data.append(content)
    print("DATA ", index, "BERHASIL DITAMBAHKAN")

print("\nNomor 4")
phrase = "pertumbuhan ekonomi, perkembangan pasar dan pergerakan harga saham"
case_folding = phrase.translate(str.maketrans('','',string.punctuation)).lower().strip()
phrase_tokenizing = nltk.tokenize.word_tokenize(case_folding)
print("QUERY TOKENIZING\n", phrase_tokenizing)
phrase_filtering = []
phrase_stemming = []
for p in phrase_tokenizing :
    if p not in stoplist :
        phrase_filtering.append(p)
        phrase_stemming.append(stemmer.stem(p))
print("QUERY FILTERING\n", phrase_filtering)
print("QUERY STEMMING\n", phrase_stemming)

print("\nNomor 2 & 3")
docs = []
i = 0
for kal in data :
    print("Data ke - ", i+1)
    kalimat = kal.translate(str.maketrans('','',string.punctuation)).lower().strip()
    tokens = nltk.tokenize.word_tokenize(kalimat)
    removed = []
    for r in tokens :
        if r not in stoplist :
            removed.append(r)
    keyword = []
    for s in removed :
        keyword.append(stemmer.stem(s))
    print("STEMMING\n",keyword)
    score = []
    kemunculan = nltk.FreqDist(keyword)
    all_values = kemunculan.values()
    max_value = max(all_values)
    med = max_value / 2
    for key, value in kemunculan.items() :
        new_dict = {}
        if value > med :
            new_dict[key] = value
            score.append(new_dict)
    print("HASIL TF \n", score)
    sum = 0
    for r in score :
        for key,value in r.items() :
            if key in phrase_stemming :
                sum = sum + value
    docs.append([sum, i+1])
    i = i+1
print(docs)
docs = pd.DataFrame(docs)
rank = docs.sort_values(0, ascending= False)
rank_docs = rank[:10]
rank_docs = rank_docs.rename(columns={0: "Total", 1: "FileName"})
rank_docs = rank_docs.replace({'FileName': {3: "data3", 1: "data1", 6: "data6", 47: "data47", 2: "data2", 44: "data44", 42:"data42",
                                9: "data9", 5: "data5", 8: "data8"}}).reset_index()
rank_docs = rank_docs.drop(['index'], axis = 1)
print(rank_docs)

print("\nNomor 6")
label = pd.read_csv("F:/Kuliah/Semester 6/Data Mining/textmining/label.csv")
print(label)

print("\nNomor 7")
label_target = label[:10]
Category = ['ekonomi','ekonomi','ekonomi','pariwisata','ekonomi','pariwisata','pariwisata','ekonomi','ekonomi','ekonomi']
rank_docs['Category'] = Category
print(rank_docs)
print(label_target)
precision = []
recall = []
for j in range (0,10) :
    print("RANK ",j+1)
    nyata = label_target[:j+1]
    prediksi = rank_docs[:j+1]
    prediksi['target'] = nyata['Category']
    sama = 0
    beda = 0
    for (x,item) in prediksi.iterrows() :
        if item['Category'] == item['target'] :
            sama = sama + 1
        else :
            beda = beda + 1
    hitung_precision = sama / (sama + beda)
    hitung_recall = sama / 10
    precision.append(hitung_precision)
    recall.append(hitung_recall)
print("Nilai Precision : ", precision)
print("Nilai Recall : ", recall)

print("\nNomor 8")
import seaborn as sns
from matplotlib import pyplot as plt
data_plot = pd.DataFrame({"precision":precision, "recall":recall})
sns.lineplot(x = "recall", y = "precision", data=data_plot)
plt.show()




