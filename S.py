from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import nltk
import string

# sastrawi
factory = StemmerFactory()
stemmerIndonesia = factory.create_stemmer()

kalimat = "[MOJOK.co] Manfaat monitoring, singing dan jogging setiap pagi yang pertama adalah meredakan stres. Olahraga itu seperti kode bagi tubuh untuk memproduksi hormon endorfin, agen perangsang rasa bahagia. Dilakukan di pagi hari, ketika udara masih bersih, sejuk, jalanan lengang, gunung terlihat jelas di sebelah utara, manfaat jogging bisa kamu rasakan secara maksimal."
kalimat = kalimat.translate(str.maketrans('', '', string.punctuation)).lower()  # menghilangkan tanda baca
case_folding = kalimat.lower()  # casefolding

tokens = nltk.tokenize.word_tokenize(case_folding)  # list
print("Hasil Tokenize\n",tokens)

kemunculan = nltk.FreqDist(tokens)
print("Hasil Kemunculan Tokenize\n",kemunculan.most_common())

listStopword = set(stopwords.words('indonesian'))
print("LIST\n", listStopword)
stemmerEnglish = PorterStemmer()

removed = []
stemmed = []
for t in tokens:
    if t not in listStopword:
        removed.append(t)
        t2 = stemmerEnglish.stem(t)  # Stem in English
        t3 = stemmerIndonesia.stem(t2)  # Stem in sastrawi
        stemmed.append(t3)

print("Hasil Filtering\n",removed) #list
print("Hasil Stemming\n",stemmed) #list
print("=========================================")
kemunculan = nltk.FreqDist(stemmed)
print("INI KEMUNCULAN ")
print(kemunculan.most_common())

mydistinct = set(stemmed)  # type SET
kemunculan = nltk.FreqDist(mydistinct)
print("KEMUNCULAN MYDISTINCT\n")
print(kemunculan.most_common())
print("INI MYDISTINCT\n")
print(mydistinct)
#####################################stemming
listdata = []
for letter in set(mydistinct):  # Convert  SET to List
    listdata.append(letter)
print("INI LIST DATA\n")
print(listdata)