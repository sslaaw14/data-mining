import numpy as np
import matplotlib.pyplot as plt
import math

inputN = input("Jumlah kota = ");
inputmaxIter = input("Maximal Iterasi = ");
N = int(inputN);
maxIter = int(inputmaxIter);

x = np.random.randint(1,N);
y = np.random.randint(1,N);
jalur = np.random.permutation(N);
#print(N);
#print(maxIter);
#print(x);
#print(y);
#print(jalur);
jarak = 0;
for i in range(1,N-1) :
    jalurX = ((x * jalur[i+1]) - (x * jalur[i]))**2;
    jalurY = ((y * jalur[i+1]) - (y * jalur[i]))**2;
    #print(jalurX);
    #print(jalurY);
    d = math.sqrt(jalurX+jalurY);
    jarak = jarak + d;
print("=====JARAK PERTAMA=====")
print(jarak);
d = math.sqrt(((x * jalur[N-1]) - (x * jalur[0]))**2 + ((y * jalur[N-1]) - (y * jalur[0]))**2);
jarak = jarak + d;
print("=====JARAK KEDUA=====")
print(jarak);
jalur_min = jalur;
jarak_min = jarak;

for i in range(0, maxIter) :
    jalur_lama = jalur;

