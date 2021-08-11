import numpy as np
from numpy import random
import matplotlib.pyplot as plt
data = np.random.poisson(2,256);
jam = 8;
menit = 0;
for j in range(0,256) :
     menit = menit + data[j];
     if menit >= 60 :
         menit = menit - 60;
         jam = jam + 1;
     jumlahBarang = random.uniform(1,8);
     print("Truck kontainer ", j+1, "datang pada pukul ", jam, ":", menit, "dengan jumlah barang : ", int(jumlahBarang));

plt.hist(data);
plt.show();
