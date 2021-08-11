import numpy as np
import matplotlib.pyplot as plt

data = np.random.poisson(4,1000);
#print(data);
jam = 8;
menit = 0;
for j in range(0,100) :
    menit = menit + data[j];
    if menit >= 60 :
        menit = menit - 60;
        jam = jam + 1;
    print("Mobil ke - ", j+1, "datang pukul ", jam,":",menit);

plt.title("2110171004-Mayshella A.W");
plt.hist(data);
plt.show();
