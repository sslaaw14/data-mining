import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pylab as plt

#preprocessing
test_image = image.load_img(r"F:\Kuliah\Semester 6\Data Mining\Luas Area\bali.jpg", color_mode = 'grayscale')
test_image = image.img_to_array(test_image)
plt.imshow(test_image)
image_shape = test_image.shape

#random titik
import random

in_picture = 0
out_picture = 0
for i in range(1000):
    x = random.randrange(0, image_shape[0])
    y = random.randrange(0, image_shape[1])

    if (test_image[x][y] != 255):
        in_picture += 1
        plt.scatter(y, x, color="Red")

    else:
        out_picture += 1
        plt.scatter(y, x, color="Blue")

print("titik yang terdeteksi didalam gambar = ", in_picture)
print("titik yang terdeteksi diluar gambar = ", out_picture)

#calculate
area = ((in_picture) / (in_picture + out_picture)) * (image_shape[0] * image_shape[1])
print("Maka luas areanya adalah ", area)

plt.show()

