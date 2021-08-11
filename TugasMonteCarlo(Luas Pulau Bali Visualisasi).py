import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pylab as plt

#preprocessing
test_image = image.load_img(r'F:\Kuliah\Semester 6\Data Mining\Luas Area\bali.jpg', color_mode = 'grayscale')
test_image = image.img_to_array(test_image)
image_shape = test_image.shape

#visualisasi random titik
import turtle

image_gif = r'F:\Kuliah\Semester 6\Data Mining\Luas Area\bali gif.gif'
screen = turtle.Screen()
screen.setup(image_shape[0], image_shape[1])

screen.addshape(image_gif)
turtle.shape(image_gif)

myDot = turtle.Turtle()
myDot.hideturtle()
myDot.speed(0)
#random titik
import random

in_picture = 0
out_picture = 0
for i in range(5):
    for j in range(1000):
        x = random.randrange(0, image_shape[0])
        y = random.randrange(0, image_shape[1])

        if(test_image[x][y] != 255.):
            in_picture += 1
            myDot.color("red")
            myDot.up()
            myDot.goto(x,y)
            myDot.down()
            myDot.dot()

        else:
            out_picture += 1
            myDot.color("black")
            myDot.up()
            myDot.goto(x, y)
            myDot.down()
            myDot.dot()

print(in_picture)
print(out_picture)

#calculate
area = ((in_picture) / (in_picture + out_picture)) * (image_shape[0] * image_shape[1])
print(area)

#plot data
plt.imshow(test_image)
plt.show()

