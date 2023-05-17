## This example shows how convnet deals with MNIST dataset. Convnet is a method created for image classification
## and works much better than the usual Dense layer configuration we used in Chapter 1 (It gives a accuracy of 99%!!)

## The reason for this is because the dense layers tend to see the average patters in the image, which convnet is trained
## to see the local absolute patters. I will show how that happens through the code. For a deeper understanding, I have
## linked the videos that showcase the logic and the mathematics!

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

## These all are the usual boring setting up measures we have seen before. Just a formality, ignore this.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## <Fun Stuff>

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
## Now it's important to understand what Conv2D actually does. It checks for the local smaller patterns that emerge in
## the code. You can say it is a magnifying glass that scans the image line by line, and as it goes through the picture
## it keeps track of all the edges, valleys or any other patterns. You can use it to check for bright areas, make images
## blurred or basically anything. The way it's much better than the conventional Dense layer is that Dense layers check
## the average patterns. Dense layers view the 'bigger picture', and thus they are not specialized for local patterns.
## That requires much more computational energy; and thus, time!
## If you wich to understand convolution exactly, checkout 3Blue1Brown's video on this, it's amazingly described!

## (Ps. this is amazingly useful in probability and statistics too!)

model.add(layers.MaxPooling2D((2, 2)))
## Another important concept here! MaxPooling is very very important in any convnet situation. With convolution doing
## lots of analysis and there being loads of information, we need a way to compress this, so we can find the general
## patterns and move smoothly. Max Pooling checks a matrix of vectors (let's say 3 x 3 from the top left of the image)
## and then returns the maximum value of it. So if you have [[1, 2, 3], [0, 4, 6], [10, 5, 7]], we will have 10 as a result.
## This is a very simpified understanding. Basically maxpool lowers our required computation and gets the work done well!
## deeplizard's video gives some useful insight into it!

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
## We need to process the output through Dense layers, to get the final answer. However, we need a 1D tensor as the
## input of the Dense layer. However, we get a 3D output from Conv2D (if you need to know how that happens, see the videos
## I have linked for convolution). Hence, we use flatten function.

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
## </Fun Stuff>

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,
          train_labels,
          epochs=5,
          batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc) ## We get a accuracy of 99%!!!
