## Example from Chapter 2 of deeplearningwithpython. MNIST dataset and the neural network corresponding to that is like the 
## print("Hello World") of deep learning. HEre we have used two layers (1 input layer with 512 points taking in images of 28 x 28
## and 1 final layer with 10 outputs of total [numbers ranging from 0 to 9])

from keras import models ## Helps in creating the model
from keras import layers ## Creates the layers for the neural network
from keras.datasets import mnist ## MNIST dataset in inbuilt in Keras!
from keras.utils import to_categorical ## Helps in creating our tensors' data into categorical form (since we have 10 categories)

## Loading the training images and labels, alongside the testing images and labels from MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## Our model has two layers, one with 512 items using relu activation (relu basically means max(0,num)) and one with 10 
## items with a softmax activation (softmax inserts a number in the output, where each number is between 0 and 1, and they all add upto 1)
## We have made a softmax output since we wish to know which number the AI think is most likely, least likely and everything in between.
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

## (Note: Technically your layer can be any size long, we usually set it up to be a power of 2 because it's easier for GPU allocation)

## Compiling with network with Categorical Cross Entropy as a loss function, I have linked a video that explains what Cross Entropy actually is!
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

## Reshaping our vectors. The Training set has 60,000 images of size 28 x 28 pixels, with each value a float32 between 0 and 1
## whereas Testing set has 10,000 images of the same specification. (Normally, this is the ratio between training and testing set)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, 
            train_labels, 
            epochs=5, 
            batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)