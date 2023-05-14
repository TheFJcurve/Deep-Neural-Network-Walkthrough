## This program performs a binary crossenstropy over a classification problem. We have 1 output that we need to decide.

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

## To vectorize the sequences into valid tensors
def vectorize_sequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


## Collecting the training data and labels, and the testing data and labels
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

## Setting up the training and testing set of data
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

## Setting up the training and testting set of labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Setting up the model with three layers, two with hidden values of 16 and one with a hidden value of 1.
## We use Relu function for the first two layers, and the sigmoid function for the last one.
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

## Setting up the loss function, we have used binary_crossentropy since the final output is a percent from 0 to 1.
## This is one of the most important things in a deep learning algorithm. Everytime you run your program, a loss function
## tells you how off you are from the actual answer.
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

## Setting up and setting aside the validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

## Training the model (this is the main part most people think is magic, it's just linear algebra!). We are doing
## 20 epochs, that is, 20 iterations through the data set. We are doing those iterations over a batch size of 512
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))

## Creates a dictionary with a history of all the values, it contains the loss and the accuracy of the training set
## and the testing set (it is called training and validation set)
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

## Stores the number of epochs that were done
epochs = range(1, len(acc) + 1)

## Creates a plot given the loss value on the training set and then the loss value on the testing set
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## Creates a plot given the accurary value on the training set and then the accuracy value on the testing set
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

## Gives the prediction values of the algorithm on the testing set
print(model.predict(x_test))


