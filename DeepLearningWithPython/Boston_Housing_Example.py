## This program predicts the median price of homes in a given Boston suburb, given the other data. Hence this is
## a regression, as it predicts a continious value instrad of a discrete label.

from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def build_model():
    ## Creates a model with two relu activation layers of length 64 and one final layer of length 1. Loss function is mse
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) ## No activation since we need the price, which can take any value. Sigmoid would squish it
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def smooth_curve(points, factor=0.9):
    ## Creates a more smooth graph
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


## Loading the training and testing data and targets respectively form the sample of data. The Training data will be
## used to train the data, with the output being the training targets. We will check the accuracy of the program with
## the testing data and targets set.
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

## Normalising the data to find the characteristics of the set. We are basically making the data into a nominal
## distribution [ N(0, 1) ]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

## Since we have a low amount of training data, we are using the K fold partition method. We use this to find out the
## best possible epochs for the situation. Currently it is set to 500; however, run the program and see the minimum at the
## graph. That value is the number of epochs you should keep. It would come out to be approximately 80. 
'''
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_history = []

for i in range(k):
    print(f'processing fold #{i}')

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]],
                                        axis = 0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]],
                                           axis = 0)

    model = build_model()

    history = model.fit(partial_train_data,
                        partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs,
                        batch_size=1,
                        verbose=0) ## Trains the model in silent mode

    mae_history = history.history['val_loss']
    all_mae_history.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
smooth_mae_history = smooth_curve(average_mae_history[10:])

## Plotting validation scores
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
'''

## Training the final model
model = build_model()
model.fit(train_data,
          train_targets,
          epochs=80,
          batch_size=16,
          verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)