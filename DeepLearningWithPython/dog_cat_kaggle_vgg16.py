from keras import models
from keras import layers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from dog_cat_kaggle_convnet_preparring_data import train_dir, validation_dir, test_dir
import matplotlib.pyplot as plt

conv_base = VGG16(weights='imagenet', ## Basically tells them 'A Weight distribution good for imagenet works!'
                  include_top=False, ## VGG16 has 1000 different animal names, we need only 2, so we tell it don't add all labels
                  input_shape=(150, 150, 3))

conv_base.summary() ## There are 14 million parameters!! That's a lot of processing.

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary() ## 16 million parameters!!!!!!

## Now there is a new and important concept. Freezing layers! VGG16 has already been trained to notice animal features, so it's already
## trained (Hence it is called a Pretrained Network). If we train the model again, it will be a huge waste of time, and will also 
## waste the VGG16's potential. Therefore, we tell the model to update the weights and biases for all the layers except the convolutional
## base that is VGG16. This is known as freezing. We would still train on last 2-3 layers of the convolutional base, to optimize it for 
## cats and dogs. But everything else will be frozen. That is known as fine-tuning. (The more you know!)

set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1': ## Don't freeze anything after this layer.
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False 

## Applying Data Augmentation 
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

num_epochs = 100 ## Change this at will. The default value is 100.

history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=num_epochs,
                    validation_data=validation_generator,
                    validation_steps=50)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=20,
                                                  class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:', test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()