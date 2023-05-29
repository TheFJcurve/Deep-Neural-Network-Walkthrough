## This file is rather unconventional with respect to other codes, we are not making any models here. Rather, it is
## taking images from your directory and using them as training, validation or testing data set. Here we have kaggle's
## annotated pictures in your drive (let's say at location LOCATION), and we are taking those images and creating 1000
## training samples, 500 validation samples and 500 testing samples (for cats and dogs individually).

## Go to https://www.kaggle.com/c/dogs-vs-cats/data , sign in/up and then click on download all. This will download
## all the image files that have been labelled (cat and dog pictures)

import os, shutil

## Input your own location where you have stored the pictures from kaggle's website.
original_dataset_dir = r'Input your path'

## This is a location where you will store your training, validation and testing images. So it's your free will to decide
## any location as such.
base_dir = r'Input your path'
if not os.path.exists(base_dir): os.mkdir(base_dir)

## Uses your base directory that you just created, and creates training, validation, and testing directories. I have
## modified the code so that you can run it multiple times even if the directory has been created before!
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir): os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir): os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir): os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir): os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir): os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir): os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir): os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir): os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir): os.mkdir(test_dogs_dir)

## This segment of code takes the picture from the directory where you have stored the kaggle images, and copies
## 1000 pictures for training, anothr 500 for validation, and another 500 for testing (for each: cats and dogs)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

## Just for sanity check to see how many pictures are where. You can manually change the size of each directory as you
## wish, it will reflect in these print images.
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir    )))
print('total test dog images:', len(os.listdir(test_dogs_dir)))