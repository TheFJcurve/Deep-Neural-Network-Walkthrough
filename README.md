# DeepLearningWithPython
I believe that mathematics and programming go hand in hand, and even diehard programmers should have mathematical intuition of what they are working on. I will try to boil down the basics of maths so that we can have the 'Ohh I see' moment without the yawn. It's not magic, it's just linear algebra!

# Chapter 1 and 2: 
These showcase the very rudimentary logic behind neural networks. Even though further chapters will explain the mathematics behind the system, I suggest watching this series just after you finish the chapters. It seems like magic, but it's simple math! And that is more mind blowing. (Now talk about application of maths!)

. 3Blue1Brown's seires on **neural networks** (It's one of the best you can have access to for free!): [(Watch the playlist!)](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

I have also added the MNIST dataset's code, with my own annotations, for anyone who is stuck or wants a clearer understanding. I am sure you would find much better videos on this. This is just my piece of mind.

_MNIST_dataset_example(chapter 1).py_

# Chapter 3: 
While going through the textbook, I noticed that some codes were not functioning. So I have corrected and implemented them here. As I get more advanced in the field, I will create a guide for begineers to understand the concepts of deep learning.

StatQuest with Josh Starmer has a good video explaining what **cross entropy** actually is (BAM!!!): [(Watch the video!)](https://youtu.be/6ArSys5qHAU)

_IMDB_Example.py
Reuters_Example.py
Boston_Housing_Example.py_

# Chapter 4:
 
. I found this video to be good for explaination of **cross validation** (in K-fold example) [ML Fundamentals: Cross Validation by StatQuest with Josh Starmer]: [(Watch the video!)](https://youtu.be/fSytzGwwBVw)

# Chapter 5:
I felt that the learning of Convnet was highly theoritical in the book, so I am linking some videos with basic to deep understanding of the logic behind the network, in an easy to digest form. I have also written the code for the basic convnet code over MNIST dataset!

. **Convolutional Neural Networks (CNNs)** Explained [By deeplizard]: [(Watch the video!)](https://youtu.be/YRhxdVk_sIs)

. If you wish to understand the mathematics behind **convolution**, here is another amazing video by 3Blue1Brown!: [(Watch the video!)](https://youtu.be/KuXjwB4LzSA)

. **Data Augmentation** Explained (By deeplizard): [(Watch the video!)](https://youtu.be/rfM4DaLTkMs)

. I will give you a little explaination of **Dropouts**. Randomly, a percentage of the 'neurons' in each layer are deactivated, and the model is trained on the remaining active 'neurons'. This makes sure that the 'neurons' don't rely on just one preceeding neuron. Imagine it like this, if you can take money from 4 people, but at any time any one can default out, then you will make sure that you will not fully rely on any singular person for money. Similarly, this practice makes sure that the 'neuron' has significant connections with the preceeding 'neurons'. That way, any overfitting (or as the creater called it 'conspiracies') is avoided.

_MNIST_dataset_example(using basic convnet).py
dog_cat_kaggle_convnet_preparring_data.py
dog_cat_kaggle_convnet.py
dog_cat_kaggle_vgg16.py_



 CURRENT UPDATE: The codes are up to Part 2 Chapter 5 level. They have my own annotations as simple understandings. They are the most basic regression, classification problems and basic convnets.
