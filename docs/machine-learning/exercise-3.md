# Exercise 3

## Instructions

The aim of the exercise is to create an image classification model (CNN) in order to classify ...
cats & dogs image ! (It's a binary classification problem)

You can download the dataset [here](https://we.tl/t-Cljse6tHAo)


The folder has the following architecture :

```
cats_and_dogs
├── train
│   ├── cats
│   └── dogs
└── test
    ├── cats
    └── dogs

```

### 1 - Exploration

Before to deep dive into the convolutional neural network, have a look to the data.
You must count the number of images per class (cats / dogs). Also, plot some images to ensure you are
able to read them properly.

### 2 - Install the necessary libraries
!!! info

    If you use google colab, installations are probably already done in the environment you are using.

If you plan to develop on your local system, installed the necessary libraries : 

- tensorflow
- matplotlib
- numpy


!!! tip "Best practice"
    
    It's a best practice to create a virtual environment when starting a new project. Then, you will install your 
    dependencies in this environment (only). You can do it as follows : 
    
    * `python3 -m venv env` Create a virtual environment
    * `source env/bin/activate` Activate the environment
    * `pip install your_package` Install a package

    

> We will use Keras, which is included in tensorflow

### 3 - Split your data into train and validation
As for other data we have already manipulated, we need to split our data into train & test.
The good news is that keras comes with some built-in preprocessing functions that help you to perform this operation.
It also allows you to perform some data augmentation at the same time.

You can have a look to the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)

Here is an example (only for train split):
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'FOLDER/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

!!! warning
    
    You also need to do it for test! However, you don't need to add some data augmentation for test set
    

### 4 - Build & Train your Model
Let's build a model, the goal is to test different architecture (add or remove some layer), varying hyperparameter etc...

A very simple model would be:

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

```
> NB : The code here is only building the model, you need to search how to train it (maybe there is a fit method...)

You can test the previous architecture, but you'll probably have very poor results. Try to build a more complex architecture 
(with more Conv2D / MaxPooling) layer succession. 
Also try to add some Dropout (look [here](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout))
and change the optimizer (you can also change the maxpooling kernel size etc...)


### 5 - Evaluate your model
As mentioned in the course, you need to evaluate your cnn model as for any other model. As it's a classification problem,
you, provide a confusion matrix. In addition, plot the curve representing accuracy/loss evolution over the epochs

Very important : Download some images from internet, and test your model !

### 6 - Publish your code on Github
