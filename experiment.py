# experiment.py is my prototype of a gradual learning protocol.
# Author: Zack Gill
# Date: 6/01/17


# Using Keras and Tensor Flow to create the Neural Network
# For this example, I'm doing image recongition. In particular, character
# recongition.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, Lambda, Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K

# Numpy has many nice quality of life features for python.
import numpy as np

# skimage and Pil are used to resize, grayscale, and otherwise modify images
# to make them universal. 
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
from skimage.util import img_as_float
from PIL import Image


import random
import string
import os
import binascii
import struct

train_path = "by_class//by_class//" # Relative path


size = 28, 28 # global size, same as NIST database, used for resizing images.


def char_to_int_key(char):
    # uppercase
    if (ord(char) <= 90):
        return int((ord(char) - 55))
    # lowercase
    if(ord(char) > 90):
        return int((ord(char) - 61))

def y_to_onehot(y):
    ret = []
    for x in range(0, 62):
        ret.append(0)
    ret[y] = 1
    return ret

# prepare image will take an image file path, resize, return the
# final image as np array
def prepare_image (img_path):
    im = Image.open(img_path, 'r')
    #im = im.convert('L') # grayscale
    im.thumbnail(size, Image.ANTIALIAS)
    return np.asarray(im)

# Generates a list of training data from the numbers of the NIST dataset
# Creates a list of all images, by class. Label with one-hot encoding.
# Will train on whole batch. While not online learning, I decided to focus more on additive for this experimental.
# The report will detail this.
def generate_nums():
    x_input = []
    y_labels = []
    for rand in range(0, 9):

        path = train_path
        path = path + str(binascii.hexlify(str(rand).encode()), 'ascii')
        path = path + "//train_" + str(binascii.hexlify(str(rand).encode()), 'ascii')


        for filename in os.listdir(path):
            if filename.endswith(".png"):
                tempPath = path + '//' + filename
                img = prepare_image(tempPath)

                x_input.append(img)
                y_labels.append(y_to_onehot(rand))
    return (x_input, y_labels)



# Generates a list of training data from the characters of the NIST dataset
# Returns images and labels.
def generate_chars():
    x_input = []
    y_labels = []
    for rand in string.ascii_letters:

        path = train_path
        path = path + str(binascii.hexlify(rand.encode()), 'ascii')
        path = path + "//train_" + str(binascii.hexlify(rand.encode()), 'ascii')

        for filename in os.listdir(path):
            if filename.endswith(".png"):

                tempPath = path + '//' + filename
                img = prepare_image(tempPath)

                x_input.append(img)
                y_labels.append(y_to_onehot(char_to_int_key(rand)))

    return (x_input, y_labels)


# As generate_nums, but generates testing data. Used for validation of the model after then number training
# Generating a 1000 datapoints to test on.
def validate_nums():
    x_input = []
    y_labels = []
    for x in range(0, 1000):
        rand = random.randint(0, 9) # random int for digit selection.

        path = train_path
        path = path + str(binascii.hexlify(str(rand).encode()), 'ascii')
        path = path + "//hsf_0" # Because the data is missing a hsf_5, doing the easy test of only 1 writer.

        # Choosing a random image to use
        rand_file = random.choice(os.listdir(path))
        path = path + '//' + rand_file
        img = prepare_image(path)

        x_input.append(img)
        y_labels.append(y_to_onehot(rand))
    return (x_input, y_labels)

# As generate_chars, but generates testing data. Used for validation of the model after the character training.
# Generates more test data due to more options.
def validate_chars():
    x_input = []
    y_labels = []
    for x in range(0, 5000):
        rand = random.choice(string.ascii_letters) # random char for digit selection.

        path = train_path
        path = path + str(binascii.hexlify(rand.encode()), 'ascii')
        path = path + "//hsf_0" # Because the data is missing a hsf_5, doing the easy test of only 1 writer.

        # Choosing a random image to use
        rand_file = random.choice(os.listdir(path))
        path = path + '//' + rand_file
        img = prepare_image(path)

        x_input.append(img)
        y_labels.append(y_to_onehot(char_to_int_key(rand)))
    return (x_input, y_labels)

num_classes = 62 # 0 - 9, upper and lowercase chars are the options for the model. Easiest to have model start with that many outputs.

# Creating the Model
# Doing 1 Hidden layer of 32 nodes, starting with a 32 node layer, and ending with our output.
model = Sequential()

model.add(Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 3)))
model.add(Conv2D(32, [3, 3], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



def train_nums():


    inp = generate_nums()

    val = validate_nums()

    valX = val[0]
    valY = val[1]

    x = inp[0]
    y = inp[1]

    # Training on whole data set. Using evaluate to see if it was a good fit. For this project, would evaluate and manually decide to
    # keep or start over once program has finished. 
    model.fit(np.asarray(x), np.asarray(y), epochs=100)

    num_accuracy = (model.evaluate(np.asarray(valX), np.asarray(valY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1
    print("Number accuracy: " + str(num_accuracy))

    # Saving good number model to compare/revert to later.

    model.save("num_model")

# Training Model on characters.
def train_chars():

    inp = generate_chars()

    val = validate_chars()

    valX = val[0]
    valY = val[1]

    x = inp[0]
    y = inp[1]

    # Training on whole data set. Using evaluate to see if it was a good fit. For this project, would evaluate and manually decide to
    # keep or start over once program has finished. 
    model.fit(np.asarray(x), np.asarray(y), epochs=100)

    char_accuracy = (model.evaluate(np.asarray(valX), np.asarray(valY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1
    print("Char accuracy: " + str(char_accuracy))



# Training the model
# To do online learning, only calling 'fit' on 1 data point each time.

# Start with numbers, then chars, then compare the model to previous state.

# We will train as long as accuracy is lower than 85% with our validate_nums

train_nums()
train_chars()


# Test this model against the numbers again, see if it's accuracy is greater than the num_accuracy of before. If not, revert and start over.
val = validate_nums()
valX = val[0]
valY = val[1]
test_acc = (model.evaluate(np.asarray(valX), np.asarray(valY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1

old_model = load_model("num_model")
num_accuracy = (old_model.evaluate(np.asarray(valX), np.asarray(valY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1

while(test_acc < num_accuracy):
    val = validate_nums()
    valX = val[0]
    valY = val[1]
    model = old_model
    train_chars()
    test_acc = (model.evaluate(np.asarray(valX), np.asarray(valY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1

# We have a model that is good at chars and nums. Final testing, just for completion sake. If these don't do well, we won't rerun at this point.

val_num = validate_nums()
val_chars = validate_chars()

valNX = val_num[0]
valNY = val_num[1]

valCX = val_chars[0]
valCY = val_chars[1]


final_acc_char = (model.evaluate(np.asarray(valCX), np.asarray(valCY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1
final_acc_nums = (model.evaluate(np.asarray(valNX), np.asarray(valNY)))[1] # returns loss and accuracy, worried about accuracy, which is element 1

print("\nChar accuracy: " + str(final_acc_char))
print("Num accuracy: " + str(final_acc_nums))

model.save("final")
