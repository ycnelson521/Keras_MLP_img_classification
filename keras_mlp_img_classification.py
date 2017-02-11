import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.utils.visualize_util import model_to_dot
from keras.datasets import mnist

# Number of epoch
NP_EPOCH = 20 
NP_BATCH_SIZE = 64 

IMG_HEIGHT = 480
IMG_WIDTH = 640
IMG_CHANNEL = 3

ACT_METHOD = "sigmoid"
#ACT_METHOD = "relu"
OUTPUT_ACT_METHOD = "softmax"
LOSS = "categorical_crossentropy"

LOAD_WEIGHT = False #True 
SAVE_WEIGHT = False #True

# Load data from directory
def load_image_assign_label(path_in):
    total_file_cnt =0
    for root, dirs, files in os.walk(path_in, topdown=False):
        for name in dirs:
            tmp_dir_path = os.path.join(root, name)
            print("Scanning %s" %tmp_dir_path)

            dir_file_cnt = 0
            for filename in glob.glob(os.path.join(tmp_dir_path, "*.jpg")):
                dir_file_cnt += 1

            print("Directory image file count = %d" % dir_file_cnt)

            total_file_cnt += dir_file_cnt
            print("Total image file count = %d" % total_file_cnt)

        label = np.zeros((total_file_cnt), np.uint8)
        data = np.zeros((total_file_cnt, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), np.uint8)

        i=0
        label_idx = 0

        for name in dirs:
            tmp_dir_path = os.path.join(root, name)
            print("Reading %s" % tmp_dir_path)

            for filename in glob.glob(os.path.join(tmp_dir_path, "*.jpg")):
                #print("Reading %s" % filename)
                data[i] = cv2.imread(filename)
                label[i] = label_idx
                i = i+1

            label_idx += 1

    data = data.astype('float32')
    data = data / 255
    return data, label, total_file_cnt, label_idx


# Load all images from one diretory
def load_data(path):
    file_cnt = 0
    for filename in glob.glob(os.path.join(path, "*.jpg")):
        file_cnt += 1

    for filename in glob.glob(os.path.join(path, "*.jpg")):
        print("Reading %s" % filename)
        data = cv2.imread(filename)
        i = i+1

    data = data.reshape(file_cnt, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL)
    data = data.astype('float32')
    data /= 255
    return data

#Load from a label text file
def load_label(label_filename):
    if not os.path.exists(label_filename):
        print("error loading label file: %s" % label_filename)
    data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Load data from each label-directory
path_train = "train//"
path_test = "test//"
print("== Reading training data and labels ==") 
(X_train, Y_train, total_file_cnt_train, label_cnt) = load_image_assign_label(path_train)
print("== Reading test data and labels ==")
(X_test, Y_test, total_file_cnt_test, label_cnt_test) = load_image_assign_label(path_test)
print(X_train.shape)
print(X_test.shape)
print(type(X_train))

print(Y_train.shape)
print(Y_test.shape)
print(type(Y_train))

# Draw samples of input images
#for i in range(80, 140, 1):
#    cv2.imshow("X_train[%d]" % i , X_train[i])
#    cv2.waitKey(250)
#    cv2.destroyAllWindows()

# Reshape training and testing data into 2-D array
data_img_size = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL
X_train = X_train.reshape(total_file_cnt_train, data_img_size)
X_test = X_test.reshape(total_file_cnt_test, data_img_size)
print(X_train.shape)
print(X_test.shape)

Y_train = np_utils.to_categorical(Y_train, label_cnt)
Y_test = np_utils.to_categorical(Y_test, label_cnt)

print(Y_train[0])
print(Y_test[0])

# build simple 2-layer mlp model
print("Activation method = %s" % ACT_METHOD)
print("Output activation method = %s" % OUTPUT_ACT_METHOD)
print("Loss = %s" % LOSS)

model = Sequential()

model.add(Dense(input_dim=data_img_size, output_dim=200, activation=ACT_METHOD))

model.add(Dense(output_dim=128, activation=ACT_METHOD))

model.add(Dense(output_dim=128, activation=ACT_METHOD))

model.add(Dense(output_dim=128, activation=ACT_METHOD))

model.add(Dense(output_dim=label_cnt, activation=OUTPUT_ACT_METHOD))

if LOAD_WEIGHT:
    print("loading weights")
    model.load_weights("model_weights.h5", by_name=True)

model.compile(loss = LOSS,
#model.compile(loss = 'mse',
              optimizer = SGD(lr=0.1),
              metrics = ['accuracy'])

def train_and_show_result(model):
    t_start = time.time()
    training_history = model.fit(X_train, Y_train,
                                 batch_size = NP_BATCH_SIZE,
                                 nb_epoch=NP_EPOCH,
                                 verbose=2)

    score = model.evaluate(X_test, Y_test)
    t_end = time.time()
    print("\n--------------------")
    print("Total Testing Loss: {} ".format(score[0]))
    print("Testing Accuracy: {} ".format(score[1]))
    print("Time= %f second\n" % (t_end - t_start))
    return training_history

def plot_training_history(training_history):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_xlabel('Epoch')
    axarr[0].set_ylabel('Training Accuracy')
    axarr[0].plot(list(range(1, NP_EPOCH+1)), training_history.history['acc'])
    axarr[1].set_ylabel('Training Error')
    axarr[1].plot(list(range(1, NP_EPOCH+1)), training_history.history['loss'])
    plt.show()

history = train_and_show_result(model)

if SAVE_WEIGHT:
    print("Saving weights")
    model.save_weights("model_weights.h5")

plot_training_history(history)

print("Activation method = %s" % ACT_METHOD)
print("Output activation method = %s" % OUTPUT_ACT_METHOD)
print("Loss = %s" % LOSS)
print("= test end =")

