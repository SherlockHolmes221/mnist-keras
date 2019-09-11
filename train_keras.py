import keras
import os, argparse

from keras.optimizers import SGD
from keras.utils import np_utils
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten, Conv2D
import numpy as np
import gzip

# Params for MNIST
from tensorflow.python.ops.gen_nn_ops import MaxPool

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

# Extract the images
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

def get_mnist_data():
    train_data = extract_data(os.path.join(os.path.dirname(__file__),'mnist/train-images-idx3-ubyte.gz'), 60000)
    train_labels = extract_labels(os.path.join(os.path.dirname(__file__),'mnist/train-labels-idx1-ubyte.gz'), 60000)
    test_data = extract_data(os.path.join(os.path.dirname(__file__),'mnist/t10k-images-idx3-ubyte.gz'), 10000)
    test_labels = extract_labels(os.path.join(os.path.dirname(__file__),'mnist/t10k-labels-idx1-ubyte.gz'), 10000)
    return train_data,train_labels,test_data,test_labels

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, y_train, x_test, y_test

def train_dnn_with_dropout(x_train, y_train, x_test, y_test):
    #training上的performance会变差
    print("train_dnn_with_dropout")
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=10,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, nb_epoch=200)
    score = model.evaluate(x_test, y_test)
    print("Total loss on Testing data: ", score[0])
    print("Accuracy of Testing data: ", score[1])

def train_dnn(x_train, y_train, x_test, y_test):
    print('train_dnn begin')
    model = Sequential()
    model.add(Dense(input_dim=28*28, output_dim=500))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=500))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    #mse的效果可能不太好 categorical_crossentropy的会比较好
    #batch size太大的效果不好 小的话gpu的耗时很大
    #model.compile(loss='mse', optimizer=SGD(lr=0.15), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, nb_epoch=30)
    score = model.evaluate(x_test, y_test)
    print("Total loss on Testing data: ", score[0])
    print("Accuracy of Testing data: ", score[1])
    #result = model.predict(x_data)

def train_cnn(x_train, y_train, x_test, y_test):
    print('train_cnn_begin')
    model = Sequential()

    #parm num: 3*3*25
    model.add(Conv2D(25, (3, 3),  input_shape=[28, 28, 1]))#25*26*26
    model.add(MaxPooling2D(2, 2))#25*13*13

    #parm num: 3*3*25*50
    model.add(Conv2D(50, (3, 3)))#50*11*11
    model.add(MaxPooling2D(2, 2))#50*5*5

    #flatten
    model.add(Flatten()) #1250

    #fc1
    model.add(Dense(output_dim=100))
    model.add(Activation('relu'))
    #fc2
    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, nb_epoch=40)
    score = model.evaluate(x_test, y_test)
    print("Total loss on Testing data: ", score[0])
    print("Accuracy of Testing data: ", score[1])

if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_data()
    #or x_train, y_train, x_test, y_test = get_mnist_data()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # print(x_train.shape) #(60000, 784)
    # print(y_train.shape) #(60000, 10)
    # print(x_test.shape)  #(10000, 784)
    # print(y_test.shape)#(10000, 10)
    # print(y_test)
    # print(x_train[0])
    # print(y_train[0])

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--train_type',type=str,default='train_cnn',
        help='input train_dnn or train_cnn or train_dnn_with_dropout'
    )
    info = parser.parse_args()
    type = info.train_type

    if(type == 'train_dnn'):
        #method——全连接神经网络
        train_dnn(x_train, y_train, x_test, y_test)
    elif(type == 'train_dnn_with_dropout'):
        #method——x_test增加了噪声，网络加了dropout
        x_test = np.random.normal(x_test)
        train_dnn_with_dropout(x_train, y_train, x_test, y_test)
    else:
        #卷积神经网络
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        train_cnn(x_train, y_train, x_test, y_test)

