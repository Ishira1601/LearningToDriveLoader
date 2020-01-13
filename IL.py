import numpy as np
import csv
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from tensorflow.python.client import device_lib
K.tensorflow_backend._get_available_gpus()
import matplotlib.pyplot as plt

def load_data(image_path, sensor_path):
    X = []          # Images go here
    for i in range(0, 300):
        # Load image and parse class name
        img = plt.imread(image_path+"/%d.jpg" % i)
        X.append(img)
    #X = np.array(X)

    y = []
    with open(sensor_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            y1= []
            for i in range(0, 5):
                y1.append(float(row[i]))
            y.append(y1)
    y = remove_extra(y, 300)
    #y = np.array(y)
    return X, y

def remove_extra(data, n):
    n_remove = len(data)-n
    for i in range(n_remove):
        remove = np.random.randint(0, len(data))
        data.pop(remove)

    return data

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

X, y = load_data("data", "16-01-09.csv")

#X_normalized= (X-np.min(X))/np.max(X)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(720, 1280, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=1))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=1))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=1))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=1))
model.add(AveragePooling2D(pool_size=(5, 5)))
model.add(Dense(1000, activation= 'sigmoid'))
model.add(Dense(5, activation= 'tanh'))

model.summary()

model.compile(loss=euclidean_distance_loss, optimizer=keras.optimizers.Adam(lr=0.0002), metrics=['accuracy'])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model.fit(X_train, y_train, epochs=640, batch_size=64, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])