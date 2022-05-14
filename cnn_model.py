import keras
import tensorflow as tf
from collections import Counter
from keras.layers import Dense, Input
from keras.models import Model, Sequential
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, LSTM, Conv3D, MaxPool3D, Conv1D, MaxPool1D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from keras import initializers, optimizers, losses, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA, IncrementalPCA
from keras import backend as K
from tensorflow.keras import datasets, layers, models

def VGG19(x_train, y_train, batch_size, nEpochs, callback, METRICS, class_weights):
  INPUT_SHAPE = x_train.shape[1:]
  vgg_layers = tf.keras.applications.vgg19.VGG19(weights=None, 
                                                 include_top=False, 
                                                 input_shape=INPUT_SHAPE) 

  model = tf.keras.models.Sequential()

  model.add(vgg_layers)

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))

  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                loss='binary_crossentropy', 
                metrics=METRICS)

  #model.summary()

  model.fit(x_train, 
            y_train, 
            batch_size = batch_size, 
            epochs = nEpochs, 
            verbose = 0, 
            callbacks = callback, 
            class_weight = class_weights)
  
  return model

def VGG16(x_train, y_train, batch_size, nEpochs, callback, METRICS, class_weights):
  INPUT_SHAPE = x_train.shape[1:]

  vgg_layers = tf.keras.applications.vgg16.VGG16(weights=None, 
                                                 include_top=False, 
                                                 input_shape=INPUT_SHAPE) 

  model = tf.keras.models.Sequential()

  model.add(vgg_layers)

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))

  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                loss='binary_crossentropy', 
                metrics=METRICS)

  #model.summary()

  model.fit(x_train, 
            y_train, 
            batch_size = batch_size, 
            epochs = nEpochs, 
            verbose = 0, 
            callbacks = callback, 
            class_weight = class_weights)
  
  return model

def ResNet(x_train, y_train, batch_size, nEpochs, callback, METRICS, class_weights):
  INPUT_SHAPE = x_train.shape[1:]

  vgg_layers = tf.keras.applications.ResNet50V2(weights=None, 
                                                 include_top=False, 
                                                 input_shape=INPUT_SHAPE) 

  model = tf.keras.models.Sequential()

  model.add(vgg_layers)

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dropout(rate=0.3))

  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                loss='binary_crossentropy', 
                metrics=METRICS)

  #model.summary()

  model.fit(x_train, 
            y_train, 
            batch_size = batch_size, 
            epochs = nEpochs, 
            verbose = 0, 
            callbacks = callback, 
            class_weight = class_weights)
  
  return model


def s2D(x_train, y_train, batch_size, nEpochs, callback, METRICS, class_weights):
  InputShape = x_train.shape[1:]
  model = Sequential()

  model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=InputShape, activation='relu'))
  model.add(LeakyReLU())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(BatchNormalization())

  model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
  model.add(LeakyReLU())
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(BatchNormalization())

  model.add(Flatten())
  model.add(BatchNormalization())

  model.add(Dense(units=128, activation='relu'))
  model.add(LeakyReLU())
  model.add(BatchNormalization())
  model.add(Dropout(0.5))

  model.add(Dense(units=1, activation='sigmoid'))

  opt = tf.keras.optimizers.Adam(learning_rate=0.001)

  model.compile(loss = 'binary_crossentropy', 
                optimizer = opt, 
                metrics = METRICS           
                )

  #model.summary()

  model.fit(x_train, 
            y_train, 
            batch_size = batch_size, 
            epochs = nEpochs, 
            verbose = 0, 
            callbacks = callback, 
            class_weight = class_weights)

  return model