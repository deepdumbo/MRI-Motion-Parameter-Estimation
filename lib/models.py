import math
import tensorflow as tf
from tensorflow import keras

def get_full_model(n):
    return(keras.Sequential([
        keras.layers.Flatten(input_shape=(n,n,2)),
        keras.layers.Dense(2*(n**2), activation=tf.nn.tanh),
        keras.layers.Dense(n**2, activation=tf.nn.tanh),
        keras.layers.Dense(n**2),
        keras.layers.Reshape((n,n,1)),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last', padding='same')
        ]))

def get_slim_model(n):
    return(keras.Sequential([
        keras.layers.Flatten(input_shape=(n,n,2)),
        keras.layers.Dense(int(4*(math.log(n,2))), activation=tf.nn.tanh),
        keras.layers.Dense(int(2*math.log(n,2)), activation = tf.nn.tanh),
        keras.layers.Dense(n**2),
        keras.layers.Reshape((n,n,1)),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last',padding='same')
        ]))

def get_conv_model(n):
    return(keras.Sequential([
        keras.layers.Conv2DTranspose(1,(5,5), input_shape=(n,n,2), strides=(1,1), data_format='channels_last',padding='same'),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last', padding='same')
        ]))

