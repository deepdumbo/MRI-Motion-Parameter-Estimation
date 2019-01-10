import tensorflow as tf
from tensorflow import keras

import data_generator

import numpy as np
import matplotlib.pyplot as plt

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data'

# Checkpointing
checkpoint_dir = os.path.join(dir_path,'../training/')
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = checkpoint_dir+checkpoint_name
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=5)

# Tensorboard
tb_dir = os.path.join(dir_path,'../tensorboard/')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
tb_callback = keras.callbacks.TensorBoard(
        log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)

# Set up model
n = 28 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(n,n,2)),
    keras.layers.Dense(2*(n**2), activation=tf.nn.tanh),
    keras.layers.Dense(n**2, activation=tf.nn.tanh),
    keras.layers.Dense(n**2),
    keras.layers.Reshape((n,n,1)),
    keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
    keras.layers.Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
    keras.layers.Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last', padding='same')
    ])

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00002,rho=0.9),
        loss='mean_squared_error',
        metrics=[keras.metrics.mae])

# Load data
data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data'
generator = data_generator.DataSequence(data_path, 100, n)

# Train model
model.fit_generator(generator, epochs=200, callbacks=[cp_callback,tb_callback])
