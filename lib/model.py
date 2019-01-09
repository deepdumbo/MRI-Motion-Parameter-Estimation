import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Import data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

# Generate k-space network input
def get_fft(img_array):
    img_fft = np.fft.fft2(img_array)
    img_fft_re = np.real(img_fft)
    img_fft_im = np.imag(img_fft)
    return np.stack([img_fft_re,img_fft_im],axis=-1)

train_k = get_fft(train_images)
test_k = get_fft(test_images)

# Format raw images for comparison
train_images = np.expand_dims(train_images,-1)
test_images = np.expand_dims(test_images,-1)

# Checkpointing
checkpoint_dir = os.path.join(dir_path,'../training/')
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = checkpoint_dir+checkpoint_name
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=1)

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

model.fit(train_k, train_images, epochs=5, batch_size=100, callbacks=[cp_callback,tb_callback])

test_loss, test_acc =  model.evaluate(test_k,test_images)
print('Test accuracy: ', test_acc)
