import tensorflow as tf
from tensorflow import keras
import data_generator
import corrupted_data_generator

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import datetime

parser = argparse.ArgumentParser(description='Train a model to reconstruct images from k-space data.')
parser.add_argument('n',type=int,help ='Dimension, in pixels, to which to crop images.')
parser.add_argument('--name',help='Name of directories containing checkpoints/tensorboard logs.')
args = parser.parse_args()
job_name = args.name
n = args.n

# Set up job name
if job_name is None:
    job_name = str(datetime.datetime.now())

# Get current and data directories
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data-preprocessed-'+str(n)+'/train'
adni_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/old/resize256-crop_0/train/vols'

# Checkpointing
checkpoint_dir = os.path.join(dir_path,'../training/',job_name)
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = os.path.join(checkpoint_dir,checkpoint_name)
if os.path.exists(checkpoint_dir):
    raise ValueError('Job name has already been used.')
else:
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=5)

# Tensorboard
tb_dir = os.path.join(checkpoint_dir,'tensorboard/',job_name)
if os.path.exists(tb_dir):
    raise ValueError('Tensorboard logs have already been created under this name.')
else:
    os.makedirs(tb_dir)
tb_callback = keras.callbacks.TensorBoard(
        log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)

# Set up model
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

# Write model summary to text file
with open(os.path.join(checkpoint_dir,'summary.txt'),'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x+'\n'))

# Load data
#generator = data_generator.DataSequence(data_path, 100, n)
motion_generator = corrupted_data_generator.DataSequence(adni_dir, 100, n, 3, 15)

# Train model
model.fit_generator(motion_generator, epochs=200, callbacks=[cp_callback,tb_callback])
