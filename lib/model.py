import tensorflow as tf
from tensorflow import keras
import imagenet_data_generator
import brain_data_generator

import numpy as np
import matplotlib.pyplot as plt

import argparse
import os
import datetime

parser = argparse.ArgumentParser(description='Train a model to reconstruct images from k-space data.')
parser.add_argument('n',type=int,help ='Dimension, in pixels, to which to crop images.')
parser.add_argument('--name',help='Name of directories containing checkpoints/tensorboard logs.')
parser.add_argument('--pretrain',action='store_true',help='Boolean indicating whether to pretrain the network with weights learned from images without motion corruption')
parser.add_argument('--dataset',default='BRAIN',help='Type of data to train on; must be IMAGENET or BRAIN')
parser.add_argument('--clean',action='store_true',help='Boolean indicating whether to train only onclean, non motion-corrupted input data')

args = parser.parse_args()
clean = args.clean
dataset = args.dataset.upper()
pretrain = args.pretrain
job_name = args.name
n = args.n

# Set up job name
if job_name is None:
    job_name = str(datetime.datetime.now())

# Get current and data directories
dir_path = os.path.dirname(os.path.realpath(__file__))

imagenet_dir = '/data/vision/polina/scratch/nmsingh/imagenet-data-preprocessed-'+str(n)+'/'
imagenet_dir_train = imagenet_dir+'train'
imagenet_dir_test = imagenet_dir+'test'

adni_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/old/resize256-crop_0/'
adni_dir_train = adni_dir+'train/vols'
adni_dir_test = adni_dir+'test/vols'

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

# Pretrain, if specified,
if(pretrain):
    print('Loading pretrained weights')
    model.load_weights('../training/automap64/cp-0200.ckpt')

# Load data
if(dataset=='IMAGENET'):
    print('Training on imagenet data')
    train_generator = imagenet_data_generator.DataSequence(imagenet_dir_train, 100, n)
    test_generator = imagenet_data_generator.DataSequence(imagenet_dir_test, 100, n)
elif(dataset=='BRAIN'):
    print('Training on brain data')
    train_generator = brain_data_generator.DataSequence(adni_dir_train, 100, n, clean)
    test_generator = brain_data_generator.DataSequence(adni_dir_test, 100, n, clean)
else:
    raise ValueError('Unrecognized dataset.')

# Train model
num_epochs = 500
model.fit_generator(train_generator, epochs=num_epochs, validation_data=test_generator, callbacks=[cp_callback,tb_callback])
