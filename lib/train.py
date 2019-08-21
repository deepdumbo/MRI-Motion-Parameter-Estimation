import tensorflow as tf
from tensorflow import keras
import models
import imagenet_data_generator
import mri_data_generator

import numpy as np
import matplotlib.pyplot as plt

import argparse
import configparser
import os
from shutil import copyfile
import datetime
import math

parser = argparse.ArgumentParser(description='Train a model to reconstruct images from k-space data.')
parser.add_argument('config',help ='Path to .ini config file.')

args = parser.parse_args()
config_path = args.config

config = configparser.ConfigParser()
config.read(args.config)

n = config.getint('DATA','n')
dataset = config.get('DATA','dataset')
corruption = config.get('DATA','corruption').upper()
if(config.has_option('DATA','corruption_extent')):
    corruption_extent = config.get('DATA', 'corruption_extent').upper()
else:
    corruption_extent = 'ONE'
if(config.has_option('DATA','patch')):
    patch = config.getboolean('DATA','patch')

architecture = config.get('MODEL','architecture')
if(config.has_option('MODEL','input_domain')):
    input_domain = config.get('MODEL','input_domain')
else:
    input_domain = 'FREQUENCY'

if(config.has_option('MODEL','output_domain')):
    output_domain = config.get('MODEL','output_domain')
    if(dataset=='IMAGENET' and output_domain=='FREQUENCY'):
        raise ValueError('Invalid dataset and output domain combination.')
else:
    output_domain = 'IMAGE'

if(architecture=='UNET' and config.has_option('MODEL','nonlinearity')):
    nonlinearity = config.get('MODEL','nonlinearity')
else:
    nonlinearity = 'relu'

pretrain = config.getboolean('TRAINING','pretrain')
num_epochs = config.getint('TRAINING','num_epochs')

if(pretrain):
    pretrain_string = 'True'
else:
    pretrain_string = 'False'

job_name = dataset+'-'+corruption+'-'+corruption_extent+'-'+'PATCH'+str(patch)+'-'+architecture+'-'+nonlinearity+'-'+input_domain+'_INDOMAIN-'+output_domain+'_OUTDOMAIN-'+pretrain_string+'-'+str(num_epochs)+'epoch-'+str(n)

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

bold_dir = '/data/vision/polina/scratch/nmsingh/bold-data/'
bold_dir_train = bold_dir+'train'
bold_dir_test = bold_dir+'test'

# Checkpointing
checkpoint_dir = os.path.join(dir_path,'../training/',job_name)
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = os.path.join(checkpoint_dir,checkpoint_name)
if os.path.exists(checkpoint_dir):
    raise ValueError('Job name has already been used.')
else:
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=5, save_weights_only=True)

# Tensorboard
tb_dir = os.path.join(checkpoint_dir,'tensorboard/')
if os.path.exists(tb_dir):
    raise ValueError('Tensorboard logs have already been created under this name.')
else:
    os.makedirs(tb_dir)
tb_callback = keras.callbacks.TensorBoard(
        log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)

if(architecture=='SLIM'):
    model = models.get_slim_model(n)
elif(architecture=='STANDARD'):
    model = models.get_full_model(n)
elif(architecture=='CONV'):
    if(input_domain =='IMAGE'):
        model = models.get_conv_model(n, (n,n,1))
    elif(input_domain =='FREQUENCY'):
        model = models.get_conv_model(n, (n,n,2))
elif(architecture=='UNET'):
    if(input_domain == 'IMAGE'):
        model = models.get_Unet(n, nonlinearity, (n,n,1))
    elif(input_domain =='FREQUENCY'):
        model = models.get_Unet(n, nonlinearity, (n,n,2))
elif(architecture=='PARAMETERIZED'):
    model = models.get_parameterized_model(n)
elif(architecture=='PARAMETERIZED_THETA'):
    model = models.get_theta_model(n)
else:
    raise ValueError('Unrecognized architecture.')

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00002,rho=0.9),
        loss='mean_squared_error',
        metrics=[keras.metrics.mae])

# Copy config and log model summary
copyfile(args.config,os.path.join(checkpoint_dir,job_name+'_config.ini'))
with open(os.path.join(checkpoint_dir,'summary.txt'),'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x+'\n'))

# Pretrain, if specified,
if(pretrain):
    print('Loading pretrained weights')
    model.load_weights('../training/automap64/cp-0200.ckpt')

# Load data
if(dataset=='IMAGENET'):
    train_generator = imagenet_data_generator.DataSequence(imagenet_dir_train, 100, n)
    test_generator = imagenet_data_generator.DataSequence(imagenet_dir_test, 100, n)
elif(dataset=='BRAIN'):
    train_generator = mri_data_generator.DataSequence(adni_dir_train, 100, n, dataset, corruption, corruption_extent, input_domain, output_domain)
    test_generator = mri_data_generator.DataSequence(adni_dir_test, 100, n, dataset, corruption, corruption_extent, input_domain, output_domain)
elif(dataset=='BOLD'):
    train_generator = mri_data_generator.DataSequence(bold_dir_train, 100, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch)
    test_generator = mri_data_generator.DataSequence(bold_dir_test, 100, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch)
else:
    raise ValueError('Unrecognized dataset.')

# Train model
model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=100, validation_data=test_generator, callbacks=[cp_callback,tb_callback])
