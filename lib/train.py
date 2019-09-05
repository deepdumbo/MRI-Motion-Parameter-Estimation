import tensorflow as tf
from tensorflow import keras
import models
import imagenet_data_generator
import mri_data_generator

import random
import numpy as np
import matplotlib.pyplot as plt

import argparse
import configparser
import os
import atexit
from shutil import rmtree
from shutil import copyfile
import datetime
import math

parser = argparse.ArgumentParser(description='Train a model to reconstruct images from k-space data.')
parser.add_argument('config',help ='Path to .ini config file.')
parser.add_argument('--debug',help='Boolean indicating whether to run small-scale training experiment.',action='store_true')
parser.add_argument('--suffix',help='Suffix appended to job name.')

args = parser.parse_args()
config_path = args.config
debug = args.debug
suffix = args.suffix

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
if(debug):
    num_epochs = 1
    batch_size = 1
else:
    num_epochs = config.getint('TRAINING','num_epochs')
    batch_size = 100

if(pretrain):
    pretrain_string = 'True'
else:
    pretrain_string = 'False'

dir_path = os.path.dirname(os.path.realpath(__file__))

if(debug):
    job_name = 'debug-job'
else:
    job_name = dataset+'-'+corruption+'-'+corruption_extent+'-'+'PATCH'+str(patch)+'-'+architecture+'-'+nonlinearity+'-'+input_domain+'_INDOMAIN-'+output_domain+'_OUTDOMAIN-'+pretrain_string+'-'+str(num_epochs)+'epoch-'+str(n)

if(suffix is not None):
    job_name += '-*'+suffix

# Set up job name
if job_name is None:
    job_name = str(datetime.datetime.now())

# Get current and data directories
dir_path = os.path.dirname(os.path.realpath(__file__))

imagenet_dir = '/data/vision/polina/scratch/nmsingh/imagenet-data-preprocessed-'+str(n)+'/'
imagenet_dir_train = imagenet_dir+'train'
imagenet_dir_test = imagenet_dir+'test'

adni_dir = '/data/ddmg/voxelmorph/data/t1_mix/proc/resize256-crop_x32-adni-split-by-subj-slice100/'
adni_dir_train = adni_dir+'train/vols'
adni_dir_val = adni_dir+'validate/vols'

bold_dir = '/data/vision/polina/scratch/nmsingh/bold-data/'
bold_dir_train = bold_dir+'train'
bold_dir_test = bold_dir+'test'

# Checkpointing
checkpoint_dir = os.path.join(dir_path,'../training/',job_name)
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = os.path.join(checkpoint_dir,checkpoint_name)
if os.path.exists(checkpoint_dir):
    raise ValueError('Job name has already been used: '+job_name)
else:
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=5, save_weights_only=True)

# Checkpoint deletion for debug mode
if(debug):
    def del_logs():
        rmtree(checkpoint_dir, ignore_errors=True)
        print('Deleted temp debug logs')
    atexit.register(del_logs)

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
    if(output_domain!='THETA'):
        raise ValueError('Architecture '+architecture+' and output domain '+output_domain+' are incompatible.')
    model = models.get_theta_model(n)
else:
    raise ValueError('Unrecognized architecture.')

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00002,rho=0.9),
        loss='mean_squared_error',
        metrics=[keras.metrics.mae])

# Copy config and log model summary
copyfile(args.config,os.path.join(checkpoint_dir,job_name+'_config.ini'))
summary_file = os.path.join(checkpoint_dir,'summary.txt')
with open(summary_file,'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x+'\n'))

# Pretrain, if specified,
if(pretrain):
    print('Loading pretrained weights')
    model.load_weights('../training/automap64/cp-0200.ckpt')

# Load train data
train_seed = np.random.randint(0,100)
np.random.seed(train_seed)
if(dataset=='IMAGENET'):
    train_generator = imagenet_data_generator.DataSequence(imagenet_dir_train, batch_size, n)
elif(dataset=='BRAIN'):
    train_generator = mri_data_generator.DataSequence(adni_dir_train, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, debug=debug)
elif(dataset=='BOLD'):
    train_generator = mri_data_generator.DataSequence(bold_dir_train, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch=patch, debug=debug)
else:
    raise ValueError('Unrecognized dataset.')

# Load test data
test_seed = 0
np.random.seed(test_seed)
if(dataset=='IMAGENET'):
    test_generator = imagenet_data_generator.DataSequence(imagenet_dir_test, batch_size, n)
elif(dataset=='BRAIN'):
    test_generator = mri_data_generator.DataSequence(adni_dir_val, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, debug=debug)
elif(dataset=='BOLD'):
    test_generator = mri_data_generator.DataSequence(bold_dir_test, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch=patch, debug=debug)
else:
    raise ValueError('Unrecognized dataset.')

# Write random seeds to summary
f=open(summary_file,'a+')
f.write('Train random seed: '+str(train_seed)+'\n')
f.write('Test random seed: '+str(test_seed))
f.close()

# Train model
model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=100, validation_data=test_generator, callbacks=[cp_callback,tb_callback])
