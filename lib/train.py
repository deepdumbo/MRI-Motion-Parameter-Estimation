import tensorflow as tf
from tensorflow import keras
import models
import imagenet_data_generator
import mri_data_generator
import visualization

import random
import numpy as np
import matplotlib.pyplot as plt

import argparse
import configparser
import os
import atexit
from shutil import rmtree
from shutil import copyfile
import pickle
import datetime
import math

parser = argparse.ArgumentParser(description='Train a model to reconstruct images from k-space data.')
parser.add_argument('config',help ='Path to .ini config file.')
parser.add_argument('--debug',help='Boolean indicating whether to run small-scale training experiment.',action='store_true')
parser.add_argument('--training_dir',help='Name of folder in which to store training logs')
parser.add_argument('--suffix',help='Suffix appended to job name.')

args = parser.parse_args()
config_path = args.config
debug = args.debug
training_dir = args.training_dir
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
if(corruption_extent == 'CONTIGUOUS'):
    if(config.has_option('DATA','num_lines')):
        num_lines = config.getint('DATA','num_lines')
    else:
        num_lines = None
else:
    num_lines = None
if(corruption_extent == 'PARTIAL'):
    if(config.has_option('DATA','num_move_lines')):
        num_move_lines = config.getint('DATA','num_move_lines')
    else:
        num_move_lines = None
else:
    num_move_lines = None
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

if(config.has_option('MODEL','nonlinearity')):
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

adni_dir = '/data/vision/polina/scratch/nmsingh/ADNI-data/uncropped/adni-split-by-subj-slice128-axial/'
adni_dir_train = adni_dir+'train/origs'
adni_dir_val = adni_dir+'test/origs'

bold_dir = '/data/vision/polina/scratch/nmsingh/bold-data/'
bold_dir_train = bold_dir+'train'
bold_dir_test = bold_dir+'test'

# Checkpointing
if((training_dir is None) or debug):
    checkpoint_dir = os.path.join(dir_path,'../training/',job_name)
else:
    checkpoint_dir = os.path.join(dir_path,'../training/',training_dir,job_name)
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
    if(output_domain!='THETA' and output_domain!='THETA_K'):
        raise ValueError('Architecture '+architecture+' and output domain '+output_domain+' are incompatible.')
    if(output_domain=='THETA_K'):
        if(input_domain=='IMAGE'):
            model = models.get_theta_k_model((n,n,1), nonlinearity)
        elif(input_domain=='FREQUENCY'):
            model = models.get_theta_k_model((n,n,2), nonlinearity)
    elif(output_domain=='THETA'):
        if(input_domain=='IMAGE'):
            model = models.get_theta_model((n,n,1), nonlinearity)
        elif(input_domain=='FREQUENCY'):
            model = models.get_theta_model((n,n,2), nonlinearity)
    if(input_domain=='IMAGE'):
        model = models.get_theta_model((n,n,1), nonlinearity)
    elif(input_domain=='FREQUENCY'):
        model = models.get_theta_model((n,n,2), nonlinearity)
elif(architecture=='PARAMETERIZED_SINGLE_THETA'):
    if(output_domain!='SINGLE_THETA' and output_domain!='SINGLE_THETA_K'):
        raise ValueError('Architecture '+architecture+' and output domain '+output_domain+' are incompatible.')
    if(corruption_extent!='CONTIGUOUS'):
        raise ValueError('Single parameter architecture cannot be used for non-contiguous motion.')
    if(output_domain=='SINGLE_THETA_K'):
        if(input_domain=='IMAGE'):
            model = models.get_theta_k_model((n,n,1), nonlinearity, single=True)
        elif(input_domain=='FREQUENCY'):
            model = models.get_theta_k_model((n,n,2), nonlinearity, single=True)
    elif(output_domain=='SINGLE_THETA'):
        if(input_domain=='IMAGE'):
            model = models.get_theta_model((n,n,1), nonlinearity, single=True)
        elif(input_domain=='FREQUENCY'):
            model = models.get_theta_model((n,n,2), nonlinearity, single=True)
else:
    raise ValueError('Unrecognized architecture.')

model.compile(optimizer=keras.optimizers.RMSprop(lr=0.00002,rho=0.9),
        loss='mean_squared_error',
        metrics=[keras.metrics.mae,keras.metrics.mse])
print('Got model')

# Copy config and log model summary
copyfile(args.config,os.path.join(checkpoint_dir,job_name+'_config.ini'))
summary_file = os.path.join(checkpoint_dir,'summary.txt')
with open(summary_file,'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x+'\n'))

# Pretrain, if specified,
if(pretrain):
    print('Loading pretrained weights')
    model.load_weights('../training/automap64/cp-0200.ckpt')

print('Loading training data')
# Load train data
train_seed = np.random.randint(0,100)
np.random.seed(train_seed)
if(dataset=='IMAGENET'):
    train_generator = imagenet_data_generator.DataSequence(imagenet_dir_train, batch_size, n)
elif(dataset=='BRAIN'):
    train_generator = mri_data_generator.DataSequence(adni_dir_train, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, num_lines=num_lines, num_move_lines=num_move_lines, debug=debug)
elif(dataset=='BOLD'):
    train_generator = mri_data_generator.DataSequence(bold_dir_train, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch=patch, debug=debug)
else:
    raise ValueError('Unrecognized dataset.')

print('Loading test data')
# Load test data
test_seed = 0
np.random.seed(test_seed)
if(dataset=='IMAGENET'):
    test_generator = imagenet_data_generator.DataSequence(imagenet_dir_test, batch_size, n)
elif(dataset=='BRAIN'):
    test_generator = mri_data_generator.DataSequence(adni_dir_val, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, num_lines=num_lines, num_move_lines=num_move_lines, debug=debug)
elif(dataset=='BOLD'):
    test_generator = mri_data_generator.DataSequence(bold_dir_test, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, patch=patch, debug=debug)
else:
    raise ValueError('Unrecognized dataset.')

print('Saving data generators')
# Save data generators
train_outfile = os.path.join(checkpoint_dir,'train_generator.pkl')
train_file = open(train_outfile, 'wb') 
pickle.dump(train_generator, train_file)

test_outfile = os.path.join(checkpoint_dir,'test_generator.pkl')
test_file = open(test_outfile, 'wb') 
pickle.dump(test_generator, test_file)

# Write random seeds to summary
f=open(summary_file,'a+')
f.write('Train random seed: '+str(train_seed)+'\n')
f.write('Test random seed: '+str(test_seed))
f.close()

# For debug mode, print the number of parameters
if(debug):
    print('Number of parameters: '+ str(model.count_params()))

# Train model
model.fit_generator(train_generator, epochs=num_epochs, steps_per_epoch=100, validation_data=test_generator, callbacks=[cp_callback,tb_callback])

# Visualize outputs
if(not debug and (output_domain=='THETA' or output_domain=='SINGLE_THETA')):
    train_outs,val_outs = visualization.get_outputs(train_generator,test_generator)
    train_model_outs,val_model_outs = visualization.get_model_outputs(model,train_generator,test_generator)
    kde_filename = os.path.join(checkpoint_dir,'kde.png')
    visualization.generate_error_kde(train_outs,train_model_outs,val_outs,val_model_outs,kde_filename)
    scatter_filename = os.path.join(checkpoint_dir,'scatter.png')
    visualization.generate_error_scatter(train_outs,train_model_outs,val_outs,val_model_outs,scatter_filename)
