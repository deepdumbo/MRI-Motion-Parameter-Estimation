import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
import layers

def get_full_model(n):
    return(keras.Sequential([
        Flatten(input_shape=(n,n,2)),
        Dense(2*(n**2), activation=tf.nn.tanh),
        Dense(n**2, activation=tf.nn.tanh),
        Dense(n**2),
        Reshape((n,n,1)),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last', padding='same')
        ]))

def get_slim_model(n):
    return(keras.Sequential([
        Flatten(input_shape=(n,n,2)),
        Dense(int(4*(math.log(n,2))), activation=tf.nn.tanh),
        Dense(int(2*math.log(n,2)), activation = tf.nn.tanh),
        Dense(n**2),
        Reshape((n,n,1)),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last',padding='same')
        ]))

def get_conv_model(n, input_size):
    return(keras.Sequential([
        Conv2DTranspose(1,(5,5), input_shape=input_size, strides=(1,1), data_format='channels_last',padding='same'),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2D(64, (5,5), strides=(1,1), activation=tf.nn.relu, padding='same'),
        Conv2DTranspose(1, (7,7), strides=(1,1), data_format='channels_last', padding='same')
        ]))

def get_Unet(n, nonlinearity, input_size):

    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = nonlinearity, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(input_size[2], 1, activation = 'linear')(conv9)
    model = keras.models.Model(inputs = inputs, outputs = conv10)
    return model
    
def get_parameterized_model(n):
    inputs = Input((n,n,2))
    conv1 = Conv2D(64, 2, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal', name = 'FirstConv')(inputs)
    conv2 = Conv2D(64, 2, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal', name = 'SecondConv')(conv1)
    conv3 = Conv2D(64, 2, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal', name = 'ThirdConv')(conv2)

    conv_flat = Flatten()(conv3)
    theta_flat = Dense(3*n, activation = tf.nn.relu)(conv_flat)
    theta = Reshape((n,3))(theta_flat)
    
    theta_modules = []
    trans_layers = []
    rot_layers = []
    mask_rot_layers = []
    
    for i in range(n):
        theta_modules.append(theta[:,i,:])
        
        trans_layer = keras.layers.Lambda(layers.batch_fouriertranslate, name = 'Trans'+str(i))
        trans_layers.append(trans_layer((inputs,theta_modules[i])))
        
        rot_layer = keras.layers.Lambda(layers.batch_fourierrotate, name = 'Rot'+str(i))
        rot_layers.append(rot_layer((trans_layers[i],theta_modules[i])))
       
        row_masks = keras.layers.Lambda(layers.get_rowmasks, name = 'RowMasks'+str(i),arguments={'n':n})((i,tf.shape(trans_layers[i])[0]))
        mask_rot_layer = keras.layers.Lambda(layers.batch_fourierrotate, name = 'MaskRot'+str(i))
        mask_rot_layers.append(mask_rot_layer((row_masks,theta_modules[i])))
        
    output = keras.layers.Lambda(layers.combine_rot_rows)((rot_layers,mask_rot_layers))
    model = keras.models.Model(inputs = inputs, outputs = output)
    return model

