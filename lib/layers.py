import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

def fouriertranslate(input_layers):
    fft_input = input_layers[0]
    shift = input_layers[1]
    N = fft_input.get_shape().as_list()[0]
    complex_input = tf.dtypes.cast(fft_input[:,:,0],'complex64')+1j*tf.dtypes.cast(fft_input[:,:,1],'complex64')

    col = np.expand_dims(np.arange(N),0)
    cols = tf.convert_to_tensor(np.fft.fftshift(np.repeat(col,N,axis=0)),dtype='float32')
    cols_exponent = 2*math.pi*1j*tf.dtypes.cast(tf.math.scalar_mul(tf.math.divide(shift[0],N),cols),'complex64')
    cols_exp = tf.math.exp(cols_exponent)

    row = np.expand_dims(np.arange(N),-1)
    rows = tf.convert_to_tensor(np.fft.fftshift(np.repeat(row,N,axis=1)),dtype='float32')
    rows_exponent = 2*math.pi*1j*tf.dtypes.cast(tf.math.scalar_mul(tf.math.divide(shift[1],N),rows),'complex64')
    rows_exp = tf.math.exp(rows_exponent)

    combined_exp = tf.math.multiply(rows_exp,cols_exp)
    output = tf.math.multiply(tf.dtypes.cast(complex_input,'complex64'),combined_exp)
    return output

def batch_fouriertranslate(input_layers):
    fft_input = input_layers[0]
    shift = tf.dtypes.cast(input_layers[1],'float32')
    return tf.map_fn(fouriertranslate,[fft_input,shift],dtype='complex64')

# DEPRECATED; keeping for reference
def combine_trans_rows(modulated_rows):
    N = modulated_rows[0].get_shape().as_list()[1]
    output = tf.concat([tf.expand_dims(modulated_rows[i][:,i,:],1) for i in range(N)],axis=1)
    output_re = tf.math.real(output)
    output_im = tf.math.imag(output)
    outputs = tf.concat([tf.expand_dims(output_re,-1),tf.expand_dims(output_im,-1)],axis=-1)
    return outputs

def fftshift2(tensor):
    # Assumes input tensor of size: (height,width)
    shape = tensor.get_shape().as_list()
    N = shape[0]
    half = int(N/2)
    x_shifted_tensor = tf.concat([tensor[half:,:],tensor[:half,:]],axis=0)
    shifted_tensor = tf.concat([x_shifted_tensor[:,half:],x_shifted_tensor[:,:half]],axis=1)
    return shifted_tensor

def fftshift(tensor,axis):
    # Assumes input tensor of size: (height,width)
    shape = tensor.get_shape().as_list()
    N = shape[0]
    half = int(N/2)
    if(axis==0):
        shifted_tensor = tf.concat([tensor[half:,:],tensor[:half,:]],axis=0)
    else:
        shifted_tensor = tf.concat([tensor[:,half:],tensor[:,:half]],axis=1)
    return shifted_tensor


def get_rowmasks(inputs,n):
    i = inputs[0]
    batch_size = inputs[1]

    mask = tf.zeros((n,n))
    mask = tf.expand_dims(tf.concat(axis=0, values=[mask[:i], tf.expand_dims(tf.ones(n),0), mask[i+1:]]),0)
    return tf.dtypes.cast(tf.tile(mask,[batch_size,1,1]),'complex64')


def fourierrotate(input_layers):
    fft_input = input_layers[0]
    theta = input_layers[1][2]
    
    N = fft_input.get_shape().as_list()[0]

    x_shift = tf.math.tan(theta/2)
    y_shift = -tf.math.sin(theta)
    img_tensor = tf.signal.ifft2d(fftshift2(fft_input)) # fine here

    row = np.expand_dims(np.arange(N),-1)-32
    rows = tf.convert_to_tensor(np.repeat(row,N,axis=1),dtype='float32')

    col = np.expand_dims(np.arange(N),0)-32
    cols = tf.convert_to_tensor(np.repeat(col,N,axis=0),dtype='float32') # fine here

    prod_x = rows*cols*x_shift
    prod_exponent_x = -2*math.pi*1j*tf.dtypes.cast(prod_x,'complex64')/N
    prod_exp_x = tf.math.exp(prod_exponent_x) 

    img_xfft = fftshift(tf.signal.fft(img_tensor),axis=1)
    x_img_shift = tf.signal.ifft(fftshift(img_xfft*prod_exp_x,axis=1))
    
    prod_y = rows*cols*y_shift
    prod_exponent_y = -2*math.pi*1j*tf.dtypes.cast(prod_y,'complex64')/N
    prod_exp_y = tf.math.exp(prod_exponent_y)

    x_img_shift_swapaxes = tf.transpose(x_img_shift,[1,0])
    img_xyfft = fftshift(tf.signal.fft(x_img_shift_swapaxes),axis=1)
    xy_img_shift_swapaxes = tf.signal.ifft(fftshift(img_xyfft*prod_exp_y,axis=1))
    xy_img_shift = tf.transpose(xy_img_shift_swapaxes,[1,0])

    img_xyxfft = fftshift(tf.signal.fft(xy_img_shift),axis=1)
    rot_img = tf.signal.ifft(fftshift(img_xyxfft*prod_exp_x,axis=1))

    rot_fft = fftshift2(tf.signal.fft2d(rot_img))
    return rot_fft

def batch_fourierrotate(input_layers):
    fft_input = input_layers[0]
    theta = tf.dtypes.cast(input_layers[1],'float32')
    return tf.map_fn(fourierrotate,[fft_input,theta],dtype='complex64')

def combine_rot_rows(rotation_data):
    rotated_rows = rotation_data[0]
    rotation_masks = rotation_data[1]
    
    fft_sum = tf.zeros_like(rotated_rows[0])
    mask_sum = tf.zeros_like(rotation_masks[0])
    for i in range(len(rotated_rows)):
        fft_sum += rotated_rows[i]*rotation_masks[i]
        mask_sum += rotation_masks[i]
    output = fft_sum/mask_sum
    output_re = tf.math.real(output)
    output_im = tf.math.imag(output)
    outputs = tf.concat([tf.expand_dims(output_re,-1),tf.expand_dims(output_im,-1)],axis=-1)
    output = tf.where(tf.math.is_nan(outputs), tf.zeros_like(outputs), outputs)
    return output
