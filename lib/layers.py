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

def combine_rows(modulated_rows):
    N = modulated_rows[0].get_shape().as_list()[1]
    output = tf.concat([tf.expand_dims(modulated_rows[i][:,i,:],1) for i in range(N)],axis=1)
    output_re = tf.math.real(output)
    output_im = tf.math.imag(output)
    outputs = tf.concat([tf.expand_dims(output_re,-1),tf.expand_dims(output_im,-1)],axis=-1)
    return outputs

