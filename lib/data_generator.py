from tensorflow import keras

import numpy as np
from PIL import Image

import os

def list_imgs(data_path):
    imgs = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            imgs.append(os.path.join(path, name))
    return imgs

def get_img(img_path,n):
    img = Image.open(img_path).convert('YCbCr')
    img_array = np.expand_dims(np.array(img)[:,:,0],-1)

    x = int((img_array.shape[0]-n)/2.0)
    y = int((img_array.shape[1]-n)/2.0)

    img_array = img_array[x:x+n,y:y+n]
    return img_array

def batch_imgs(batch_paths,n):
    for img_path in batch_paths:
        img_array = get_img(img_path,n)
        if(np.min(img_array.shape[0:2])>=n):
            yield img_array

def get_fft(img_array):
    img_fft = np.fft.fft2(img_array)
    img_fft_re = np.real(img_fft)
    img_fft_im = np.imag(img_fft)
    return np.concatenate([img_fft_re,img_fft_im], axis=3)

class DataSequence(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, n):
        self.img_paths = list_imgs(data_path)
        self.batch_size = batch_size
        self.n = n

    def __len__(self):
        return int(np.ceil(len(self.img_paths)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_paths = self.img_paths[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_y = np.stack(list(batch_imgs(batch_paths,self.n)), axis=0)
        batch_x = get_fft(batch_y)

        return batch_x, batch_y
