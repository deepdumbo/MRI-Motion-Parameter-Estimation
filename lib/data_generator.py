from tensorflow import keras
from imgaug import augmenters as iaa

import numpy as np
from scipy import misc
from PIL import Image

import os

def list_imgs(data_path):
    imgs = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            imgs.append(os.path.join(path, name))
    return imgs

def batch_imgs(batch_paths,n):
    for img_path in batch_paths:
        img = Image.open(img_path).convert('L')
        img_array = np.expand_dims(np.array(img),-1)
        yield img_array

def augment(batch_imgs):
    affine = iaa.Affine(rotate=[0,90,180,270])
    aug_imgs = affine.augment_images(batch_imgs)
    return aug_imgs

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
        aug_batch_y = augment(batch_y)
        aug_batch_x = get_fft(aug_batch_y)

        return aug_batch_x, aug_batch_y
