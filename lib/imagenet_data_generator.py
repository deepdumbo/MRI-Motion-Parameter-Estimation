from tensorflow import keras
from imgaug import augmenters as iaa

import numpy as np
from scipy import misc
from PIL import Image

import os

import motion

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
        img_array = img_array-img_array.mean()
        img_array = img_array/255.
        yield img_array

def augment(batch_imgs):
    affine = iaa.Affine(rotate=[0,90,180,270])
    aug_imgs = affine.augment_images(batch_imgs)
    return aug_imgs

def get_fft(img_array):
    num_pix = 0
    k_line = 0

    batch_ks = []
    batch_size = np.shape(img_array)[0]
    for i in range(batch_size):
        img = img_array[i,:,:,0]
        _,_,k = motion.add_horiz_translation(img,num_pix,k_line,return_k=True)
        k = np.expand_dims(k,-1)
        k_re = np.real(k)
        k_im = np.imag(k)
        k = np.concatenate([k_re,k_im], axis=2)
        batch_ks.append(k)
    
    batch_ks = np.stack(batch_ks)
    return batch_ks


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
