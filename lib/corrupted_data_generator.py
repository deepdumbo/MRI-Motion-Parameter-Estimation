from tensorflow import keras
from imgaug import augmenters as iaa

import numpy as np
from scipy import misc
from PIL import Image

import os

import motion

def batch_corrupted_imgs(dir_name,image_names,n):
    batch_imgs = []
    batch_ks = []
    i=0
    for img in image_names:
        vol_data = np.load(os.path.join(dir_name,img))['vol_data']
        _,_,z = vol_data.shape

        img_data = vol_data[:,:,int(z/2)]
        img_data = np.array(Image.fromarray(img_data).resize((n,n)))
        img_data = img_data-img_data.mean()
        img_data = img_data/np.max(img_data)

        num_pix = np.random.randint(0,10)
        k_line = np.random.randint(0,32)

        _,corrupted_img,corrupted_k = motion.add_horiz_translation(img_data,num_pix,k_line,return_k=True)

        corrupted_k = np.expand_dims(corrupted_k,-1)
        corrupted_k_re = np.real(corrupted_k)
        corrupted_k_im = np.imag(corrupted_k)
        corrupted_k = np.concatenate([corrupted_k_re,corrupted_k_im], axis=2)
        batch_imgs.append(np.expand_dims(img_data,axis=-1))
        batch_ks.append(corrupted_k)
        
        i+=1
    
    batch_imgs = np.stack(batch_imgs)
    batch_ks = np.stack(batch_ks)
    return(batch_imgs,batch_ks)

class DataSequence(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, n):
        self.dir_name = data_path
        self.img_names = os.listdir(data_path)
        self.batch_y, self.batch_x = batch_corrupted_imgs(self.dir_name,self.img_names,n)
        self.batch_size = batch_size
        self.n = n

    def __len__(self):
        return int(np.ceil(len(self.img_names)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_y = self.batch_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
        batch_x = self.batch_x[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
        return batch_x, batch_y
