from tensorflow import keras
from imgaug import augmenters as iaa

import numpy as np
from scipy import misc
import nibabel as nib
from PIL import Image

import os

import motion

def get_mid_slice(vol_data,n,patch=False,coord=None):
    _,_,z = vol_data.shape

    sl_data = vol_data[:,:,int(z/2)]
    if(patch):
        sl_data = sl_data[coord[0]:coord[0]+n,coord[1]:coord[1]+n]
    else:
        sl_data = np.array(Image.fromarray(sl_data).resize((n,n)))
    sl_data = sl_data - sl_data.mean()
    sl_data = sl_data/np.max(sl_data)
    return sl_data

def batch_imgs(dir_name,image_names,n,corruption,input_domain,output_domain,patch):
    inputs = []
    outputs = []
    
    for img in image_names:
        vol_data = np.load(os.path.join(dir_name,img))['vol_data']
        coord = np.random.randint(vol_data.shape[0]-n,size=2)
        sl_data  = get_mid_slice(vol_data,n,patch,coord)
       
        if(corruption=='CLEAN'):
            num_pix = 0
            angle = 0
        elif(corruption=='TRANS'):
            num_pix = np.random.randint(0,10)
            angle = 0
        elif(corruption=='ALL'):
            num_pix = np.random.randint(0,10)
            angle = np.random.randint(0,45)
        elif(corruption=='SEQUENTIAL'):
            frame_jump = np.random.randint(0,30)
        else:
            raise ValueError('Unrecognized motion corruption setting.')

        k_line = np.random.randint(0,32)

        if(corruption=='SEQUENTIAL'):
            if(sl_data.shape!=(n,n) or np.isnan(sl_data).any()):
                continue
            end = -(len('.npz')) #end index of volume number
            num_len = 4
            start = end-num_len #start index of volume number
            next_img = int(img[start:end])+frame_jump
            next_name = img[:start]+str(next_img).zfill(4)+'.npz'
            try:
                next_img_vol = np.load(os.path.join(dir_name,next_name))['vol_data']
                next_img_sl = get_mid_slice(next_img_vol,n,patch,coord)
                if(next_img_sl.shape!=(n,n) or np.isnan(next_img_sl).any()):
                    continue
            except:
                continue
            corrupted_img,corrupted_k = motion.add_next_frame(sl_data,next_img_sl,k_line,return_k=True)
        else:
            corrupted_img,corrupted_k = motion.add_rotation_and_translation(sl_data,angle,num_pix,k_line,return_k=True)

        corrupted_k = np.expand_dims(corrupted_k,-1)
        corrupted_k_re = np.real(corrupted_k)
        corrupted_k_im = np.imag(corrupted_k)
        corrupted_k = np.concatenate([corrupted_k_re,corrupted_k_im], axis=2)
        if(output_domain=='IMAGE'):
            outputs.append(np.expand_dims(sl_data,axis=-1))
        elif(output_domain=='FREQUENCY'):
            true_k = np.expand_dims(np.fft.fftshift(np.fft.fft2(sl_data)), axis=-1)
            true_k_re = np.real(true_k)
            true_k_im = np.imag(true_k)
            true_k = np.concatenate([true_k_re,true_k_im], axis=2)
            outputs.append(true_k)

        if(input_domain=='FREQUENCY'):
            inputs.append(corrupted_k)
        elif(input_domain=='IMAGE'):
            inputs.append(np.expand_dims(corrupted_img,axis=-1))
            
    inputs = np.stack(inputs)        
    outputs = np.stack(outputs)

    return(inputs,outputs)

class DataSequence(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, n, corruption, input_domain, output_domain,patch):
        self.dir_name = data_path
        if(corruption=='SEQUENTIAL'):
            self.img_names = []
            for s in sorted (os.listdir(data_path)):
                for v in sorted(os.listdir(os.path.join(data_path,s))):
                    self.img_names.append(os.path.join(s,v))
        else:
            self.img_names = os.listdir(data_path)
        self.batch_x, self.batch_y = batch_imgs(self.dir_name,self.img_names,n,corruption,input_domain,output_domain,patch)
        self.batch_size = batch_size
        self.n = n

    def __len__(self):
        return int(np.ceil(len(self.batch_x)/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_y = self.batch_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
        batch_x = self.batch_x[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
        return batch_x, batch_y