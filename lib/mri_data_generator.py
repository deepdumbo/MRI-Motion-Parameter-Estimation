from tensorflow import keras
from imgaug import augmenters as iaa

import numpy as np
from scipy import misc
import nibabel as nib
from PIL import Image

import os
import time
import random

import motion

def get_mid_slice(vol_data,n,patch=False,coord=None):
    sl_data = vol_data
    if(patch):
        sl_data = sl_data[coord[0]:coord[0]+n,coord[1]:coord[1]+n]
    else:
        sl_data = np.array(Image.fromarray(sl_data).resize((n,n)))
    sl_data = sl_data - sl_data.mean()
    sl_data = sl_data/np.max(sl_data)
    return sl_data

def batch_imgs(dir_name,image_names,n,corruption,corruption_extent,input_domain,output_domain,patch,num_lines=None,num_move_lines=None):
    inputs = []
    outputs = []
    for img in image_names:
        vol_data = np.load(os.path.join(dir_name,img),mmap_mode='r')['vol_data']
        coord = np.random.randint(vol_data.shape[0]-n,size=2)
        sl_data  = get_mid_slice(vol_data,n,patch,coord)
        k_line = np.random.randint(0,n)
        k_vect = np.zeros(n)
        if(corruption_extent=='CONTIGUOUS'):
            n = sl_data.shape[0]
            if(num_lines is not None):
                k_line = n-num_lines
            else:
                k_line = np.random.randint(0,n)
                k_vect[k_line] = 1
            if(corruption=='CLEAN'):
                num_pix = np.zeros(n)
                angle = np.zeros(n)
            elif(corruption=='TRANS'):
                num_pix = np.zeros(n)
                num_pix[k_line:] = np.random.random()*20-10
                angle = np.zeros(n)
            elif(corruption=='ALL'):
                #num_pix = np.zeros(n)
                #num_pix[k_line:] = np.random.random()*10
                num_pix = np.zeros((n,2))
                num_pix[k_line:,0] = np.random.random()*20-10
                num_pix[k_line:,1] = np.random.random()*20-10
                angle = np.zeros(n)
                angle[k_line:] = np.random.random()*90-45
            elif(corruption=='SEQUENTIAL'):
                frame_jump = np.random.randint(0,30)
            else:
                raise ValueError('Unrecognized motion corruption setting.')
        elif(corruption_extent=='PARTIAL'):
            move_inds = random.sample(range(1,n),num_move_lines)
            move_inds.sort()
            move_ends = move_inds.copy()
            move_ends.extend([None])
            k_vect = np.zeros(n)
            num_pix = np.zeros((n,2))
            angle = np.zeros(n)

            for i in range(len(move_inds)):
                k_vect[move_inds[i]] = 1
                num_pix[move_ends[i]:move_ends[i+1],0] = np.random.random()*20-10
                num_pix[move_ends[i]:move_ends[i+1],1] = np.random.random()*20-10
                angle[move_ends[i]:move_ends[i+1]] = np.random.random()*90-45                          
        elif(corruption_extent=='ALL'):
            num_corrupt = n
            if(corruption=='TRANS'):
                num_pix = np.random.random(size=num_corrupt)*10
                angle = np.zeros(shape=num_corrupt)
            elif(corruption=='ALL'):
                num_pix = np.random.random(size=(num_corrupt,2))*20-10
                angle = np.random.random(size=(num_corrupt))*90-45
            else:
                raise ValueError('All-line motion corruption unsupported for this corruption type.')

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
            corrupted_img,corrupted_k = motion.add_rotation_and_translation(sl_data,angle,num_pix,return_k=True)

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
        elif(output_domain=='THETA'):
            outputs.append(np.transpose(np.stack([num_pix[:,0],num_pix[:,1],angle])))
        elif(output_domain=='THETA_K'):
            outputs.append(np.transpose(np.stack([num_pix[:,0],num_pix[:,1],angle,k_vect])))
        elif(output_domain=='SINGLE_THETA'):
            outputs.append(np.expand_dims(np.array([num_pix[-1,0],num_pix[-1,1],angle[-1]]),axis=0))
        elif(output_domain=='SINGLE_THETA_K'):
            outputs.append(np.expand_dims(np.array([num_pix[-1,0],num_pix[-1,1],angle[-1],np.array(k_line)]),axis=0))
        if(input_domain=='FREQUENCY'):
            inputs.append(corrupted_k)
        elif(input_domain=='IMAGE'):
            inputs.append(np.expand_dims(corrupted_img,axis=-1))
            
    inputs = np.stack(inputs)        
    outputs = np.stack(outputs)
    return(inputs,outputs)

class DataSequence(keras.utils.Sequence):
    def __init__(self, data_path, batch_size, n, dataset, corruption, corruption_extent, input_domain, output_domain, num_lines=None, num_move_lines=None, patch=False, debug=False, num_el=-1):
        self.dir_name = data_path
        self.output_domain = output_domain
        if(dataset == 'BOLD'):
            self.img_names = []
            for s in sorted (os.listdir(data_path)):
                for v in sorted(os.listdir(os.path.join(data_path,s))):
                    self.img_names.append(os.path.join(s,v))
            if(debug):
                self.img_names = [self.img_names[0]]
        else:
            if(debug):
                num_el = 1
            self.img_names = os.listdir(data_path)[:num_el]
        self.batch_x, self.batch_y = batch_imgs(self.dir_name,self.img_names,n,corruption,corruption_extent,input_domain,output_domain,patch,num_lines,num_move_lines)
        self.batch_size = batch_size
        self.n = n

    def __len__(self):
        return int(np.ceil(len(self.batch_x)/float(self.batch_size)))

    def __getitem__(self, idx):
        if(self.output_domain in ['THETA','THETA_K','SINGLE_THETA','SINGLE_THETA_K']):
            batch_y = self.batch_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:]
            batch_x = self.batch_x[idx*self.batch_size:(idx+1)*self.batch_size,:,:]            
        else:
            batch_y = self.batch_y[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
            batch_x = self.batch_x[idx*self.batch_size:(idx+1)*self.batch_size,:,:,:]
        return batch_x, batch_y
