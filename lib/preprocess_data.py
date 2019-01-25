import argparse
import os
import subprocess

import numpy as np
from scipy import misc
from PIL import Image

parser = argparse.ArgumentParser(description='Resample and center-crop images, and normalize images across a dataset.')
parser.add_argument('n',type=int,help ='Dimension, in pixels, to which to crop images.')
args = parser.parse_args()

n = args.n

data_path = '/data/vision/polina/scratch/nmsingh/imagenet-data'
preprocess_path = '/data/vision/polina/scratch/nmsingh/imagenet-data-preprocessed-'+str(n)

def crop_img(category,synset,i,n):
    img_path = os.path.join(data_path,category,synset,i)
    img = Image.open(img_path).convert('YCbCr')

    # Resize smallest dimension to n
    small_dim = np.argmin(img.size)
    resize = n/float(img.size[small_dim])
    new_shape = tuple(int(np.ceil(i*resize)) for i in img.size)
    img_array = np.array(img.resize(new_shape))[:,:,0]

    # Crop image
    x = int((img_array.shape[0]-n)/2.0)
    y = int((img_array.shape[1]-n)/2.0)
    cropped_img_array = img_array[x:x+n,y:y+n]

    # Save cropped image
    new_img_dir = os.path.join(preprocess_path,category,synset)
    if(not os.path.exists(new_img_dir)):
        os.makedirs(new_img_dir)
    cropped_img = Image.fromarray(cropped_img_array)
    cropped_img.save(os.path.join(new_img_dir,i))

# If the preprocessed data path already exists, make sure the user wants to overwrite it
answer = ''
if(os.path.exists(preprocess_path)):
    while answer not in ['y','n']:
        answer = input('Preprocessed data path already exists. Overwrite? [Y/N]')
        answer = answer.lower()
    if(answer=='n'):
        sys.exit()
# Otherwise, create it
else:
    os.makedirs(preprocess_path)

# Crop images and compute the global max intensity
global_max = -float('inf')
print('Cropping Images')
print('---------------')
for category in os.listdir(data_path):
    print('Category: ' + category)
    category_path = os.path.join(data_path,category)

    for synset in os.listdir(category_path):
        print('-Synset: ' + synset)
        synset_path = os.path.join(category_path,synset)

        for i in os.listdir(synset_path):
            img_max = crop_img(category,synset,i,n)
