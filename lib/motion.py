import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import skimage.segmentation
import scipy.ndimage as ndimage
import scipy.signal as signal

import numpy as np
import cv2

# Induce horizontal translation, within a slice
def add_horiz_translation(sl,num_pix,k_line,return_k=False):
    sl_slide = ndimage.interpolation.shift(sl,[0,num_pix])
    sl_k = np.fft.fftshift(np.fft.fft2(sl))
    sl_k_slide = np.fft.fftshift(np.fft.fft2(sl_slide))
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_slide[:,:k_line]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    if(return_k):
        return sl_slide, sl_motion, sl_k_combined
    else:
        return sl_slide, sl_motion

def plot_horiz_trans(img):
    fig,axes = plt.subplots( 4,4, figsize=[12,12] )
    numpixs = [0,1,3,10]
    klines = [0,15,30,60]
    for i,iax in enumerate( axes.flatten() ):
        r = i%4
        c = int((i-r)/4)
        this_img = add_horiz_translation(img,numpixs[r],klines[c])[1]
        iax.imshow(np.real(this_img),cmap='gray', interpolation='nearest')
        iax.set_yticks([])
        iax.set_xticks([])
        if(r==0):
            iax.set_ylabel('k-space boundary: '+str(klines[c]))
        if(c==3):
            iax.set_xlabel(str(numpixs[r])+'-px translation')
    fig.show()

# Induce rotation, within a slice
def add_rotation(sl,angle,k_line):
    sl_rotate = ndimage.rotate(sl, angle, reshape=False)
    sl_k = np.fft.fftshift(np.fft.fft2(sl))
    sl_k_rotate = np.fft.fftshift(np.fft.fft2(sl_rotate))
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_rotate[:,:k_line]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    return sl_rotate, sl_motion


def plot_rotation(img):
    fig,axes = plt.subplots( 4,4, figsize=[12,12] )
    numdegs = [0,5,10,20]
    klines = [0,15,30,60]
    for i,iax in enumerate( axes.flatten() ):
        r = i%4
        c = int((i-r)/4)
        this_img = add_rotation(img,numdegs[r],klines[c])[1]
        iax.imshow(np.real(this_img),cmap='gray', interpolation='nearest')
        iax.set_yticks([])
        iax.set_xticks([])
        if(r==0):
            iax.set_ylabel('k-space boundary: '+str(klines[c]))
        if(c==3):
            iax.set_xlabel(str(numdegs[r])+'degree rotation')
    fig.show()

# Induce a rotation and a horizontal translation, within a slice
def add_rotation_and_translation(sl,angle,num_pix,return_k=False):
    sl_k_combined = np.empty(sl.shape,dtype='complex64')
    for i in range(sl.shape[0]):
        sl_rotate = ndimage.rotate(sl,angle[i],reshape=False)
        sl_moved = ndimage.interpolation.shift(sl_rotate,[0,num_pix[i]])
        sl_after = np.fft.fftshift(np.fft.fft2(sl_moved))
        sl_k_combined[i,:] = sl_after[i,:]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    return sl_motion, sl_k_combined

def get_pixels_to_fill(sl,angle,num_pix):
    blank = np.zeros(sl.shape)
    blank_rotate = ndimage.rotate(blank, angle, reshape=False, cval=1)
    blank_moved = ndimage.interpolation.shift(blank_rotate, [0,-num_pix], cval=1)
    return blank_moved

# Induce a rotation and a horizontal translation, within a slice, and fill it in
def add_filled_rotation_and_translation(sl,angle,num_pix,k_line,return_k=False):
    # Move and segment image
    sl_rotate = ndimage.rotate(sl, angle, reshape=False)
    sl_moved = ndimage.interpolation.shift(sl_rotate, [0,-num_pix])
    seg = get_pixels_to_fill(sl,angle,num_pix)
    
    # In-paint slid part of image
    sl_filled = cv2.inpaint(sl_moved.astype(np.uint16),seg.astype(np.uint8),3,cv2.INPAINT_TELEA)
    
    # Combine k-space representations
    sl_k = np.fft.fftshift(np.fft.fft2(sl))
    sl_k_filled = np.fft.fftshift(np.fft.fft2(sl_filled))  
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_filled[:,:k_line]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    
    if(return_k):
        return sl_filled, sl_motion, sl_k_combined
    else:
        return sl_filled, sl_motion
    
def plot_filled_motion(img):
    fig,axes = plt.subplots( 4,4, figsize=[12,12] )
    numdegs = [0,5,10,20]
    numpixs = [0,5,10, 15]
    klines = [90,120,150,180]
    for i,iax in enumerate( axes.flatten() ):
        r = i%4
        c = int((i-r)/4)
        this_img = add_filled_rotation_and_translation(img,numdegs[r],numpixs[r],klines[c])[1]
        iax.imshow(np.real(this_img),cmap='gray', interpolation='nearest')
        iax.set_yticks([])
        iax.set_xticks([])
        if(r==0):
            iax.set_ylabel('k-space boundary: '+str(klines[c]))
        if(c==3):
            iax.set_xlabel(str(numpixs[r])+'-px, '+str(numdegs[r])+'-degree')
    fig.show()

# Induce translation, through multiple slices
def add_oop_horiz_translation(img,num_pix,k_line):
    img_slide = ndimage.interpolation.shift(img,[0,0,num_pix])
    sl = img[:,:,int(img.shape[2]/2)]
    sl_slide = img_slide[:,:,int(img.shape[2]/2)]
    sl_k = np.fft.fftshift(np.fft.fft2(sl))
    sl_k_slide = np.fft.fftshift(np.fft.fft2(sl_slide))
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_slide[:,:k_line]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    return sl_slide, sl_motion

def plot_oop_horiz_trans(img):
    fig,axes = plt.subplots( 4,4, figsize=[12,12] )
    numpixs = [0,1,5,20]
    klines = [0,15,30,60]
    for i,iax in enumerate( axes.flatten() ):
        r = i%4
        c = int((i-r)/4)
        this_img = add_oop_horiz_translation(img,numpixs[r],klines[c])[1]
        iax.imshow(np.real(this_img),cmap='gray', interpolation='nearest')
        iax.set_yticks([])
        iax.set_xticks([])
        if(r==0):
            iax.set_ylabel('k-space boundary: '+str(klines[c]))
        if(c==3):
            iax.set_xlabel(str(numpixs[r])+'-px translation')
    fig.show()

# Induce motion as simulated by the next frame
def add_next_frame(sl,sl_next,k_line,return_k=False):
    sl_k = np.fft.fftshift(np.fft.fft2(sl))
    sl_k_next = np.fft.fftshift(np.fft.fft2(sl_next))
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_next[:,:k_line]
    sl_motion = np.fft.ifft2(np.fft.ifftshift(sl_k_combined))
    return sl_motion, sl_k_combined
