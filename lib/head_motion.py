import numpy as np 
import cv2
import matplotlib.pyplot as plt

def slide_head(sl,seg,pix):
    sl_erase = sl.copy()
    sl_erase[np.where(seg)] = 0
    
    sl_filled = cv2.inpaint(sl_erase.astype(np.uint16),seg.astype(np.uint8),3,cv2.INPAINT_TELEA)
    
    offset_seg = np.roll(seg,pix)
    
    sl_shifted = sl_filled.copy()
    sl_shifted[offset_seg==1] = sl[seg==1]
    
    return sl_erase,sl_filled,sl_shifted

# Induce head horizontal translation, within a slice
def add_head_slide(sl,seg,num_pix,k_line):
    sl_slide = slide_head(sl,seg,num_pix)[2]
    sl_k = np.fft.fft2(sl)
    sl_k_slide = np.fft.fft2(sl_slide)
    sl_k_combined = sl_k
    sl_k_combined[:,:k_line] = sl_k_slide[:,:k_line]
    sl_motion = np.fft.ifft2(sl_k_combined)
    return sl_slide, sl_motion

def plot_single_head_slide(sl,seg,pix):
    fig,axes = plt.subplots(1,4, figsize=[12,3])
    
    (sl_erase,sl_filled,sl_shifted) = slide_head(sl,seg,pix)
    axes[0].imshow(sl,cmap='gray')
    axes[0].set_xlabel('Original Image')
    axes[1].imshow(sl_erase,cmap='gray')
    axes[1].set_xlabel('Erased Image')
    axes[2].imshow(sl_filled,cmap='gray')
    axes[2].set_xlabel('Fast Marching Interpolation')
    axes[3].imshow(sl_shifted,cmap='gray')
    axes[3].set_xlabel('Moved Head')
    
    for i,iax in enumerate(axes.flatten()):
        iax.set_yticks([])
        iax.set_xticks([])

def plot_head_trans(sl,seg):
    fig,axes = plt.subplots( 4,4, figsize=[12,12] )
    numpixs = [0,1,3,10]
    klines = [0,15,30,60]
    for i,iax in enumerate( axes.flatten() ):
        r = i%4
        c = int((i-r)/4)
        this_img = add_head_slide(sl,seg,numpixs[r],klines[c])[1]
        iax.imshow(np.real(this_img),cmap='gray', interpolation='nearest')
        iax.set_yticks([])
        iax.set_xticks([])
        if(r==0):
            iax.set_ylabel('k-space boundary: '+str(klines[c]))
        if(c==3):
            iax.set_xlabel(str(numpixs[r])+'-px translation')
    fig.show()
