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
