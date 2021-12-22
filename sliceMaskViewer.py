import numpy as np
import cv2

def show_slice(img,seg_img,t):
    """
    Applying image blending between an indexed slice and its segmentation mask 
    as well as thresholding the mask for better vessel segments viewing.

    Args:
        img: medical image slice to blend with the segmentation mask.
        seg_img: the mask to be blended with a slice image.
        t: current threshold value for better vessel extraction.
        
    Returns: 
       Byte .PNG image of the blended mask-slice image.
    """
    # Increase the resolution of the image using histogramic equalization
    # Making the views more visible
    img = np.uint8(img)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
   
    # Applying thresholding over the image 
    seg_th_img = seg_img > t
    seg_img = np.where(seg_th_img == True, 255, 0)
    seg_img = np.uint8(seg_img)
    seg_img = cv2.equalizeHist(seg_img)

    #Create the segmenation image
    seg_mask = np.zeros(img.shape, dtype='uint8')
    seg_mask[:,:,2] = seg_img 

    # Blending the image with the mask
    img = cv2.addWeighted(img,
                   0.4, 
                   seg_mask, 
                   0.6,1)
    img = cv2.resize(img, (320, 320))

    #Convert the blended image into byte image for GUI package image viewing
    _,img = cv2.imencode(".png", img)
    return img.tobytes()