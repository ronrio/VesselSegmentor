from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
import scipy.ndimage

from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom import dcmread
import pydicom

import numpy as np

import os
from glob import glob

dicom_f_dir = '/Users/noorio/Downloads/Fall\'21/IT_management/CQ500CT0 CQ500CT0/Unknown Study/CT 4cc sec 150cc D3D on'
output_path = working_path = "./"
g = glob(dicom_f_dir + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print('\n'.join(g[:5]))
#      
# Loop over the image files and store everything into a list.
# 

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Segment for the vessel values
    image[image < -500] = 0


    print("available intensities in the CT scan", np.unique(image))

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

id=0
# Load stack of .DICOM images
patient = load_scan(dicom_f_dir)

# Convert the HU standard values into pixel values
imgs = get_pixels_hu(patient)

# Save the images as numpy array
np.save(output_path + "fullimages_%d.npy" % (id), imgs)

# Load the data from a numpy saved array
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))


# # Re-sampling 
print("Slice Thickness: %f" % patient[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)

# Save the images as numpy array
np.save(output_path + "images_after_resampling_%d.npy" % (id), imgs_after_resamp)

# # # Visualizing the data using Vedo
# vol = Volume(imgs_after_resamp)
# # plt = IsosurfaceBrowser(vol, c='gold') # Plotter instance
# # plt.show(axes=7, bg2='lb').close()


# # Slices shower
# plt = SlicerPlotter( vol,
#                      bg='white', bg2='lightblue',
#                      cmaps=("gist_ncar_r","jet","Spectral_r","hot_r","bone_r"),
#                      useSlider3D=False,
#                    )

# #Can now add any other object to the Plotter scene:
# #plt += Text2D('some message', font='arial')

# plt.show().close()