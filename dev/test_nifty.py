import nibabel

import itk

import numpy as np

import os 
from glob import glob

sigma = 1.0
alpha1 = 0.5
alpha2 = 2.0

path = '/Users/noorio/Desktop/medImage'
nifti_dir = glob(os.path.join(path,'*.nii'))[0]
nifti_file = nibabel.load(nifti_dir)
imgs_after_resamp = nifti_file.get_fdata()

print("Shape of the imgs for segmentation: ", imgs_after_resamp.shape)
imgs_itk = itk.image_from_array(imgs_after_resamp.astype(np.single))
print("Shape of the hessian for segmentation: ", imgs_itk)
print("The segma value is :", sigma)

hessian_image = itk.hessian_recursive_gaussian_image_filter(
    imgs_itk, sigma=sigma
)
vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
    itk.ctype("float")
].New()
vesselness_filter.SetInput(hessian_image)
vesselness_filter.SetAlpha1(alpha1)
vesselness_filter.SetAlpha2(alpha2)

#Determine the threshold, it should be between min and max depending on the values given by the user
max_intensity = np.amax(itk.array_from_image(vesselness_filter.GetOutput()))
min_intensity = np.amin(itk.array_from_image(vesselness_filter.GetOutput()))
threshold = min_intensity+(max_intensity-min_intensity)*t
print("Max intensity in the segmentation result: ", np.amax(max_intensity))
print("Min intensity in the segmentation result: ", np.amin(min_intensity))
print("Thresholding by the value of: ", threshold)


# Copy of itk.Image, pixel data is copied
binary_segmentation = np.array(itk.array_from_image(vesselness_filter.GetOutput())) > threshold
