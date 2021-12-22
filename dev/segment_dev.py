#!/usr/bin/env python

import argparse

import itk

import numpy as np

from distutils.version import StrictVersion as VS

from vedo import *

import matplotlib.pyplot as plt

if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

def get_pixels_hu(scans):
    image = np.stack([s for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    print(image.shape)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    print("available intensities in the CT scan", np.unique(image))

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

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


def normalizeSlice(img):
    img_shape = img.shape
    norm_img = (img.flatten().astype(np.single) - img.min()) / (img.max() - img.min())
    new_vals = (norm_img * 255).astype(np.uint).astype(np.single)
    return new_vals.reshape(img_shape)

parser = argparse.ArgumentParser(description="Segment blood vessels with multi-scale Hessian-based measure.")
parser.add_argument("--sigma_minimum", type=float, default=0.5)
parser.add_argument("--sigma_maximum", type=float, default=2.0)
parser.add_argument("--number_of_sigma_steps", type=int, default=10)
args = parser.parse_args()

imgs = np.load('../images_after_resampling_0.npy')
values, count = np.unique(imgs, return_counts=True)
print(values, count)

#Mapping values to pixel range values
sample_img = imgs[0]
seg_array = np.empty(imgs.shape, dtype=float, order='C')
for idx,img in enumerate(imgs):
    

    # plt.figure(1)
    # plt.imshow(sample_img)
    # plt.figure(2)
    # plt.imshow(my_img)
    # plt.show()

    # samp_img = itk.imread('./Sidestream_dark_field_image.png', itk.F)

    # print("The sample size image", my_img.shape)
    # values, count = np.unique(my_img, return_counts=True)
    # print("=========VALUES&COUNTS=========")
    # print(values, count)
    def segmentVessels(img):
        img = normalizeSlice(img)
        input_img = itk.image_from_array(img)
        ImageType = type(input_img)
        print("Image datatype is : ",ImageType)
        Dimension = input_img.GetImageDimension()
        # print("Image dimension are : ", Dimension)
        HessianPixelType = itk.SymmetricSecondRankTensor[itk.D,Dimension]
        HessianImageType = itk.Image[HessianPixelType, Dimension]

        objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
            HessianImageType, ImageType
        ].New()
        objectness_filter.SetBrightObject(False)
        objectness_filter.SetScaleObjectnessMeasure(False)
        objectness_filter.SetAlpha(0.5)
        objectness_filter.SetBeta(1.0)
        objectness_filter.SetGamma(5.0)

        multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
            ImageType, HessianImageType, ImageType
        ].New()
        multi_scale_filter.SetInput(input_img)
        multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
        multi_scale_filter.SetSigmaStepMethodToLogarithmic()
        multi_scale_filter.SetSigmaMinimum(args.sigma_minimum)
        multi_scale_filter.SetSigmaMaximum(args.sigma_maximum)
        multi_scale_filter.SetNumberOfSigmaSteps(args.number_of_sigma_steps)

        OutputPixelType = itk.UC
        OutputImageType = itk.Image[OutputPixelType, Dimension]

        rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
        rescale_filter.SetInput(multi_scale_filter)
        dicom_res = rescale_filter.GetOutput()


# # itk.imwrite(dicom_res, 'segmentation_result.tif')
# values, count = np.unique(seg_array, return_counts=True)
# print("Existing values and counts are: ")
# print(values, count)

# Plot the sampled image
# plt.figure(1)
# plt.imshow(my_img)
# plt.figure(2)
# plt.imshow(dicom_res)
# plt.figure(3)
# plt.imshow(sample_img)
# plt.show()
seg_array = np.where(seg_array > 20, 255, seg_array)
vol = Volume(seg_array)

plt = Plotter(shape=(1, 1))
plt.show(vol)


