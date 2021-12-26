#!/usr/bin/env python

from __future__ import print_function

import itk
import sys
import os

from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter


import numpy as np


dirName = '/Users/noorio/Downloads/Fall\'21/IT_management/dicom/study1'

PixelType = itk.F
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

namesGenerator = itk.GDCMSeriesFileNames.New()
namesGenerator.SetUseSeriesDetails(True)
namesGenerator.AddSeriesRestriction("0008|0021")
namesGenerator.SetGlobalWarningDisplay(False)
namesGenerator.SetDirectory(dirName)

seriesUID = namesGenerator.GetSeriesUIDs()

if len(seriesUID) < 1:
    print("No DICOMs in: " + dirName)
    sys.exit(1)

print("The directory: " + dirName)
print("Contains the following DICOM Series: ")
for uid in seriesUID:
    print(uid)

seriesFound = False
for uid in seriesUID:
    seriesIdentifier = uid
    # if args.series_name:
    #     seriesIdentifier = args.series_name
    #     seriesFound = True
    print("Reading: " + seriesIdentifier)
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)

    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.ForceOrthogonalDirectionOff()

imgs = np.load('../images_after_resampling_0.npy')
imgs = itk.image_from_array(imgs)

hessian_image_1 = itk.hessian_recursive_gaussian_image_filter(
    reader.GetOutput(), sigma=1.0
)

hessian_image_2 = itk.hessian_recursive_gaussian_image_filter(
    imgs, sigma=1.0
)

vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
    itk.ctype("float")
].New()
vesselness_filter.SetInput(hessian_image_1)
vesselness_filter.SetAlpha1(0.5)
vesselness_filter.SetAlpha2(2.0)
dicom_1 = vesselness_filter.GetOutput()
vesselness_filter.SetInput(hessian_image_2)
vesselness_filter.SetAlpha1(0.5)
vesselness_filter.SetAlpha2(2.0)
dicom_2 = vesselness_filter.GetOutput()

# if ("SITK_NOSHOW" not in os.environ):
#     sitk.Show(image, "Dicom Series")
imgs_array_1 = itk.array_from_image(dicom_1)
imgs_array_2 = itk.array_from_image(dicom_2)

vol_1 = Volume(imgs_array_1)
vol_2 = Volume(imgs_array_2)

s_plt = SlicerPlotter(vol_1,
                            bg='white', bg2='lightblue',
                            cmaps=(["bone_r"]),
                            useSlider3D=False,
                            )

plt = Plotter(shape=(1, 1))
plt.show(vol_2, at=0)

s_plt.show().close()