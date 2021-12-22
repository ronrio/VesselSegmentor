import itk
from itk import TubeTK as ttk

from itkwidgets import view

import numpy as np

ImageType = itk.Image[itk.F, 3]

#im1 = itk.imread("../Data/MRI-Cases/mra.mha", itk.F)
im1 = itk.imread("../dicom/study1", itk.F)
print(im1.shape)

resampler = ttk.ResampleImage.New( Input=im1, MakeHighResIso=True )
resampler.Update()
im1iso = resampler.GetOutput()


imMath = ttk.ImageMath[ImageType,ImageType].New( Input=im1iso )
imMath.Blur(1)
imBlur = imMath.GetOutput()

numSeeds = 8#40

vSeg = ttk.SegmentTubes[ImageType].New()
# vSeg.SetInput(im1iso)
# vSeg.SetVerbose(True)
# vSeg.SetMinCurvature(0.0)
# vSeg.SetMinRoundness(0.01)
# vSeg.SetRadiusInObjectSpace( 1 )
# vSeg.SetRadiusMin(1)
# vSeg.SetRadiusMax(8)
# vSeg.SetDataMinMaxLimits(50,256)
# vSeg.SetSeedMask( imBlur )
# vSeg.SetUseSeedMaskAsProbabilities(True)
# vSeg.SetSeedMaskMaximumNumberOfPoints( numSeeds )
# vSeg.ProcessSeeds()

# tubeMaskImage = vSeg.GetTubeMaskImage()

# array_img = itk.GetArrayFromVnlMatrix(tubeMaskImage)
# view(image=im1iso, label_image=tubeMaskImage)
# print(type(vSeg.GetTubeGroup()))
# SOWriter = itk.SpatialObjectWriter[3].New(vSeg.GetTubeGroup())
# SOWriter.SetFileName( "Demo-SegmentBrightVesselsd.tre" )
# SOWriter.Update()
