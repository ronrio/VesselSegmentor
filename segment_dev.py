
import numpy as np

import argparse

from vedo import *

import itk
from distutils.version import StrictVersion as VS

if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

imgs = np.load('../images_after_resampling_0.npy')


parser = argparse.ArgumentParser(description="Segment blood vessels.")
# parser.add_argument("input_image")
# parser.add_argument("output_image")
parser.add_argument("--sigma", type=float, default=1.0)
parser.add_argument("--alpha1", type=fçloat, default=0.5)
parser.add_argument("--alpha2", type=float, default=2.0)
args = parser.parse_args()

# input_image = itk.imread(args.input_image, itk.ctype("float"))

# Convert back to ITK, data is copied
imgs_itk = itk.image_from_array(imgs)

hessian_image = itk.hessian_recursive_gaussian_image_filter(
    imgs_itk, sigma=args.sigma
)

print(hessian_image.shape)
print(imgs_itk.shape)

vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
    itk.ctype("float")
].New()
vesselness_filter.SetInput(hessian_image)
vesselness_filter.SetAlpha1(args.alpha1)
vesselness_filter.SetAlpha2(args.alpha2)


# Copy of itk.Image, pixel data is copied
np_copy = itk.array_from_image(vesselness_filter.GetOutput())

vol = Volume(np_copy)
plt = Plotter(shape=(1, 1))
plt.show(vol, at=0)

