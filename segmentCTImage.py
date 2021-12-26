import itk

import PySimpleGUI as sg

import numpy as np

def segment_vessel(imgs, sigma, alpha1, alpha2, window, img_type):
    """
    Segementation algorithm for blood vessel extraction

    Args:
        imgs: medical volume slices for applying segmentation.
        sigma: first segmentation parameter for Hessian method.
        sigma: first segmentation parameter for Hessian method.
        alpha1: second segmentation parameter for Hessian method.
        alpha2: second segmentation parameter for Hessian method.
        window: the GUI window containing the threshold widget to update its min, max values of segmentation result, log container
        img_type: the type of medical image to be segmented
    Returns: 
       segmentation_slices: numpy array containing the segmentation of the medical images
    """
    # Convert back to ITK, data is copied
    imgs_itk = itk.image_from_array(imgs.astype(np.single))

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
    window['-THRESHOLD-'].update(range=(min_intensity,max_intensity))

    # Copy of itk.Image, pixel data is copied
    if img_type == '-NIFTI-':
        segmentation_slices = itk.array_from_image(vesselness_filter.GetOutput()).transpose(2,1,0) 
    else:
        segmentation_slices = itk.array_from_image(vesselness_filter.GetOutput())
    return segmentation_slices
