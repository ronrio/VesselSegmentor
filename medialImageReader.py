import nibabel
import pydicom

from glob import glob
import os

import scipy.ndimage
import PySimpleGUI as sg

from GUILogging import log_in_red, log_in_green, log_in_process
import numpy as np

def get_pixels_hu(scans):
    """
    Convert the pixels of the image from HU medical format to 8-bit pixel image

    Args:
        scans: list of loaded medical slices
    Returns: 
        numpy array of image after conversion
    """
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    print("the shape of the DICOM image: ", image.shape)

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
    """
    Correcting the pixel sample rate based on the .Dicom file meta data

    Args:
        image: list of images to be corrected
        scan: slices contatining the meta data of the .Dicom file for sampling correction
        new_spacing: list of scale factors for sampling correction (Default: [1,1,1])
    Returns: 
        numpy array of image after sampling correction, the new spacing between slices in each direction
    """
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def load_scan(window, path):
    """
    Identiying the type of medical image contained by a path and load them

    Args:
        window: the GUI window containing the widget the carries the medical image path, log container
        path: the path of the medical image
    Returns: 
        numpy array containing the slices of the medical images, the type of the medical image
    """
    # Check if the data passed are of type DICOM or NIFTY
    isDicom = any('.dcm' in filename for filename in os.listdir(path))
    isNifty = any('.nii.gz' in filename for filename in os.listdir(path))
    if isDicom:

        # Get list of files in folder
        # Load stack of .DICOM images
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.InstanceNumber))

        #Loading and sorting the slices and determining the thickness
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
        for s in slices:
            s.SliceThickness = slice_thickness

        # Convert the HU standard values into pixel values
        if slices != None:
            imgs = get_pixels_hu(slices)
            imgs_after_resamp, spacing = resample(imgs, slices, [1,1,1])
            img_type = '-DICOM-'
        else:
            log_in_red(window, 'Patient scan is empty !!')
            return

    elif isNifty:
            nifti_dir = glob(os.path.join(path,'*.nii.gz'))[0]
            nifti_file = nibabel.load(nifti_dir)
            imgs_after_resamp = nifti_file.get_fdata()
            # vol = Volume(imgs_after_resamp)
            log_in_green(window, 'The Nifti file is loaded !')
            img_type = '-NIFTI-'
    else:
        log_in_red(window, 'Neither Nifti, nor DICOM file can be loaded, give right path!')
        return

    return imgs_after_resamp, img_type

def extract_volume(window, dicom_path):
    """
    Extracting the medical volume helper function

    Args:
        window: the GUI window containing the widget the carries the medical image path, log container
        dicom_path: the path of the medical image
    Returns: 
        boolean for indicating that extraction is successful, numpy array containing the slices of the medical images, the type of the medical image
    """
    if os.path.isdir(dicom_path):
        log_in_process(window, 'Loading dicom slices and apply volume processing ...')
        imgs, img_type = load_scan(window, dicom_path)
        return (imgs != None).any(), imgs, img_type
    else:
        log_in_red(window, 'Please re-enter a valid dicom image path ...')
        return False, imgs, img_type