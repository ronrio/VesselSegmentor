import gdcm
import PySimpleGUI as sg
import nibabel


from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter

import itk

import pydicom

import os

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
import scipy.ndimage

import numpy as np

from multiprocessing import Process
import concurrent.futures
import queue
import threading

#Global variable initializations 
global dicom_path, vol, sigma, alpha1, alpha2, imgs_after_resamp

# 'Volume is not ready to be viewed yet !!'
def log_in_red(window, msg):
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='#FF0000')
    return

# 'Volume is not ready to be viewed yet !!'
def log_in_green(window, msg):
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='#AAFF00')
    return

def log_in_process(window, msg):
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='white')
    return

def load_scan(window, path):
    global vol, imgs_after_resamp
    #Loading and sorting the slices and determining the thickness
    try:
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.InstanceNumber))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
            
        for s in slices:
            s.SliceThickness = slice_thickness
            
    except:
        #log_in_red(window, 'The path does not contain valid dicom images, please re-enter a valid path for dicom !')
        #slices = None

        try:
            nifti_dir = path + '/' + os.listdir(path)[0]
            nifti_file = nibabel.load(nifti_dir)
            imgs_after_resamp = nifti_file.get_fdata()
            vol = Volume(imgs_after_resamp)
            log_in_green(window, 'The Nifti file is loaded !')
            return 'nifti'
        except:
            log_in_green(window, 'Neither Nifti, nor DICOM file can be loaded, give right path!')
            return

    return slices



def get_pixels_hu(scans):
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
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def extract_volume(window, dicom_path):
    # Get list of files in folder
    # Load stack of .DICOM images
    global vol, imgs_after_resamp
    if os.path.isdir(dicom_path):
        log_in_process(window, 'Loading dicom slices and apply volume processing ...')
        patient = load_scan(window, dicom_path)
        # Convert the HU standard values into pixel values

        if patient == 'nifti':
            return True

        if patient == None:
            return False
        else:
            imgs = get_pixels_hu(patient)
            imgs_after_resamp, spacing = resample(imgs, patient, [1,1,1])

            vol = Volume(imgs_after_resamp)
            return True
    else:
        log_in_red(window, 'Please re-enter a valid dicom image path ...')
        return False

def segmentVessel(sigma, alpha1, alpha2):
    # Convert back to ITK, data is copied
    global vol
    imgs_itk = itk.image_from_array(imgs_after_resamp)

    hessian_image = itk.hessian_recursive_gaussian_image_filter(
        imgs_itk, sigma=sigma
    )

    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
        itk.ctype("float")
    ].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)

    max_intensity = np.amax(itk.array_from_image(vesselness_filter.GetOutput()))
    min_intensity = np.amin(itk.array_from_image(vesselness_filter.GetOutput()))
    threshold = (max_intensity-min_intensity)/4
    print("Max intensity in the segmentation result: ", np.amax(max_intensity))
    print("Min intensity in the segmentation result: ", np.amin(min_intensity))
    print("Thresholding by the value of: ", threshold)


    # Copy of itk.Image, pixel data is copied
    binary_segmentation = np.array(itk.array_from_image(vesselness_filter.GetOutput())) > threshold

    vol = Volume(binary_segmentation)
    return

layout = [
    [
        sg.Text("DICOM Directory / Directory of Nifti file"),
        sg.In(size=(100, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],

    [sg.Text(key='-OUT-',
            auto_size_text=True,
            text='Please Select the path of the dicom/nifti image from the browse button !!',
            text_color='White',
            pad = ((5,5),(5,5)),
            justification='center')],
    [
        sg.Text("Segmentation parameters", justification='center')
    ],
    [
        [sg.Text("Sigma", justification = 'center')],
        [sg.Slider(key="-SIGMA-",
        range=(0, 10),
        resolution=0.1,
        default_value=1,
        size=(100,15),
        orientation='horizontal',
        font=('Helvetica', 12),
        enable_events=True)],
    ],
    [
        [sg.Text("Alpha1", justification = 'center')],
        [sg.Slider(key="-ALPHA1-",
        range=(0.0, 5.0),
        resolution=0.1,
        default_value=0.5,
        size=(100,15),
        orientation='horizontal',
        font=('Helvetica', 12),
        enable_events=True)],
    ],
    [
        [sg.Text("Alpha2",justification = 'center')],
        [sg.Slider(key="-ALPHA2-",
        range=(0.0,5.0),
        resolution=0.1,
        default_value=2,
        size=(100,15),
        orientation='horizontal',
        font=('Helvetica', 12),
        enable_events=True)],
        
    ],
    [
        sg.Button('Apply Segmentation', key='-SEGMENT-')
    ],
    [
            sg.Button("3D viewer #1 (surface_viewer)", key="-SURFACE_PLOTTER-"),
            sg.Button("3D viewer #2 (slice_viewer)", key="-SLICER_PLOTTER-"),
            sg.Button("3D viewer #3 (view both on different windows)", key="-SURFACE-SLICE-VIEWER-"),

    ],
]

window = sg.Window("DICOM Viewer", layout)

## STARTING long run by starting a thread
# with concurrent.futures.ThreadPoolExecutor() as executor:
isExtracted = False
InSegment_process = False
while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        try:
            window.perform_long_operation(lambda: extract_volume(window,values['-FOLDER-']), '-EXTRACTION DONE-')
        except BaseException as err:
            raise(f"Unexpected {err=}, {type(err)=}")


    elif event  == '-EXTRACTION DONE-':
        isExtracted = values['-EXTRACTION DONE-']
        if isExtracted:
            log_in_green(window,'Volume extraction is done, You can view it by choosing one the below options !!')
        else:
            log_in_red(window, 'Please re-enter a valid dicom/nifti image path ...')


    if event == "-SIGMA-":
        if isExtracted:
            log_in_process(window, 'Sigma value is updated ! Press \'Apply Segmentation\' button to segment !')
        else:
            log_in_process(window, 'Sigma value is updated !')
    if event == "-ALPHA1-":
        if isExtracted:
            log_in_process(window,'Alpha1 value is changed ! Press \'Apply Segmentation\' button to segment !')
        else:
            log_in_process(window, 'Alpha1 value is updated !')

    if event == "-ALPHA2-":
        if isExtracted:
            log_in_process(window,'Alpha2 value is updated ! Press \'Apply Segmentation\' button to segment !')
        else:
            log_in_process(window,'Alpha2 value is updated !')

    if event == "-SEGMENT-":
        if isExtracted:
            log_in_process(window, 'Segmentation process starts ...')
            window.perform_long_operation(lambda: segmentVessel(values['-SIGMA-'], values['-ALPHA1-'], values['-ALPHA2-']), '-SEGMENTATION IS DONE-')
            InSegment_process = True;
        else:
            log_in_red(window, 'Volume is not ready to be viewed yet !!')


    if event == '-SEGMENTATION IS DONE-':
        InSegment_process = False
        log_in_green(window,'Segmentation is ready, You can view it by choosing one the below options !!')
    
    if event == "-SLICER_PLOTTER-":
        # Slices shower
            if isExtracted and not InSegment_process:
                plt = SlicerPlotter(vol,
                                    bg='white', bg2='lightblue',
                                    cmaps=['gist_ncar_r', 'hot_r', 'bone_r', 'jet', 'Spectral_r'],
                                    useSlider3D=False, alpha=1
                                )
                #Can now add any other object to the Plotter scene:
                # plt += Text2D('some message', font='arial')
                plt.show().close()
            else:
                log_in_red(window,'Volume is not ready to be viewed yet !!')


    if event == "-SURFACE_PLOTTER-":
        if isExtracted and not InSegment_process:
            plt = Plotter(shape=(1, 1))
            plt.show(vol)
        else:
            log_in_red(window,'Volume is not ready to be viewed yet !!')

    if event == "-SURFACE-SLICE-VIEWER-":
        if isExtracted and not InSegment_process:
            s_plt = SlicerPlotter(vol,
                            bg='white', bg2='lightblue',
                            cmaps=(['gist_ncar_r', 'hot_r', 'bone_r', 'jet', 'Spectral_r']),
                            useSlider3D=False,
                            )

            plt = Plotter(shape=(1, 1))
            plt.show(vol, at=0)

            s_plt.show().close()
        else:
            log_in_red(window,'Volume is not ready to be viewed yet !!')

window.close()
