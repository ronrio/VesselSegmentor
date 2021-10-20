import PySimpleGUI as sg

from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter

from pydicom.pixel_data_handlers.util import apply_voi_lut
from pydicom import dcmread
import pydicom

import os

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
import scipy.ndimage

import numpy as np

# # Load the data from a numpy saved array
# id = 0
# imgs = np.load('./images_after_resampling_0.npy')
# vol = Volume(imgs)

def load_scan(path):
    #Loading and sorting the slices and determining the thickness
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

layout = [
    [
        sg.Text("DICOM Directory"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
            sg.Button("3D viewer #1 (surface_viewer)", key="-SURFACE_PLOTTER-"),
            sg.Button("3D viewer #2 (slice_viewer)", key="-SLICER_PLOTTER-"),
            sg.Button("3D viewer #3 (view both on different windows)", key="-SURFACE-SLICE-VIEWER-"),

    ],
]

window = sg.Window("DICOM Viewer", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    print(event)


    if event == "-SLICER_PLOTTER-":
    # Slices shower
        plt = SlicerPlotter(vol,
                            bg='white', bg2='lightblue',
                            cmaps=(["bone_r"]),
                            useSlider3D=False, alpha=1
                        )


        #Can now add any other object to the Plotter scene:
        # plt += Text2D('some message', font='arial')

        plt.show().close()

    if event == "-SURFACE_PLOTTER-":
        #plt = IsosurfaceBrowser(vol, c='gold') # Plotter instance

        show(vol)

    if event == "-SURFACE-SLICE-VIEWER-":

        #plt_iso = IsosurfaceBrowser(vol, c='gold') # Plotter instance

        s_plt = SlicerPlotter(vol,
                            bg='white', bg2='lightblue',
                            cmaps=(["bone_r"]),
                            useSlider3D=False,
                            )

        plt = Plotter(shape=(1, 1))
        plt.show(vol, at=0)

        s_plt.show().close()
        # Can now add any other object to the Plotter scene:
        # plt += Text2D('some message', font='arial')


    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            # Load stack of .DICOM images
            patient = load_scan(folder)
            # Convert the HU standard values into pixel values
            imgs = get_pixels_hu(patient)
            imgs_after_resamp, spacing = resample(imgs, patient, [1,1,1])
            vol = Volume(imgs_after_resamp)
        except:
            vol = Volume()
        
window.close()
