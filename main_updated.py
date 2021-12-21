import PySimpleGUI as sg
import nibabel
from glob import glob

from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter

import itk

import pydicom

import os

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
import scipy.ndimage

import numpy as np

import cv2

#Global variable initializations 
global dicom_path, vol, sigma, alpha1, alpha2, imgs_after_resamp, segmentation_slices, thres

#Global images for slices
global x_imgs, y_imgs, z_imgs

#Global segmentation for the slices
global x_segs, y_segs, z_segs

color_back_window = 'white'

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
            nifti_dir = glob(os.path.join(path,'*.nii'))[0]
            nifti_file = nibabel.load(nifti_dir)
            imgs_after_resamp = nifti_file.get_fdata()
            vol = Volume(imgs_after_resamp)
            log_in_green(window, 'The Nifti file is loaded !')
            return 'nifti'
        except:
            log_in_red(window, 'Neither Nifti, nor DICOM file can be loaded, give right path!')
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
            print("The patient object is empty !!")
            return False
        else:
            imgs = get_pixels_hu(patient)
            imgs_after_resamp, spacing = resample(imgs, patient, [1,1,1])

            vol = Volume(imgs_after_resamp)
            return True
    else:
        log_in_red(window, 'Please re-enter a valid dicom image path ...')
        print("I could not load the images !!")
        return False

def segmentVessel(sigma, alpha1, alpha2, t, window):
    # Convert back to ITK, data is copied
    global vol, segmentation_slices
    print("Shape of the imgs for segmentation: ", imgs_after_resamp.shape)
    imgs_itk = itk.image_from_array(imgs_after_resamp.astype(np.single))
    print("Shape of the hessian for segmentation: ", type(imgs_itk))
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
    window['-THRESHOLD-'].update(range=(min_intensity,max_intensity))
    print("Max intensity in the segmentation result: ", np.amax(max_intensity))
    print("Min intensity in the segmentation result: ", np.amin(min_intensity))


    # Copy of itk.Image, pixel data is copied
    segmentation_slices = itk.array_from_image(vesselness_filter.GetOutput()).transpose(2,1,0) 
    vol = Volume(segmentation_slices)
    return

def showSlice(img,seg_img,t):

    values, counts = np.unique(seg_img, return_counts=True)
    print('=========BEFORE SEG : VALUES&COUNTS=========')
    print(values,counts)
    img = np.uint8(img)

    # Increase the resolution of the image using histogramic equalization
    # Making the views more visible
    img = cv2.equalizeHist(img)

    # Applying thresholding over the image 

    print("The shape of the array image :",img.shape)
    print("The type of the array image", type(img))
    # img = normalizeSlice(img)
    # initialize empty images
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    seg_th_img = seg_img > t
    print("Threshold value is : ", t)
    seg_img = np.where(seg_th_img == True, 255, 0)
    seg_img = np.uint8(seg_img)
    seg_img = cv2.equalizeHist(seg_img)
    values, counts = np.unique(seg_th_img, return_counts=True)
    print('=========AFTER SEG : VALUES&COUNTS=========')
    print(values,counts)
    #Create the segmenation image
    seg_mask = np.zeros(img.shape, dtype='uint8')
    seg_mask[:,:,2] = seg_img 
    # values, counts = np.unique(seg_mask, return_counts=True)
    # print('=========MASK VALUES&COUNTS=========')
    # print(values,counts)
    # values, counts = np.unique(img, return_counts=True)
    # print('=========IMG VALUES&COUNTS=========')
    # print(values,counts)
    # Blending the image with the mask
    img = cv2.addWeighted(img,
                   0.4, 
                   seg_mask, 
                   0.6,1)
    img = cv2.resize(img, (320, 320))
    print("The shape of the final image:", img.shape)
    _,img = cv2.imencode(".png", img)
    return img.tobytes()

x_slice_layout = [
        [sg.Image(filename="", key="-SLICE_X-")],
        [sg.Slider(
                (0, 100),
                0,
                1,
                orientation="h",
                size=(46, 15),
                key="-X_SLIDER-",
                enable_events=True,
                disabled=True),
        ]
]

y_slice_layout = [
        [sg.Image(filename="", key="-SLICE_Y-")],
        [sg.Slider(
                (0, 100),
                0,
                1,
                orientation="h",
                size=(46, 15),
                key="-Y_SLIDER-",
                enable_events=True,
                disabled=True),
        ]
]

z_slice_layout = [
        [sg.Image(filename="", key="-SLICE_Z-")],
        [sg.Slider(
                (0, 100),
                0,
                1,
                orientation="h",
                size=(46, 15),
                key="-Z_SLIDER-",
                enable_events=True,disabled=True),
        ]
]

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
        [sg.Text("Alpha2",justification = 'center',)],
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
        [sg.Text("Threshold", justification='center', visible=False, key='-THRES-TITLE-')],
        [sg.Slider(key="-THRESHOLD-",
                   range=(0.0, 1.0),
                   resolution=0.01,
                   default_value=0.3,
                   size=(100, 15),
                   orientation='horizontal',
                   font=('Helvetica', 12),
                   enable_events=True,
                   visible=False)],

    ],

    [
        sg.Button('Apply Segmentation', key='-SEGMENT-')
    ],
    [
            sg.Button("3D viewer #1 (surface_viewer)", key="-SURFACE_PLOTTER-"),
            sg.Button("3D viewer #2 (slice_viewer)", key="-SLICER_PLOTTER-"),
            sg.Button("3D viewer #3 (view both on different windows)", key="-SURFACE-SLICE-VIEWER-"),

    ],
    [sg.Column(x_slice_layout, background_color= color_back_window),sg.Column(y_slice_layout, background_color= color_back_window),sg.Column(z_slice_layout, background_color= color_back_window)],
]

window = sg.Window("DICOM Viewer", layout)

## STARTING long run by starting a thread
# with concurrent.futures.ThreadPoolExecutor() as executor:
isExtracted = False
InSegment_process = False
segReady = False
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

    if event == "-THRESHOLD-":
        if isExtracted:
            log_in_process(window, 'Threshold value is updated ! Press \'Apply Segmentation\' button to segment !')
        else:
            log_in_process(window, 'Threshold value is updated !')

    if event == "-SEGMENT-":
        if isExtracted:
            log_in_process(window, 'Segmentation process starts ...')
            window.perform_long_operation(lambda: segmentVessel(values['-SIGMA-'], values['-ALPHA1-'], values['-ALPHA2-'], values['-THRESHOLD-'],window), '-SEGMENTATION IS DONE-')
            InSegment_process = True;
        else:
            log_in_red(window, 'Volume is not ready to be viewed yet !!')

    if event == '-SEGMENTATION IS DONE-':
        InSegment_process = False
        segReady = True
        log_in_green(window,'Initial Segmentation is ready, You can view it by choosing one the below options. Also, tune \'threshold\' slider for better vessel extraction !!')
        # Initialize the slice viewers
        print("Imgs shape :", imgs_after_resamp.shape)
        print("Segmentation shape :", segmentation_slices.shape)
        x_imgs = imgs_after_resamp
        y_imgs = imgs_after_resamp.transpose(1,0,2)
        z_imgs = imgs_after_resamp.transpose(2,0,1)
        x_segs = segmentation_slices
        y_segs = segmentation_slices.transpose(1,0,2)
        z_segs = segmentation_slices.transpose(2,0,1)
        #Enable viewing threshold slider
        window['-THRES-TITLE-'].Update(visible = True)
        window['-THRESHOLD-'].Update(visible=True)
        window["-X_SLIDER-"].update(range=(0, x_imgs.shape[0]-1), disabled=False)
        window["-Y_SLIDER-"].update(range=(0, y_imgs.shape[0]-1), disabled=False)
        window["-Z_SLIDER-"].update(range=(0, z_imgs.shape[0]-1), disabled=False)
        window["-SLICE_X-"].update(data=showSlice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])],values["-THRESHOLD-"]))
        window["-SLICE_Y-"].update(data=showSlice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])],values["-THRESHOLD-"]))
        window["-SLICE_Z-"].update(data=showSlice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])],values["-THRESHOLD-"]))
    
    #===============SLICE VIEWER SECTION==================
    if event == "-X_SLIDER-":
        print("Slider Range is : ", window["-X_SLIDER-"].Range)
        window["-SLICE_X-"].update(data=showSlice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])],values["-THRESHOLD-"]))
    if event == "-Y_SLIDER-":
        print("Slider Value is : ", values["-Y_SLIDER-"])
        window["-SLICE_Y-"].update(data=showSlice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])],values["-THRESHOLD-"]))
    if event == "-Z_SLIDER-":
        print("Slider Value is : ", values["-Z_SLIDER-"])
        window["-SLICE_Z-"].update(data=showSlice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])],values["-THRESHOLD-"]))

    #=============== On Threshold update ===============
    if event == '-THRESHOLD-':
        if isExtracted and segReady:
            window["-SLICE_X-"].update(data=showSlice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])],values["-THRESHOLD-"]))
            window["-SLICE_Y-"].update(data=showSlice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])],values["-THRESHOLD-"]))
            window["-SLICE_Z-"].update(data=showSlice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])],values["-THRESHOLD-"]))
            binary_segmentation = segmentation_slices > values['-THRESHOLD-']
            vol = Volume(binary_segmentation)

        
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