import PySimpleGUI as sg

from vedo import *
from vedo.applications import IsosurfaceBrowser, SlicerPlotter

import matplotlib.pyplot as plt

from GUILogging import log_in_red, log_in_green, log_in_process
from medialImageReader import extract_volume
from segmentCTImage import segment_vessel
from layoutComponents import create_slice_layout, create_param_slider_layout
from sliceMaskViewer import show_slice
from layoutUpdates import update_seg_param_val, get_directional_slices, enable_slice_viewer, update_slice_viewer

#Global variable initializations 
global  vol, imgs_after_resamp, segmentation_slices

#Global images for slices & segmentation masks
global x_imgs, y_imgs, z_imgs, x_segs, y_segs, z_segs

# control states
isExtracted = False
InSegment_process = False
segReady = False
img_type = '-NIFTI-'

# Frontend colors
color_text_window = "#443A31"
color_back_window = "white"

x_slice_layout = create_slice_layout('X')
y_slice_layout = create_slice_layout('Y')
z_slice_layout = create_slice_layout('Z')

sigma_slider = create_param_slider_layout("SIGMA", (0, 10), 0.1, 1)
alpha1_slider = create_param_slider_layout("ALPHA1", (0, 5.0), 0.1, 0.5)
alpha2_slider = create_param_slider_layout("ALPHA2", (0, 5.0), 0.1, 2.0)

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
    sigma_slider,
    alpha1_slider,
    alpha2_slider,
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
    [[
        sg.Button('Apply Segmentation', key='-SEGMENT-')
    ],
    [
            sg.Button("3D viewer #1 (surface_viewer)", key="-SURFACE_PLOTTER-"),
            sg.Button("3D viewer #2 (slice_viewer)", key="-SLICER_PLOTTER-"),
            sg.Button("3D viewer #3 (view both on different windows)", key="-SURFACE-SLICE-VIEWER-"),

    ]],
    [   sg.Column(x_slice_layout),
        sg.Column(y_slice_layout),
        sg.Column(z_slice_layout),],
]

window = sg.Window("DICOM Viewer", layout)

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
        isExtracted, imgs, img_type = values['-EXTRACTION DONE-']
        if isExtracted:
            log_in_green(window,'Volume extraction is done, You can view it by choosing one the below options !!')
            vol = Volume(imgs)
            imgs_after_resamp = imgs

            #Reset Initial state of the program
            segReady = False
            #Disable viewing threshold slider
            window['-THRES-TITLE-'].Update(visible=False)
            window['-THRESHOLD-'].Update(visible=False)
            #Disable viewing the slice with the segmentation mask
            enable_slice_viewer('X', window, False)
            enable_slice_viewer('Y', window, False)
            enable_slice_viewer('Z', window, False)

        else:
            log_in_red(window, 'Please re-enter a valid dicom/nifti image path ...')

    if event in ["-SIGMA-", "-ALPHA1-", "-ALPHA2-"]:
        update_seg_param_val(event[1:-1], isExtracted, window)

    if event == "-SEGMENT-":
        if isExtracted:
            log_in_process(window, 'Segmentation process starts ...')
            window.perform_long_operation(lambda: segment_vessel(imgs_after_resamp, values['-SIGMA-'], values['-ALPHA1-'], values['-ALPHA2-'], window, img_type), '-SEGMENTATION IS DONE-')
            InSegment_process = True;
            
        else:
            log_in_red(window, 'Volume is not ready to be viewed yet !!')

    if event == '-SEGMENTATION IS DONE-':
        InSegment_process = False
        segReady = True
        log_in_green(window,'Initial Segmentation is ready, You can view it by choosing one the below options. Tune \'threshold\' slider for better vessel extraction !!')
        
        # Initialize the slice viewers
        segmentation_slices = values['-SEGMENTATION IS DONE-']

        # Update the 3D volume to the one with segmentation
        binary_segmentation = segmentation_slices > values['-THRESHOLD-']
        vol = Volume(binary_segmentation)

        #Get the 2D slices for each direction
        x_imgs, y_imgs, z_imgs = get_directional_slices(imgs_after_resamp)
        x_segs, y_segs, z_segs = get_directional_slices(segmentation_slices)

        #Enable viewing threshold slider
        window['-THRES-TITLE-'].Update(visible=True)
        window['-THRESHOLD-'].Update(visible=True)

        #Enable viewing the slice with the segmentation mask
        enable_slice_viewer('X', window, True)
        enable_slice_viewer('Y', window, True)
        enable_slice_viewer('Z', window, True)

        # Update viewing the slice with the segmentation mask
        update_slice_viewer('X', x_imgs, x_segs, values['-THRESHOLD-'], window)
        update_slice_viewer('Y', y_imgs, y_segs, values['-THRESHOLD-'], window)
        update_slice_viewer('Z', z_imgs, z_segs, values['-THRESHOLD-'], window)

    #===============SLICE VIEWER SLICE UPDATE==================
    if event == "-X_SLIDER-":
        window["-SLICE_X-"].update(data=show_slice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])],values["-THRESHOLD-"]))
    if event == "-Y_SLIDER-":
        window["-SLICE_Y-"].update(data=show_slice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])],values["-THRESHOLD-"]))
    if event == "-Z_SLIDER-":
        window["-SLICE_Z-"].update(data=show_slice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])],values["-THRESHOLD-"]))

    #=============== On THRESHOLD UPDATE ===============
    if event == '-THRESHOLD-':
        if isExtracted and segReady:
            window["-SLICE_X-"].update(data=show_slice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])],values["-THRESHOLD-"]))
            window["-SLICE_Y-"].update(data=show_slice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])],values["-THRESHOLD-"]))
            window["-SLICE_Z-"].update(data=show_slice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])],values["-THRESHOLD-"]))
            binary_segmentation = segmentation_slices > values['-THRESHOLD-']
            vol = Volume(binary_segmentation)

    #=============== VOLUME PLOTTING ==================
    if event == "-SLICER_PLOTTER-":
        # Slices shower
            if isExtracted and not InSegment_process:
                plt = SlicerPlotter(vol,
                                    bg='white', bg2='lightblue',
                                    cmaps=['gist_ncar_r', 'hot_r', 'bone_r', 'jet', 'Spectral_r'],
                                    useSlider3D=False, alpha=1
                                )
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