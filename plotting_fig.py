#!/usr/bin/env python
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

import itk

import cv2
matplotlib.use('TkAgg')

color_back_window = "white"

imgs = np.load('../before_thresholding.npy')
segs = np.load('../segment_output.npy')

x_imgs = imgs
y_imgs = imgs.transpose((1,0,2))
z_imgs = imgs.transpose((2,0,1))

x_segs = segs
y_segs = segs.transpose((1,0,2))
z_segs = segs.transpose((2,0,1))

def normalizeSlice(img):
    img_shape = img.shape
    norm_img = (img.flatten().astype(np.single) - img.min()) / (img.max() - img.min())
    new_vals = (norm_img * 255).astype(np.uint)
    return new_vals.reshape(img_shape)

def segmentMultiScaleVessels(img):
        # img = normalizeSlice(img)
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
        multi_scale_filter.SetSigmaMinimum(1.0)
        multi_scale_filter.SetSigmaMaximum(10.0)
        multi_scale_filter.SetNumberOfSigmaSteps(10)

        OutputPixelType = itk.UC
        OutputImageType = itk.Image[OutputPixelType, Dimension]

        rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
        rescale_filter.SetInput(multi_scale_filter)
        print("The output shape is :",type(rescale_filter.GetOutput()))
        return itk.array_from_image(rescale_filter.GetOutput())

def showSlice(img,seg_img):
    # Convert it to binary image
    seg_img = np.where(seg_img == True, 255, 0)
    # Increase the resolution of the image using histogramic equalization
    seg_img = np.uint8(seg_img)
    img = np.uint8(img)

    img = cv2.equalizeHist(img)
    seg_img = cv2.equalizeHist(seg_img)

    print("The shape of the array image :",img.shape)
    print("The type of the array image", type(img))
    # img = normalizeSlice(img)
    # initialize empty images
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    #Create the segmenation image
    seg_mask = np.zeros(img.shape, dtype='uint8')
    seg_mask[:,:,2] = seg_img 
    values, counts = np.unique(seg_mask, return_counts=True)
    print('=========VALUES&COUNTS=========')
    print(values,counts)
    # Blending the image with the mask
    img = cv2.addWeighted(img,
                   0.4, 
                   seg_mask, 
                   0.6,1)
    img = cv2.resize(img, (320, 320))
    print("The shape of the final image:", img.shape)
    _,img = cv2.imencode(".png", img)
    return img.tobytes()

# ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

# ------------------------------- Beginning of Matplotlib helper code -----------------------

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# ------------------------------- Beginning of GUI CODE -------------------------------

# define the window layout

x_slice_layout = [
        [sg.Image(filename="", key="-SLICE_X-")],
        [sg.Slider(
                (0, imgs.shape[0] - 1),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-X_SLIDER-",
                enable_events=True,
                disabled=True),
        ]
]

y_slice_layout = [
        [sg.Image(filename="", key="-SLICE_Y-")],
        [sg.Slider(
                (0, imgs.shape[0] - 1),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-Y_SLIDER-",
                enable_events=True),
        ]
]

z_slice_layout = [
        [sg.Image(filename="", key="-SLICE_Z-")],
        [sg.Slider(
                (0, imgs.shape[0] - 1),
                0,
                1,
                orientation="h",
                size=(40, 15),
                key="-Z_SLIDER-",
                enable_events=True),
        ]
]

layout = [
        [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
        [sg.Column(x_slice_layout, background_color= color_back_window),sg.Column(y_slice_layout, background_color= color_back_window),sg.Column(z_slice_layout, background_color= color_back_window)],
        [sg.Button('Show Image', key='-IMAGE_BT-')],
        ]

# create the form and show it without the plot
window = sg.Window("OpenCV Integration", layout, location=(800, 400))

# add the plot to the window
# fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
event, values = window.read(timeout=20)
window["-SLICE_X-"].update(data=showSlice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])]))
window["-SLICE_Y-"].update(data=showSlice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])]))
window["-SLICE_Z-"].update(data=showSlice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])]))
    
while True:
    event, values = window.read(timeout=20)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # if event == "-IMAGE_BT-":
    window["-X_SLIDER-"].Update(disabled=False)

    if event == "-X_SLIDER-":
        print("Slider Value is : ", values["-X_SLIDER-"])
        window["-SLICE_X-"].update(data=showSlice(x_imgs[int(values["-X_SLIDER-"])],x_segs[int(values["-X_SLIDER-"])]))
    if event == "-Y_SLIDER-":
        print("Slider Value is : ", values["-Y_SLIDER-"])
        window["-SLICE_Y-"].update(data=showSlice(y_imgs[int(values["-Y_SLIDER-"])],y_segs[int(values["-Y_SLIDER-"])]))
    if event == "-Z_SLIDER-":
        print("Slider Value is : ", values["-Z_SLIDER-"])
        window["-SLICE_Z-"].update(data=showSlice(z_imgs[int(values["-Z_SLIDER-"])],z_segs[int(values["-Z_SLIDER-"])]))
        

window.close()