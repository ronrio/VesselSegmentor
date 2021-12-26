import PySimpleGUI as sg

color_text_window = "#443A31"
color_back_window = "white"


def create_slice_layout(axis):
    """
    Creating a layout for each directional slice viewer

    Args:
        axis: the direction of slice viewer

    Returns:
        list of layout widgets of the slice viewer
    """
    return [[sg.Text("In {} direction".format(axis), font=('Helvetica', 10), justification = 'center', pad =(5,5), key = '-{}_TITLE-'.format(axis),visible=False, text_color=color_text_window, background_color=color_back_window)],
            [sg.Image(filename="", key="-SLICE_{}-".format(axis),visible=False)],
            [sg.Slider(
                    (0, 100),
                    0,
                    1,
                    orientation="h",
                    size=(46, 15),
                    key="-{}_SLIDER-".format(axis),
                    enable_events=True,
                    disabled=True,
                    visible=False),
            ]]

def create_param_slider_layout(param_name, range_val, resolution, default_val):
    """
    Creating a layout for each segmentation parameter

    Args:
        param_name: parameter name to be created
        range_val: range of values of this parameter
        reolution: step size for parameter slider
        default_val: default value assigned to the slider on its creation

    Returns:
        list of layout widgets for the parameter tuner
    """
    return [[sg.Text("{}".format(param_name), font=('Helvetica', 10),justification = 'center', pad =(5,5), text_color=color_text_window, background_color=color_back_window)],
        [sg.Slider(key="-{}-".format(param_name),
        range=range_val,
        resolution=resolution,
        default_value=default_val,
        size=(50,15),
        orientation='horizontal',
        font=('Helvetica', 10),
        enable_events=True,
        background_color = "white",
        text_color=color_text_window)]]