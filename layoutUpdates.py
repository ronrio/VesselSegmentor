from sliceMaskViewer import show_slice
import PySimpleGUI as sg

def update_seg_param_val(param_name, isExtracted, window):
    """
    Creating a layout for each segmentation parameter

    Args:
        param_name: parameter name to be created
        is_Extracted: Boolean value to make sure that segmentation can be applied on an extracted volume
        window: GUI window containing log container for process update
    Returns:
    """
    if isExtracted:
        log_in_process(window, '{} value is updated ! Press \'Apply Segmentation\' button to segment !'.format(param_name))
    else:
        log_in_process(window, '{} value is updated !'.format(param_name))

def get_directional_slices(imgs):
    """
    Returning three image volumes of the CT scan for 2D directional views

    Args:
        imgs: numpy image volume
    Returns:
        Three volumes of the image corresponding to X, Y, Z directional views of the original volume
    """
    return imgs,imgs.transpose(1,0,2),imgs.transpose(2,0,1)

def update_slice_viewer(axis, imgs, segs, t, window):
    """
    Update the slice views based on the current selected threshold

    Args:
        axis: the directional volume view of the volume.
        imgs: the volume from which the slice is indexed.
        segs: the volume mask from which the mask is indexed to blend it with the indexed image.
        t: the index of slice to be viewed.
        window: the GUI window containing the slice view components.
    """
    window["-{}_SLIDER-".format(axis)].update(range=(0, imgs.shape[0]-1), disabled=False)
    window["-SLICE_{}-".format(axis)].update(data=show_slice(imgs[int(window["-{}_SLIDER-".format(axis)].DefaultValue)],segs[int(window["-{}_SLIDER-".format(axis)].DefaultValue)],t))

def enable_slice_viewer(axis, window, state):
    """
        Enable/Disable the slice viewers

    Args:
        axis: the directional volume view of the volume.
        window: the GUI window containing the slice view components.
        state: True = Enable Viewing, False = Disable Viewing.
    """
    window["-{}_SLIDER-".format(axis)].update(visible=state)
    window["-SLICE_{}-".format(axis)].update(visible=state)
    window["-{}_TITLE-".format(axis)].update(visible=state)
