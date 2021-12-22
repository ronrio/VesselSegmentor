import PySimpleGUI as sg

def log_in_red(window, msg):
    """
    Logging an invalid state in Red

    Args:
        window: GUI window containing log container
        msg: The message to be displayed

    Returns:
    """
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='#FF0000')
    return 


def log_in_green(window, msg):
    """
    Logging successful state in Green

    Args:
        window: GUI window containing log container
        msg: The message to be displayed

    Returns:
    """
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='#AAFF00')
    return


def log_in_process(window, msg):
    """
    Logging a current state in White

    Args:
        window: GUI window containing log container
        msg: The message to be displayed

    Returns:
    """
    window['-OUT-'].update(msg)
    window['-OUT-'].update(text_color='white')
    return