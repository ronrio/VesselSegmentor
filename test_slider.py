import PySimpleGUI as sg

layout = [[sg.Slider(range=(-10, 10), default_value=0, size=(10, 20), orientation='horizontal', font=("Helvetica", 15),
                     key="Slider", enable_events=True)],
          [sg.Button('Change')]]

window = sg.Window('Slider error', layout)

while True:                                 # Event Loop
    event, values = window.read() 
    if event in (None, 'Quit'):             # if user closed the window using X or clicked Quit button
        break
    if event == 'Slider':
        print("------")
        slider = window[event]
        slider_val = values['Slider']
        print("slider range: ", slider.Range)
        print("slider val: ", slider_val)
    elif event == 'Change':
        slider = window['Slider']
        slider.Update(range=(0, 5))