import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib
#matplotlib qt

def launch_data_visualizer(elecData, ionData, all_axes, config):
    if config["other"]["extraoptions"]["load_ion_spec"]:
        X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])
        
        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, ionData)
        line, = ax.plot([config["data"]["lineouts"]["start"], config["data"]["lineouts"]["start"]], [all_axes["iaw_y"][0], all_axes["iaw_y"][0]], lw=2)
        ax.set_xlabel(all_axes["x_label"])
        ax.set_ylabel("Wavelength")
        
        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax_start = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        start_slider = Slider(
            ax=ax_start,
            label='Startign Pixel',
            valmin=0,
            valmax=config["other"]["CCDsize"][0],
            valinit=config["data"]["lineouts"]["start"],
)
    if config["other"]["extraoptions"]["load_ele_spec"]:
        X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])
        
        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, elecData)
        line, = ax.plot([config["data"]["lineouts"]["start"], config["data"]["lineouts"]["start"]], [all_axes["epw_y"][0], all_axes["epw_y"][0]], lw=2)
        ax.set_xlabel(all_axes["x_label"])
        ax.set_ylabel("Wavelength")
        
        fig.subplots_adjust(left=0.25, bottom=0.25)
        ax_start = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        start_slider = Slider(
            ax=ax_start,
            label='Startign Pixel',
            valmin=0,
            valmax=config["other"]["CCDsize"][0],
            valinit=config["data"]["lineouts"]["start"],
)
        start_slider.on_changed(update)
        print("here")
        plt.show()
        print("here")


def update(val):
    line.set_xdata([start_slider.val, start_slider.val])
    fig.canvas.draw_idle()
    


#     config["data"]["lineouts"]["val"] = [
#         i
#         for i in range(
#             config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
#         )
#     ]
# num_slices = len(config["data"]["lineouts"]["val"])
#     batch_size = config["optimizer"]["batch_size"]
#     config["data"]["lineouts"]["val"] = config["data"]["lineouts"]["val"][:-(num_slices % batch_size)]