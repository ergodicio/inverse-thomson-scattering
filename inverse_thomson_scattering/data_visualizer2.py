import numpy as np
import matplotlib.pyplot as plt
import tempfile, mlflow, os

            
def launch_data_visualizer(elecData, ionData, all_axes, config):
    with tempfile.TemporaryDirectory() as td:
        #until this can be made interactive this plots all the data regions
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

            fig, ax = plt.subplots()
            ax.pcolormesh(X, Y, ionData, 
                    cmap="gist_ncar",
                    vmin=np.amin(ionData),
                    vmax=np.amax(ionData),)
            sline, = ax.plot([all_axes["iaw_x"][config["data"]["lineouts"]["start"]], all_axes["iaw_x"][config["data"]["lineouts"]["start"]]], [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]], lw=2, color = 'w')
            eline, = ax.plot([all_axes["iaw_x"][config["data"]["lineouts"]["end"]], all_axes["iaw_x"][config["data"]["lineouts"]["end"]]], [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]], lw=2, color = 'w')
            
            lamsline, = ax.plot([all_axes["iaw_x"][0], all_axes["iaw_x"][-1]], [config["data"]["fit_rng"]["iaw_min"], config["data"]["fit_rng"]["iaw_min"]], lw=2, color = 'w', linestyle = '--')
            lameline, = ax.plot([all_axes["iaw_x"][0], all_axes["iaw_x"][-1]], [config["data"]["fit_rng"]["iaw_max"], config["data"]["fit_rng"]["iaw_max"]], lw=2, color = 'w', linestyle = '--')
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength")
            fig.savefig(os.path.join(td, "ion_fit_ranges.png"), bbox_inches="tight")


        if config["other"]["extraoptions"]["load_ele_spec"]:
            X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

            fig, ax = plt.subplots()
            ax.pcolormesh(X, Y, elecData, 
                    cmap="gist_ncar",
                    vmin=np.amin(elecData),
                    vmax=np.amax(elecData),)
            sline, = ax.plot([all_axes["epw_x"][config["data"]["lineouts"]["start"]], all_axes["epw_x"][config["data"]["lineouts"]["start"]]], [all_axes["epw_y"][0], all_axes["epw_y"][-1]], lw=2, color = 'w')
            eline, = ax.plot([all_axes["epw_x"][config["data"]["lineouts"]["end"]], all_axes["epw_x"][config["data"]["lineouts"]["end"]]], [all_axes["epw_y"][0], all_axes["epw_y"][-1]], lw=2, color = 'w')
            
            lamsline, = ax.plot([all_axes["epw_x"][0], all_axes["epw_x"][-1]], [config["data"]["fit_rng"]["blue_min"], config["data"]["fit_rng"]["blue_min"]], lw=2, color = 'w', linestyle = '--')
            lameline, = ax.plot([all_axes["epw_x"][0], all_axes["epw_x"][-1]], [config["data"]["fit_rng"]["blue_max"], config["data"]["fit_rng"]["blue_max"]], lw=2, color = 'w', linestyle = '--')
            lamsline, = ax.plot([all_axes["epw_x"][0], all_axes["epw_x"][-1]], [config["data"]["fit_rng"]["red_min"], config["data"]["fit_rng"]["red_min"]], lw=2, color = 'w', linestyle = '--')
            lameline, = ax.plot([all_axes["epw_x"][0], all_axes["epw_x"][-1]], [config["data"]["fit_rng"]["red_max"], config["data"]["fit_rng"]["red_max"]], lw=2, color = 'w', linestyle = '--')
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength")
            fig.savefig(os.path.join(td, "electron_fit_ranges.png"), bbox_inches="tight")
            print("here")
            
        mlflow.log_artifacts(td)
        print("here")