import numpy as np
import matplotlib.pyplot as plt
import tempfile, mlflow, os


def launch_data_visualizer(elecData, ionData, all_axes, config):
    """

    Args:
        elecData:
        ionData:
        all_axes:
        config:

    Returns:

    """
    if config["data"]["lineouts"]["type"] == "ps" or config["data"]["lineouts"]["type"] == "um":
        LineoutPixelE = [
            np.argmin(abs(all_axes["epw_x"] - loc - config["data"]["ele_t0"]))
            for loc in config["data"]["lineouts"]["val"]
        ]
        IAWtime = (
            config["data"]["ion_t0_shift"] / all_axes["iaw_x"][1]
        )  # corrects the iontime to be in the same units as the lineout
        LineoutPixelI = [
            np.argmin(abs(all_axes["iaw_x"] - loc - config["data"]["ele_t0"]))
            for loc in config["data"]["lineouts"]["val"]
        ]
    elif config["data"]["lineouts"]["type"] == "pixel":
        LineoutPixelE = config["data"]["lineouts"]["val"]
        LineoutPixelI = config["data"]["lineouts"]["val"]
        IAWtime = config["data"]["ion_t0_shift"]
    else:
        raise NotImplementedError
    LineoutPixelI = np.round(np.array(LineoutPixelI) - IAWtime).astype(int)

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        # until this can be made interactive this plots all the data regions
        if config["other"]["extraoptions"]["load_ion_spec"]:
            X, Y = np.meshgrid(all_axes["iaw_x"], all_axes["iaw_y"])

            fig, ax = plt.subplots()
            ax.pcolormesh(
                X,
                Y,
                ionData,
                cmap="gist_ncar",
                vmin=np.amin(ionData),
                vmax=np.amax(ionData),
            )
            sline = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[0]], all_axes["iaw_x"][LineoutPixelI[0]]],
                [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                lw=2,
                color="w",
            )
            eline = ax.plot(
                [all_axes["iaw_x"][LineoutPixelI[-1]], all_axes["iaw_x"][LineoutPixelI[-1]]],
                [all_axes["iaw_y"][0], all_axes["iaw_y"][-1]],
                lw=2,
                color="w",
            )

            lamsline = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_min"], config["data"]["fit_rng"]["iaw_min"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lamsline = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_cf_min"], config["data"]["fit_rng"]["iaw_cf_min"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lamsline = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_cf_max"], config["data"]["fit_rng"]["iaw_cf_max"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lameline = ax.plot(
                [all_axes["iaw_x"][0], all_axes["iaw_x"][-1]],
                [config["data"]["fit_rng"]["iaw_max"], config["data"]["fit_rng"]["iaw_max"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength")
            fig.savefig(os.path.join(td, "plots", "ion_fit_ranges.png"), bbox_inches="tight")

        if config["other"]["extraoptions"]["load_ele_spec"]:
            X, Y = np.meshgrid(all_axes["epw_x"], all_axes["epw_y"])

            fig, ax = plt.subplots()
            ax.pcolormesh(
                X,
                Y,
                elecData,
                cmap="gist_ncar",
                vmin=np.amin(elecData),
                vmax=np.amax(elecData),
            )
            sline = ax.plot(
                [all_axes["epw_x"][LineoutPixelE[0]], all_axes["epw_x"][LineoutPixelE[0]]],
                [all_axes["epw_y"][0], all_axes["epw_y"][-1]],
                lw=2,
                color="w",
            )
            eline = ax.plot(
                [all_axes["epw_x"][LineoutPixelE[-1]], all_axes["epw_x"][LineoutPixelE[-1]]],
                [all_axes["epw_y"][0], all_axes["epw_y"][-1]],
                lw=2,
                color="w",
            )

            lamsline = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["blue_min"], config["data"]["fit_rng"]["blue_min"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lameline = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["blue_max"], config["data"]["fit_rng"]["blue_max"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lamsline = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["red_min"], config["data"]["fit_rng"]["red_min"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            lameline = ax.plot(
                [all_axes["epw_x"][0], all_axes["epw_x"][-1]],
                [config["data"]["fit_rng"]["red_max"], config["data"]["fit_rng"]["red_max"]],
                lw=2,
                color="w",
                linestyle="--",
            )
            ax.set_xlabel(all_axes["x_label"])
            ax.set_ylabel("Wavelength")
            fig.savefig(os.path.join(td, "plots", "electron_fit_ranges.png"), bbox_inches="tight")

        mlflow.log_artifacts(td)
