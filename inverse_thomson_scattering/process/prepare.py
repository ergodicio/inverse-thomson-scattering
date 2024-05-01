from typing import Dict

import numpy as np
from inverse_thomson_scattering.process.evaluate_background import get_shot_bg
from inverse_thomson_scattering.data_handleing.load_ts_data import loadData
from inverse_thomson_scattering.process.correct_throughput import correctThroughput
from inverse_thomson_scattering.data_handleing.calibration import get_calibrations, get_scattering_angles
from inverse_thomson_scattering.process.lineouts import get_lineouts
from inverse_thomson_scattering.data_handleing.data_visualizer import launch_data_visualizer


def prepare_data(config: Dict) -> Dict:
    """
    Loads and preprocesses the data for fitting

    Args:
        config:

    Returns:

    """
    # load data
    [elecData, ionData, xlab, config["other"]["extraoptions"]["spectype"]] = loadData(
        config["data"]["shotnum"], config["data"]["shotDay"], config["other"]["extraoptions"]
    )

    # get scattering angles and weights
    sa = get_scattering_angles(config)

    # Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, stddev] = get_calibrations(
        config["data"]["shotnum"], config["other"]["extraoptions"]["spectype"], config["other"]["CCDsize"]
    )
    all_axes = {"epw_x": axisxE, "epw_y": axisyE, "iaw_x": axisxI, "iaw_y": axisyI, "x_label": xlab}

    # turn off ion or electron fitting if the corresponding spectrum was not loaded
    if not config["other"]["extraoptions"]["load_ion_spec"]:
        config["other"]["extraoptions"]["fit_IAW"] = 0
        print("IAW data not loaded, omitting IAW fit")
    if not config["other"]["extraoptions"]["load_ele_spec"]:
        config["other"]["extraoptions"]["fit_EPWb"] = 0
        config["other"]["extraoptions"]["fit_EPWr"] = 0
        print("EPW data not loaded, omitting EPW fit")

    # Correct for spectral throughput
    if config["other"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(
            elecData, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"]
        )

    # load and correct background
    [BGele, BGion] = get_shot_bg(config, axisyE, elecData)

    # extract ARTS section
    if (config["data"]["lineouts"]["type"] == "range") & (config["other"]["extraoptions"]["spectype"] == "angular"):
        config["other"]["extraoptions"]["spectype"] = "angular_full"
        # config["other"]["PhysParams"]["amps"] = np.array([np.amax(elecData), 1])
        sa["angAxis"] = axisxE

        # down sample image to resolution units by summation
        ang_res_unit = config["other"]["ang_res_unit"]  # in pixels
        lam_res_unit = config["other"]["lam_res_unit"]  # in pixels

        data_res_unit = np.array(
            [np.average(elecData[i : i + lam_res_unit, :], axis=0) for i in range(0, elecData.shape[0], lam_res_unit)]
        )
        bg_res_unit = np.array(
            [np.average(BGele[i : i + lam_res_unit, :], axis=0) for i in range(0, BGele.shape[0], lam_res_unit)]
        )
        data_res_unit = np.array(
            [
                np.average(data_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, data_res_unit.shape[1], ang_res_unit)
            ]
        )
        bg_res_unit = np.array(
            [
                np.average(bg_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, bg_res_unit.shape[1], ang_res_unit)
            ]
        )

        axisyE = np.array(
            [np.average(axisyE[i : i + lam_res_unit], axis=0) for i in range(0, axisyE.shape[0], lam_res_unit)]
        )
        all_axes["epw_y"] = axisyE.reshape((-1, 1))
        axisxE = np.array(
            [np.average(axisxE[i : i + ang_res_unit], axis=0) for i in range(0, axisxE.shape[0], ang_res_unit)]
        )
        all_axes["epw_x"] = axisxE.reshape((-1, 1))
        all_data = {"e_data": data_res_unit, "e_amps": np.amax(data_res_unit, axis=1, keepdims=True)}
        all_data["i_data"] = all_data["i_amps"] = np.zeros(len(data_res_unit))
        # changed this 8-29-23 not sure how it worked with =0?
        config["other"]["PhysParams"]["noiseI"] = np.zeros(np.shape(bg_res_unit))
        config["other"]["PhysParams"]["noiseE"] = bg_res_unit
        config["other"]["CCDsize"] = np.shape(data_res_unit)
        config["data"]["lineouts"]["start"] = int(config["data"]["lineouts"]["start"] / ang_res_unit)
        config["data"]["lineouts"]["end"] = int(config["data"]["lineouts"]["end"] / ang_res_unit)

    else:
        all_data = get_lineouts(
            elecData,
            ionData,
            BGele,
            BGion,
            axisxE,
            axisxI,
            axisyE,
            axisyI,
            config["data"]["ele_t0"],
            config["data"]["ion_t0_shift"],
            xlab,
            sa,
            config,
        )

    # Lauch the data visualizer to show linout selection, not currently interactable
    if config["data"]["launch_data_visualizer"]:
        launch_data_visualizer(elecData, ionData, all_axes, config)

    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [axisyE[0], axisyE[-1]]
    config["other"]["lamrangI"] = [axisyI[0], axisyI[-1]]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

    return all_data, sa, all_axes
