from typing import Dict

import numpy as np
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc import plotters
from inverse_thomson_scattering.evaluate_background import get_shot_bg
from inverse_thomson_scattering.misc.load_ts_data import loadData
from inverse_thomson_scattering.process.correct_throughput import correctThroughput
from inverse_thomson_scattering.misc.calibration import get_calibrations, get_scattering_angles
from inverse_thomson_scattering.lineouts import get_lineouts


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
    sa = get_scattering_angles(config["other"]["extraoptions"]["spectype"])

    # Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = get_calibrations(
        config["data"]["shotnum"], config["other"]["extraoptions"]["spectype"], config["other"]["CCDsize"]
    )

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
        config["other"]["PhysParams"]["amps"] = np.array([np.amax(elecData), 1])
        sa["angAxis"] = axisxE

        if config["other"]["extraoptions"]["plot_raw_data"]:
            plotters.ColorPlots(
                axisxE,
                axisyE,
                conv2(elecData - BGele, np.ones([5, 5]) / 25, mode="same"),
                vmin=0,
                XLabel=xlab,
                YLabel="Wavelength (nm)",
                title="Shot : " + str(config["data"]["shotnum"]) + " : " + "TS : Corrected and background subtracted",
            )

        # down sample image to resolution units by summation
        ang_res_unit = 10  # in pixels
        lam_res_unit = 5  # in pixels

        data_res_unit = np.array(
            [np.average(elecData[i : i + lam_res_unit, :], axis=0) for i in range(0, elecData.shape[0], lam_res_unit)]
        )
        data_res_unit = np.array(
            [
                np.average(data_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, data_res_unit.shape[1], ang_res_unit)
            ]
        )
        all_data = data_res_unit
        config["other"]["PhysParams"]["noiseI"] = 0
        config["other"]["PhysParams"]["noiseE"] = BGele

    else:
        all_data = get_lineouts(
            elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, 0, IAWtime, xlab, sa, config
        )

    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [axisyE[0], axisyE[-1]]
    config["other"]["lamrangI"] = [axisyI[0], axisyI[-1]]
    config["other"]["npts"] = config["other"]["CCDsize"][0] * config["other"]["points_per_pixel"]

    return all_data, sa
