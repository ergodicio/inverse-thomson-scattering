import time, os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mlflow, tempfile, yaml
import multiprocessing as mp
import xarray as xr
from flatten_dict import flatten, unflatten

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc.calibration import get_scattering_angles

# from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.misc.gen_num_dist_func import DistFunc
from inverse_thomson_scattering.model.TSFitter import TSFitter
from inverse_thomson_scattering.fitter import init_param_norm_and_shift
from inverse_thomson_scattering.misc import utils
from inverse_thomson_scattering.misc.calibration import get_calibrations
from inverse_thomson_scattering.misc import plotters


if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def load_and_make_folders(cfg_path: str) -> Tuple[str, Dict]:
    """
    This is used to queue runs on NERSC

    Args:
        cfg_path:

    Returns:

    """
    all_configs = {}
    basedir = os.path.join(os.getcwd(), f"{cfg_path}")
    for k in ["defaults", "inputs"]:
        with open(f"{os.path.join(basedir, k)}.yaml", "r") as fi:
            all_configs[k] = yaml.safe_load(fi)

    if "mlflow" in all_configs["inputs"].keys():
        experiment = all_configs["inputs"]["mlflow"]["experiment"]
        run_name = all_configs["inputs"]["mlflow"]["run"]

    else:
        experiment = all_configs["defaults"]["mlflow"]["experiment"]
        run_name = all_configs["defaults"]["mlflow"]["run"]

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

    return mlflow_run.info.run_id, all_configs


def run(cfg_path: str, mode: str) -> str:
    """
    Wrapper for lower level runner

    Args:
        cfg_path:
        mode:

    Returns:

    """
    run_id, all_configs = load_and_make_folders(cfg_path)
    defaults = flatten(all_configs["defaults"])
    defaults.update(flatten(all_configs["inputs"]))
    config = unflatten(defaults)
    with mlflow.start_run(run_id=run_id) as mlflow_run:
        _run_(config, mode=mode)

    return run_id


def _run_(config: Dict, mode: str = "fit"):
    """
    Either performs a forward pass or an entire fitting routine

    Relies on mlflow to log parameters, metrics, and artifacts

    Args:
        config:
        mode:

    Returns:

    """
    utils.log_params(config)
    t0 = time.time()
    if mode == "fit":
        fit_results, loss = fitter.fit(config=config)
    elif mode == "forward" or mode == "series":
        calc_series(config=config)
    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")


def run_job(run_id: str, nested: bool):
    """
    This is used to run queued runs on NERSC. It picks up the `run_id` and finds that using MLFlow and does the fitting


    Args:
        run_id:
        nested:

    Returns:

    """
    with mlflow.start_run(run_id=run_id, nested=nested) as run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            all_configs = {}
            for k in ["defaults", "inputs"]:
                dest_file_path = utils.download_file(f"{k}.yaml", run.info.artifact_uri, temp_path)
                with open(f"{os.path.join(temp_path, k)}.yaml", "r") as fi:
                    all_configs[k] = yaml.safe_load(fi)
            defaults = flatten(all_configs["defaults"])
            defaults.update(flatten(all_configs["inputs"]))
            config = unflatten(defaults)

        _run_(config, mode="fit")

    if "MLFLOW_EXPORT" in os.environ:
        utils.export_run(run_id)


#
# def calc_spec(config: Dict):
#     """
#     Just performs a forward pass
#
#
#     Args:
#         config:
#
#     Returns:
#
#     """
#     # get scattering angles and weights
#     config["optimizer"]["batch_size"] = 1
#     config["other"]["extraoptions"]["spectype"] = "temporal"
#     stddev = dict()
#     stddev["spect_stddev_ion"] = 0.015  # 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
#     stddev["spect_stddev_ele"] = 0.1  # 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
#     config["other"]["PhysParams"]["widIRF"] = stddev
#     config["other"]["lamrangE"] = [
#         config["data"]["fit_rng"]["forward_epw_start"],
#         config["data"]["fit_rng"]["forward_epw_end"],
#     ]
#     config["other"]["lamrangI"] = [
#         config["data"]["fit_rng"]["forward_iaw_start"],
#         config["data"]["fit_rng"]["forward_iaw_end"],
#     ]
#     config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
#     # config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
#     # if config["parameters"]["fe"]["symmetric"]:
#     #    config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])
#
#     dist_obj = DistFunc(config)
#     config["velocity"], config["parameters"]["fe"]["val"] = dist_obj(config["parameters"]["m"])
#     config["parameters"]["fe"]["val"] = np.log(config["parameters"]["fe"]["val"])
#     # if not config["parameters"]["fe"]["val"]:
#     #    config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))
#
#     config["units"] = init_param_norm_and_shift(config)
#
#     sas = get_scattering_angles(config)
#     dummy_batch = {
#         "i_data": np.array([1]),
#         "e_data": np.array([1]),
#         "noise_e": 0,
#         "noise_i": 0,
#         "e_amps": 1,
#         "i_amps": 1,
#     }
#     ts_fitter = TSFitter(config, sas, dummy_batch)
#     params = ts_fitter.weights_to_params(ts_fitter.pytree_weights["active"])
#     t_start = time.time()
#     ThryE, ThryI, lamAxisE, lamAxisI = ts_fitter.spec_calc(params, dummy_batch)
#     spectime = time.time() - t_start
#
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
#     if config["other"]["extraoptions"]["load_ele_spec"]:
#         ax[0].plot(lamAxisE, ThryE)
#         ax[0].set_title("Simulated Data, fontsize=14")
#         ax[0].set_ylabel("Amp (arb. units)")
#         ax[0].set_xlabel("Wavelength (nm)")
#         ax[0].grid()
#
#         coords_ele = (("Wavelength", lamAxisE),)
#         ele_dat = {"Sim": ThryE}
#         ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
#
#     if config["other"]["extraoptions"]["load_ion_spec"]:
#         ax[1].plot(lamAxisI, ThryI)
#         ax[1].set_title("Simulated Data, fontsize=14")
#         ax[1].set_ylabel("Amp (arb. units)")
#         ax[1].set_xlabel("Wavelength (nm)")
#         ax[1].grid()
#
#         coords_ion = (("Wavelength", lamAxisI),)
#         ion_dat = {"Sim": ThryI}
#         ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
#
#     with tempfile.TemporaryDirectory() as td:
#         if config["other"]["extraoptions"]["load_ion_spec"]:
#             ion_data.to_netcdf(os.path.join(td, "ion_data.nc"))
#         if config["other"]["extraoptions"]["load_ele_spec"]:
#             ele_data.to_netcdf(os.path.join(td, "ele_data.nc"))
#         fig.savefig(os.path.join(td, "simulated_data"), bbox_inches="tight")
#         mlflow.log_artifacts(td)
#         metrics_dict = {"spectrum_calc_time": spectime}
#         mlflow.log_metrics(metrics=metrics_dict)
#     plt.close(fig)


def calc_series(config):
    """
    Calculates a spectrum or series of spectra from the input deck, i.e. performs a forward pass or series of forward
     passes.


    Args:
        config: Dictionary - Configuration dictionary created from the input deck. For series of spectra contains the special
         field 'series'. This field can have up to 8 subfields [param1, vals1, param2, vals2, param3, vals3, param4, vals4].
         the param subfields are a string identifying which fields of "parameters" are to be looped over. The vals subfields
         give the values of that subfield for each spectrum in the series.

    Returns:
        Ion data, electron data, and plots are saved to mlflow

    """
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1
    config["other"]["lamrangE"] = [
        config["data"]["fit_rng"]["forward_epw_start"],
        config["data"]["fit_rng"]["forward_epw_end"],
    ]
    config["other"]["lamrangI"] = [
        config["data"]["fit_rng"]["forward_iaw_start"],
        config["data"]["fit_rng"]["forward_iaw_end"],
    ]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])

    dist_obj = DistFunc(config)
    config["velocity"], config["parameters"]["fe"]["val"] = dist_obj(config["parameters"]["m"])
    config["parameters"]["fe"]["val"] = np.log(config["parameters"]["fe"]["val"])

    config["units"] = init_param_norm_and_shift(config)

    sas = get_scattering_angles(config)
    # if (config["data"]["lineouts"]["type"] == "range") & (config["other"]["extraoptions"]["spectype"] == "angular"):
    if config["other"]["extraoptions"]["spectype"] == "angular":
        [axisxE, _, _, _, _, _] = get_calibrations(
            104000, config["other"]["extraoptions"]["spectype"], config["other"]["CCDsize"]
        )  # shot number hardcoded to get calibration
        config["other"]["extraoptions"]["spectype"] = "angular_full"

        sas["angAxis"] = axisxE
    # all_data, sas, all_axes = prepare.prepare_data(config)

    dummy_batch = {
        "i_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
        "e_data": np.ones((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1])),
        "noise_e": 0,
        "noise_i": 0,
        "e_amps": 1,
        "i_amps": 1,
    }

    if "series" in config.keys():
        serieslen = len(config["series"]["vals1"])
    else:
        serieslen = 1
    ThryE = [None] * serieslen
    ThryI = [None] * serieslen
    lamAxisE = [None] * serieslen
    lamAxisI = [None] * serieslen

    t_start = time.time()
    for i in range(serieslen):
        if "series" in config.keys():
            config["parameters"][config["series"]["param1"]]["val"] = config["series"]["vals1"][i]
            if "param2" in config["series"].keys():
                config["parameters"][config["series"]["param2"]]["val"] = config["series"]["vals2"][i]
            if "param3" in config["series"].keys():
                config["parameters"][config["series"]["param3"]]["val"] = config["series"]["vals3"][i]
            if "param4" in config["series"].keys():
                config["parameters"][config["series"]["param4"]]["val"] = config["series"]["vals4"][i]

        ts_fitter = TSFitter(config, sas, dummy_batch)
        params = ts_fitter.weights_to_params(ts_fitter.pytree_weights["active"])
        ThryE[i], ThryI[i], lamAxisE[i], lamAxisI[i] = ts_fitter.spec_calc(params, dummy_batch)

    spectime = time.time() - t_start
    ThryE = np.array(ThryE)
    ThryI = np.array(ThryI)
    lamAxisE = np.array(lamAxisE)
    lamAxisI = np.array(lamAxisI)

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        os.makedirs(os.path.join(td, "binary"), exist_ok=True)
        os.makedirs(os.path.join(td, "csv"), exist_ok=True)
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            savedata = plotters.plot_data_angular(
                config,
                {"ele": np.squeeze(ThryE)},
                {"e_data": np.zeros((config["other"]["CCDsize"][0], config["other"]["CCDsize"][1]))},
                {"epw_x": sas["angAxis"], "epw_y": lamAxisE},
                td,
            )
            plotters.plot_dist(
                config,
                {"fe": config["parameters"]["fe"]["val"], "v": config["velocity"]},
                np.zeros_like(config["parameters"]["fe"]["val"]),
                td,
            )
        else:
            print("Need to refactor lines")
        mlflow.log_artifacts(td)
        metrics_dict = {"spectrum_calc_time": spectime}
        mlflow.log_metrics(metrics=metrics_dict)

    #    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)

    # if config["other"]["extraoptions"]["load_ele_spec"]:
    #     ax[0].plot(lamAxisE.transpose(), ThryE.transpose())  # transpose might break single specs?
    #     ax[0].set_title("Simulated Data, fontsize=14")
    #     ax[0].set_ylabel("Amp (arb. units)")
    #     ax[0].set_xlabel("Wavelength (nm)")
    #     ax[0].grid()
    #
    #     if "series" in config.keys():
    #         ax[0].legend([str(ele) for ele in config["series"]["vals1"]])
    #         if config["series"]["param1"] == "fract" or config["series"]["param1"] == "Z":
    #             coords_ele = (("series", np.array(config["series"]["vals1"])[:, 0]), ("Wavelength", lamAxisE[0, :]))
    #         else:
    #             coords_ele = (("series", config["series"]["vals1"]), ("Wavelength", lamAxisE[0, :]))
    #         ele_dat = {"Sim": ThryE}
    #         ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
    #     else:
    #         coords_ele = (("series", [0]), ("Wavelength", lamAxisE[0, :]))
    #         ele_dat = {"Sim": ThryE}
    #         ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
    #
    # if config["other"]["extraoptions"]["load_ion_spec"]:
    #     ax[1].plot(lamAxisI.transpose(), ThryI.transpose())
    #     ax[1].set_title("Simulated Data, fontsize=14")
    #     ax[1].set_ylabel("Amp (arb. units)")
    #     ax[1].set_xlabel("Wavelength (nm)")
    #     ax[1].grid()
    #
    #     if "series" in config.keys():
    #         ax[1].legend([str(ele) for ele in config["series"]["vals1"]])
    #         if config["series"]["param1"] == "fract" or config["series"]["param1"] == "Z":
    #             coords_ion = (("series", np.array(config["series"]["vals1"])[:, 0]), ("Wavelength", lamAxisI[0, :]))
    #         else:
    #             coords_ion = (("series", config["series"]["vals1"]), ("Wavelength", lamAxisI[0, :]))
    #         ion_dat = {"Sim": ThryI}
    #         ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
    #     else:
    #         coords_ion = (("series", [0]), ("Wavelength", lamAxisI[0, :]))
    #         ion_dat = {"Sim": ThryI}
    #         ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
