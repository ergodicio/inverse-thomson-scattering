import time, os

import numpy as np
import matplotlib.pyplot as plt
import mlflow, tempfile, yaml
import multiprocessing as mp
import xarray as xr
from flatten_dict import flatten, unflatten

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc.calibration import get_scattering_angles
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.model.loss_function import TSFitter
from inverse_thomson_scattering.fitter import init_param_norm_and_shift
from inverse_thomson_scattering.misc import utils


if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def load_and_make_folders(cfg_path):
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


def run(cfg_path, mode):
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


def _run_(config, mode="fit"):
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
    elif mode == "forward":
        calc_spec(config=config)
    elif mode == "series":
        calc_series(config=config)
    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")


def run_job(run_id, nested):
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


def calc_spec(config):
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1
    config["other"]["extraoptions"]["spectype"] = "temporal"
    stddev = dict()
    stddev["spect_stddev_ion"] = 0.015  # 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
    stddev["spect_stddev_ele"] = 0.1  # 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [config["data"]["fit_rng"]["forward_epw_start"], config["data"]["fit_rng"]["forward_epw_end"]]
    config["other"]["lamrangI"] = [config["data"]["fit_rng"]["forward_iaw_start"], config["data"]["fit_rng"]["forward_iaw_end"]]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
    if config["parameters"]["fe"]["symmetric"]:
        config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])
        
    NumDistFunc = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])
    if not config["parameters"]["fe"]["val"]:
        config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))
    
    config["units"] = init_param_norm_and_shift(config)

    sas = get_scattering_angles(config)
    dummy_batch = {
        "i_data": np.array([1]),
        "e_data": np.array([1]),
        "noise_e": 0,
        "noise_i": 0,
        "e_amps": 1,
        "i_amps": 1,
    }
    ts_fitter = TSFitter(config, sas, dummy_batch)
    params = ts_fitter.weights_to_params(ts_fitter.pytree_weights["active"])
    ThryE, ThryI, lamAxisE, lamAxisI = ts_fitter.spec_calc(params, dummy_batch)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
    ax[0].plot(lamAxisE, ThryE)
    ax[0].set_title("Simulated Data, fontsize=14")
    ax[0].set_ylabel("Amp (arb. units)")
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].grid()

    ax[1].plot(lamAxisI, ThryI)
    ax[1].set_title("Simulated Data, fontsize=14")
    ax[1].set_ylabel("Amp (arb. units)")
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].grid()

    coords_ion = (("Wavelength", lamAxisI),)
    coords_ele = (("Wavelength", lamAxisE),)
    ion_dat = {"Sim": ThryI}
    ele_dat = {"Sim": ThryE}

    ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
    ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
        
    with tempfile.TemporaryDirectory() as td:
        ion_data.to_netcdf(os.path.join(td, "ion_data.nc"))
        ele_data.to_netcdf(os.path.join(td, "ele_data.nc"))
        fig.savefig(os.path.join(td, "simulated_data"), bbox_inches="tight")
        mlflow.log_artifacts(td)
    plt.close(fig)

    
def calc_series(config):
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1
    config["other"]["extraoptions"]["spectype"] = "temporal"
    stddev = dict()
    stddev["spect_stddev_ion"] = 0.015  # 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
    stddev["spect_stddev_ele"] = 0.1  # 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [config["data"]["fit_rng"]["forward_epw_start"], config["data"]["fit_rng"]["forward_epw_end"]]
    config["other"]["lamrangI"] = [config["data"]["fit_rng"]["forward_iaw_start"], config["data"]["fit_rng"]["forward_iaw_end"]]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
    if config["parameters"]["fe"]["symmetric"]:
        config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])
        
    NumDistFunc = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])
    if not config["parameters"]["fe"]["val"]:
        config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))
    
    config["units"] = init_param_norm_and_shift(config)

    sas = get_scattering_angles(config)
    dummy_batch = {
        "i_data": np.array([1]),
        "e_data": np.array([1]),
        "noise_e": 0,
        "noise_i": 0,
        "e_amps": 1,
        "i_amps": 1,
    }
    
    ThryE=[None]*len(config["series"]["vals"])
    ThryI=[None]*len(config["series"]["vals"])
    lamAxisE=[None]*len(config["series"]["vals"])
    lamAxisI=[None]*len(config["series"]["vals"])
    for i,val in enumerate(config["series"]["vals"]):
        config["parameters"][config["series"]["param"]]["val"] = val
        
        ts_fitter = TSFitter(config, sas, dummy_batch)
        params = ts_fitter.weights_to_params(ts_fitter.pytree_weights["active"])
        ThryE[i], ThryI[i], lamAxisE[i], lamAxisI[i] = ts_fitter.spec_calc(params, dummy_batch)

    ThryE = np.array(ThryE)
    ThryI = np.array(ThryI)
    lamAxisE = np.array(lamAxisE)
    lamAxisI = np.array(lamAxisI)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
    ax[0].plot(lamAxisE.transpose(), ThryE.transpose())
    ax[0].set_title("Simulated Data, fontsize=14")
    ax[0].set_ylabel("Amp (arb. units)")
    ax[0].set_xlabel("Wavelength (nm)")
    ax[0].legend([str(ele) for ele in config["series"]["vals"]])
    ax[0].grid()

    ax[1].plot(lamAxisI.transpose(), ThryI.transpose())
    ax[1].set_title("Simulated Data, fontsize=14")
    ax[1].set_ylabel("Amp (arb. units)")
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].legend([str(ele) for ele in config["series"]["vals"]])
    ax[1].grid()

    if config["series"]["param"] == "fract":
        coords_ion = (("series",np.array(config["series"]["vals"])[:,0]),("Wavelength", lamAxisI[0,:]))
        coords_ele = (("series",np.array(config["series"]["vals"])[:,0]),("Wavelength", lamAxisE[0,:]))
    else:
        coords_ion = (("series",config["series"]["vals"]),("Wavelength", lamAxisI[0,:]))
        coords_ele = (("series",config["series"]["vals"]),("Wavelength", lamAxisE[0,:]))
    ion_dat = {"Sim": ThryI}
    ele_dat = {"Sim": ThryE}

    ion_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ion) for k, v in ion_dat.items()})
    ele_data = xr.Dataset({k: xr.DataArray(v, coords=coords_ele) for k, v in ele_dat.items()})
    #ion_data = xr.Dataset(
        
    with tempfile.TemporaryDirectory() as td:
        ion_data.to_netcdf(os.path.join(td, "ion_data.nc"))
        ele_data.to_netcdf(os.path.join(td, "ele_data.nc"))
        fig.savefig(os.path.join(td, "simulated_data"), bbox_inches="tight")
        mlflow.log_artifacts(td)
    plt.close(fig)
