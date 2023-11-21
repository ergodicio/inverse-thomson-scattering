import time

import numpy as np
import matplotlib.pyplot as plt
import os, mlflow, tempfile
import multiprocessing as mp
from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc.calibration import get_scattering_angles
from inverse_thomson_scattering.model.loss_function import TSFitter
from inverse_thomson_scattering.fitter import init_param_norm_and_shift
from inverse_thomson_scattering.misc import utils


if "BASE_TEMPDIR" in os.environ:
    BASE_TEMPDIR = os.environ["BASE_TEMPDIR"]
else:
    BASE_TEMPDIR = None


def run(config, mode="fit"):
    utils.log_params(config)
    t0 = time.time()
    if mode == "fit":
        fit_results, loss = fitter.fit(config=config)
    elif mode == "forward pass":
        calc_spec(config=config)
    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")


def run_job(run_id, nested):
    with mlflow.start_run(run_id=run_id, nested=nested) as run:
        with tempfile.TemporaryDirectory(dir=BASE_TEMPDIR) as temp_path:
            cfg = utils.get_cfg(artifact_uri=run.info.artifact_uri, temp_path=temp_path)
        run(cfg)


def calc_spec(config):
    # get scattering angles and weights
    config["optimizer"]["batch_size"] = 1
    config["other"]["extraoptions"]["spectype"] = "temporal"
    stddev = dict()
    stddev["spect_stddev_ion"] = 0.015  # 0.0153  # spectral IAW IRF for 8 / 26 / 21(grating was masked)
    stddev["spect_stddev_ele"] = 0.1  # 1.4294  # spectral EPW IRF for 200um pinhole used on 8 / 26 / 21
    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [400, 700]
    config["other"]["lamrangI"] = [524, 529]
    config["other"]["npts"] = int(config["other"]["CCDsize"][1] * config["other"]["points_per_pixel"])
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
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
    ThryE, ThryI, lamAxisE, lamAxisI = ts_fitter.calculate_spectra(ts_fitter.pytree_weights, dummy_batch)

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

    with tempfile.TemporaryDirectory() as td:
        fig.savefig(os.path.join(td, "simulated_data"), bbox_inches="tight")
        mlflow.log_artifacts(td)
    plt.close(fig)
