from collections import defaultdict
from typing import Dict
import time, os, tempfile

from jax.flatten_util import ravel_pytree
import numpy as np
import xarray as xr
import scipy.optimize as spopt

import optax, jaxopt, pandas, mlflow
from tqdm import trange
from matplotlib import pyplot as plt

from inverse_thomson_scattering.misc import plotters
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function
from inverse_thomson_scattering.process import prepare


def init_param_norm_and_shift(config: Dict) -> Dict:
    # init_params = {}
    lb = {}
    ub = {}
    parameters = config["parameters"]
    active_params = []
    for key in parameters.keys():
        if parameters[key]["active"]:
            active_params.append(key)
            lb[key] = parameters[key]["lb"]
            ub[key] = parameters[key]["ub"]

    norms = {}
    shifts = {}
    if config["optimizer"]["parameter_norm"]:
        for k in active_params:
            norms[k] = ub[k] - lb[k]
            shifts[k] = lb[k]
    else:
        for k in active_params:
            norms[k] = 1.0
            shifts[k] = 0.0
    return {"norms": norms, "shifts": shifts, "lb": lb, "ub": ub}


def validate_inputs(config):
    # get derived quantities
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])

    # get slices
    config["data"]["lineouts"]["val"] = [
        i
        for i in range(
            config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
        )
    ]

    # create fes
    NumDistFunc = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])
    config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))
    config["parameters"]["fe"]["lb"] = np.multiply(
        config["parameters"]["fe"]["lb"], np.ones(config["parameters"]["fe"]["length"])
    )
    config["parameters"]["fe"]["ub"] = np.multiply(
        config["parameters"]["fe"]["ub"], np.ones(config["parameters"]["fe"]["length"])
    )
    num_slices = len(config["data"]["lineouts"]["val"])
    batch_size = config["optimizer"]["batch_size"]

    if not len(config["data"]["lineouts"]["val"]) % config["optimizer"]["batch_size"] == 0:
        print(f"total slices: {num_slices}")
        print(f"{batch_size=}")
        print(f"batch size = {config['optimizer']['batch_size']} is not a round divisor of the number of lineouts")
        num_batches = np.ceil(len(config["data"]["lineouts"]["val"]) / config["optimizer"]["batch_size"])
        config["optimizer"]["batch_size"] = int(len(config["data"]["lineouts"]["val"]) // num_batches)
        print(f"new batch size = {config['optimizer']['batch_size']}")

    config["units"] = init_param_norm_and_shift(config)

    return config


def fit(config):
    """
    This function fits the Thomson scattering spectral density function to experimental data, or plots specified spectra. All inputs are derived from the input dictionary config.

    Summary of additional needs:
          A wrapper to allow for multiple lineouts or shots to be analyzed and gradients to be handled
          Better way to handle data finding since the location may change with computer or on a shot day
          Better way to handle shots with multiple types of data
          Way to handle calibrations which change from one to shot day to the next and have to be recalculated frequently (adding a new function to attempt this 8/8/22)
          A way to handle the expanded ion calculation when colapsing the spectrum to pixel resolution
          A way to handle different numbers of points

    Depreciated functions that need to be restored:
       Time axis alignment with fiducials
       interactive confirmation of new table creation
       ability to generate different table names without the default values


    Args:
        config:

    Returns:

    """
    t1 = time.time()
    config = validate_inputs(config)
    # prepare data
    all_data, sa = prepare.prepare_data(config)
    all_batches = {
        "data": all_data["data"],
        "amps": all_data["amps"],
        "noise_e": config["other"]["PhysParams"]["noiseE"],
        # "noise_i": config["other"]["PhysParams"]["noiseI"][: config["optimizer"]["batch_size"]],
    }

    # prepare optimizer / solver

    batch_indices = np.arange(len(all_data["data"]))
    num_batches = len(batch_indices) // config["optimizer"]["batch_size"]
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    t1 = time.time()
    if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
        test_batch = {k: v[config["optimizer"]["batch_size"]] for k, v in all_batches.items()}
        loss_dict = get_loss_function(config, sa, test_batch)
        jaxopt_kwargs = dict(
            fun=loss_dict["vg_func"], maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
        )
        opt = optax.adam(config["optimizer"]["learning_rate"])
        solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

        weights = loss_dict["init_weights"]
        opt_state = solver.init_state(weights, batch=test_batch)

        # start train loop
        t1 = time.time()
        print("minimizing")
        mlflow.set_tag("status", "minimizing")

        epoch_loss = 1e19
        best_loss = 1e16
        for i_epoch in range(config["optimizer"]["num_epochs"]):
            if config["nn"]["use"]:
                np.random.shuffle(batch_indices)
            batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
            with trange(num_batches, unit="batch") as tbatch:
                tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
                epoch_loss = 0.0
                for i_batch in tbatch:
                    inds = batch_indices[i_batch]
                    batch = {
                        "data": all_data["data"][inds],
                        "amps": all_data["amps"][inds],
                        "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                    }
                    weights, opt_state = solver.update(params=weights, state=opt_state, batch=batch)
                    epoch_loss += opt_state.value
                    tbatch.set_postfix({"Prev Batch Loss": opt_state.value})

                epoch_loss /= num_batches
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_weights = weights

                mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)
            batch_indices = batch_indices.flatten()

    elif config["optimizer"]["method"] == "l-bfgs-b":
        best_weights = {}
        batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))

        overall_loss = 0.0
        with trange(num_batches, unit="batch") as tbatch:
            for i_batch in tbatch:
                inds = batch_indices[i_batch]
                batch = {
                    "data": all_data["data"][inds],
                    "amps": all_data["amps"][inds],
                    "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                }
                func_dict = get_loss_function(config, sa, batch)

                res = spopt.minimize(
                    func_dict["vg_func"] if config["optimizer"]["grad_method"] == "AD" else func_dict["v_func"],
                    func_dict["init_weights"],
                    args=batch,
                    method=config["optimizer"]["method"],
                    jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                    # hess=hess_fn if config["optimizer"]["hessian"] else None,
                    bounds=func_dict["bounds"],
                    options={"disp": True},
                )
                best_weights[i_batch] = func_dict["unravel_pytree"](res["x"])
                overall_loss += res["fun"]
        mlflow.log_metrics({"overall loss": float(overall_loss / num_batches)})
    else:
        raise NotImplementedError
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})

    t1 = time.time()
    final_params = postprocess(config, batch_indices, all_data, best_weights, func_dict)
    mlflow.log_metrics({"inference_time": round(time.time() - t1, 2)})

    return final_params


def recalculate_with_chosen_weights(config, batch_indices, all_data, best_weights, func_dict):
    """
    Gets parameters and the result of the full forward pass i.e. fits


    Args:
        config:
        batch_indices:
        all_data:
        best_weights:
        func_dict:

    Returns:

    """

    # Setup x0
    #xie = np.linspace(-7, 7, parameters["fe"]["length"])

    #NumDistFunc = get_num_dist_func(parameters["fe"]["type"], xie)
    #parameters["fe"]["val"] = np.log(NumDistFunc(parameters["m"]["val"]))
    #parameters["fe"]["lb"] = np.multiply(parameters["fe"]["lb"], np.ones(parameters["fe"]["length"]))
    #parameters["fe"]["ub"] = np.multiply(parameters["fe"]["ub"], np.ones(parameters["fe"]["length"]))

    #units = initialize_parameters(config)

    all_params = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            all_params[key] = np.empty(0)
    batch_indices.sort()
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    losses = np.zeros_like(batch_indices, dtype=np.float64)
    sigmas = np.zeros((all_data["data"].shape[0], len(all_params.keys())))
    fits = np.zeros((all_data["data"].shape[0], all_data["data"].shape[2]))
    for i_batch, inds in enumerate(batch_indices):
        batch = {
            "data": all_data["data"][inds],
            "amps": all_data["amps"][inds],
            "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
        }
        if not config["optimizer"]["method"] == "adam":
            these_weights = best_weights[i_batch]
        else:
            these_weights = best_weights

        loss, [ThryE, _, params] = func_dict["array_loss_fn"](these_weights, batch)
        these_params = func_dict["get_active_params"](these_weights, batch)
        hess = func_dict["h_func"](these_params, batch)
        losses[i_batch] = np.mean(loss, axis=1)

        sigmas[inds] = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
        fits[inds] = ThryE

        for k in all_params.keys():
            all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])

    return losses, fits, sigmas, all_params


def get_sigmas(keys, hess, batch_size):
    sigmas = np.zeros((batch_size, len(keys)))

    for i in range(batch_size):
        temp = np.zeros((len(keys), len(keys)))
        for k1, param in enumerate(keys):
            for k2, param2 in enumerate(keys):
                temp[k1, k2] = np.squeeze(hess[param][param2])[i, i]

        inv = np.linalg.inv(temp)

        for k1, param in enumerate(keys):
            sigmas[i, k1] = np.sqrt(inv[k1, k1])

    return sigmas


def postprocess(config, batch_indices, all_data: Dict, best_weights, func_dict):
    losses, fits, sigmas, all_params = recalculate_with_chosen_weights(
        config, batch_indices, all_data, best_weights, func_dict
    )

    losses = losses.flatten() / np.amax(all_data["data"][:, 0, :], axis=-1)
    loss_inds = losses.argsort()[::-1]
    sorted_losses = losses[loss_inds]
    sorted_fits = fits[loss_inds]
    sorted_data = all_data["data"][loss_inds]

    num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2
    with tempfile.TemporaryDirectory() as td:
        t1 = time.time()
        os.makedirs(os.path.join(td, "worst"))
        os.makedirs(os.path.join(td, "best"))

        plotters.model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds)

        mlflow.log_metrics({"plot time": round(time.time() - t1, 2)})

        # store fitted parameters
        final_params = pandas.DataFrame(all_params)
        final_params.to_csv(os.path.join(td, "learned_parameters.csv"))
        mlflow.set_tag("status", "done plotting")

        # fit vs data storage and plot
        coords = ("lineout", np.array(config["data"]["lineouts"]["val"])), ("Wavelength", np.arange(fits.shape[1]))
        dat = {"fit": fits, "data": all_data["data"][:, 0, :]}
        savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in dat.items()})
        savedata.to_netcdf(os.path.join(td, "fit_and_data.nc"))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 11)
        savedata["fit"].T.plot(ax=ax[0], cmap="gist_ncar", levels=clevs)
        savedata["data"].T.plot(ax=ax[1], cmap="gist_ncar", levels=clevs)
        fig.savefig(os.path.join(td, "fit_and_data.png"), bbox_inches="tight")

        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        lslc = slice(config["other"]["crop_window"], -config["other"]["crop_window"])
        losses = np.mean((savedata["data"][:, lslc] - savedata["fit"][:, lslc]) ** 2.0, axis=-1) / np.square(
            np.amax(savedata["data"])
        )
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
        ax.hist(losses, 128)
        ax.set_yscale("log")
        ax.set_xlabel(r"$N^{-1}~ \sum~ (\hat{y} - y)^2 / y_{max}^2$")
        ax.set_ylabel("Counts")
        ax.set_title("Normalized $L^2$ Norm of the Error")
        ax.grid()
        fig.savefig(os.path.join(td, "error_hist.png"), bbox_inches="tight")

        sigmas_ds = xr.Dataset(
            {k: xr.DataArray(sigmas[:, i], coords=(coords[0],)) for i, k in enumerate(all_params.keys())}
        )
        sigmas_ds.to_netcdf(os.path.join(td, "sigmas.nc"))

        mlflow.log_artifacts(td)

    return final_params
