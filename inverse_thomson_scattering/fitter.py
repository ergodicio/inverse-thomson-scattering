from typing import Dict
import time, os, tempfile

import numpy as np
import xarray as xr
import scipy.optimize as spopt

import optax, jaxopt, pandas, mlflow
from tqdm import trange
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
    #This function fits the Thomson scattering spectral dnesity fucntion to experimental data, or plots specified spectra. All inputs are derived from the input dictionary config.

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
    test_batch = {
        "data": all_data["data"][: config["optimizer"]["batch_size"]],
        "amps": all_data["amps"][: config["optimizer"]["batch_size"]],
        "noise_e": config["other"]["PhysParams"]["noiseE"][: config["optimizer"]["batch_size"]],
        # "noise_i": config["other"]["PhysParams"]["noiseI"][: config["optimizer"]["batch_size"]],
    }

    # prepare optimizer / solver

    batch_indices = np.arange(len(all_data["data"]))
    num_batches = len(batch_indices) // config["optimizer"]["batch_size"]
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    t1 = time.time()
    if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
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
                batch = {
                    "data": all_data["data"][inds],
                    "amps": all_data["amps"][inds],
                    "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                }
                loss_dict = get_loss_function(config, sa, batch)
                inds = batch_indices[i_batch]
                res = spopt.minimize(
                    loss_dict["vg_func"] if config["optimizer"]["grad_method"] == "AD" else loss_dict["v_func"],
                    loss_dict["init_weights"],
                    args=batch,
                    method=config["optimizer"]["method"],
                    jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                    # hess=hess_fn if config["optimizer"]["hessian"] else None,
                    bounds=loss_dict["bounds"],
                    options={"disp": True},
                )
                best_weights[i_batch] = loss_dict["unravel_pytree"](res["x"])
                overall_loss += res["fun"]
        mlflow.log_metrics({"overall loss": float(overall_loss / num_batches)})
    else:
        raise NotImplementedError
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})

    t1 = time.time()
    final_params = postprocess(config, batch_indices, all_data, best_weights, loss_dict["array_loss_fn"])
    mlflow.log_metrics({"inference_time": round(time.time() - t1, 2)})

    return final_params


def get_inferences(config, batch_indices, all_data, best_weights, func, empty_params):
    batch_indices.sort()
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    losses = np.zeros_like(batch_indices, dtype=np.float64)
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
        loss, [ThryE, _, params] = func(these_weights, batch)
        losses[i_batch] = np.mean(loss, axis=1)
        fits[inds] = ThryE
        for k in empty_params.keys():
            empty_params[k] = np.concatenate([empty_params[k], params[k].reshape(-1)])

    return losses, fits, empty_params


def postprocess(config, batch_indices, all_data: Dict, best_weights, array_loss_fn):
    all_params = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            all_params[key] = np.empty(0)

    losses, fits, all_params = get_inferences(config, batch_indices, all_data, best_weights, array_loss_fn, all_params)

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

        all_params["lineout"] = config["data"]["lineouts"]["val"]
        final_params = pandas.DataFrame(all_params)
        final_params.to_csv(os.path.join(td, "learned_parameters.csv"))
        mlflow.set_tag("status", "done plotting")

        coords = ("lineout", np.array(config["data"]["lineouts"]["val"])), ("Wavelength", np.arange(fits.shape[1]))
        dat = {"fit": fits, "data": all_data["data"][:, 0, :]}
        savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in dat.items()})
        savedata.to_netcdf(os.path.join(td, "fit_and_data.nc"))

        mlflow.log_artifacts(td)

    return final_params
