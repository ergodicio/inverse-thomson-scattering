from typing import Dict, Tuple, List
import time
import numpy as np
import pandas as pd
import scipy.optimize as spopt

import jaxopt, mlflow, optax
from tqdm import trange
from jax.flatten_util import ravel_pytree

from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.model.TSFitter import TSFitter
from inverse_thomson_scattering.process import prepare, postprocess


def init_param_norm_and_shift(config: Dict) -> Dict:
    """
    Initializes the dictionary that contains the normalization constants for all the parameters

    The parameters are all normalized from 0 to 1 in order to improve gradient flow

    Args:
        config: Dict

    Returns: Dict

    """
    lb = {}
    ub = {}
    parameters = config["parameters"]
    active_params = []
    for key in parameters.keys():
        if parameters[key]["active"]:
            active_params.append(key)
            if np.size(parameters[key]["val"]) > 1:
                lb[key] = parameters[key]["lb"] * np.ones(np.size(parameters[key]["val"]))
                ub[key] = parameters[key]["ub"] * np.ones(np.size(parameters[key]["val"]))
            else:
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


def _validate_inputs_(config: Dict) -> Dict:
    """
    This function adds derived configuration quantities that are necessary for the fitting process

    Args:
        config: Dict

    Returns: Dict

    """
    # get derived quantities
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
    if config["parameters"]["fe"]["symmetric"]:
        config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])

    # get slices
    config["data"]["lineouts"]["val"] = [
        i
        for i in range(
            config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
        )
    ]

    # create fes
    NumDistFunc = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])
    if not config["parameters"]["fe"]["val"]:
        config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))

    config["parameters"]["fe"]["lb"] = np.multiply(
        config["parameters"]["fe"]["lb"], np.ones(config["parameters"]["fe"]["length"])
    )
    config["parameters"]["fe"]["ub"] = np.multiply(
        config["parameters"]["fe"]["ub"], np.ones(config["parameters"]["fe"]["length"])
    )
    num_slices = len(config["data"]["lineouts"]["val"])
    batch_size = config["optimizer"]["batch_size"]

    if not num_slices % batch_size == 0:
        print(f"total slices: {num_slices}")
        # print(f"{batch_size=}")
        print(f"batch size = {batch_size} is not a round divisor of the number of lineouts")
        config["data"]["lineouts"]["val"] = config["data"]["lineouts"]["val"][: -(num_slices % batch_size)]
        print(f"final {num_slices % batch_size} lineouts have been removed")

    config["units"] = init_param_norm_and_shift(config)

    return config


def scipy_angular_loop(config: Dict, all_data: Dict, sa) -> Tuple[Dict, float, TSFitter]:
    """
    Performs angular thomson scattering i.e. ARTEMIS fitting exercise using the SciPy optimizer routines


    Args:
        config:
        all_data:
        best_weights:
        all_weights:
        sa:

    Returns:

    """
    print("Running Angular, setting batch_size to 1")
    config["optimizer"]["batch_size"] = 1
    batch = {
        "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": config["other"]["PhysParams"]["noiseE"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
        "noise_i": config["other"]["PhysParams"]["noiseI"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
    }

    ts_fitter = TSFitter(config, sa, batch)
    all_weights = {k: [] for k in ts_fitter.pytree_weights["active"].keys()}

    if config["optimizer"]["num_mins"] > 1:
        print(Warning("multiple num mins doesnt work. only running once"))
    for i in range(1):
        ts_fitter = TSFitter(config, sa, batch)
        init_weights = copy.deepcopy(ts_fitter.flattened_weights)

        # ts_fitter.flattened_weights = ts_fitter.flattened_weights * np.random.uniform(
        #     0.97, 1.03, len(ts_fitter.flattened_weights)
        # )
        res = spopt.minimize(
            ts_fitter.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter.loss,
            init_weights,
            args=batch,
            method=config["optimizer"]["method"],
            jac=True if config["optimizer"]["grad_method"] == "AD" else False,
            bounds=ts_fitter.bounds,
            options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
        )
        these_weights = ts_fitter.unravel_pytree(res["x"])
        for k in all_weights.keys():
            all_weights[k].append(these_weights[k])

        if i == config["optimizer"]["num_mins"] - 1:
            break
        config["parameters"]["fe"]["length"] = (
            config["optimizer"]["refine_factor"] * config["parameters"]["fe"]["length"]
        )
        refined_v = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
        if config["parameters"]["fe"]["symmetric"]:
            refined_v = np.linspace(0, 7, config["parameters"]["fe"]["length"])

        refined_fe = np.interp(refined_v, config["velocity"], np.squeeze(these_weights["fe"]))

        config["parameters"]["fe"]["val"] = refined_fe.reshape((1, -1))
        config["velocity"] = refined_v
        config["parameters"]["ne"]["val"] = these_weights["ne"].squeeze()
        config["parameters"]["Te"]["val"] = these_weights["Te"].squeeze()

        config["parameters"]["fe"]["ub"] = -0.5
        config["parameters"]["fe"]["lb"] = -50
        config["parameters"]["fe"]["lb"] = np.multiply(
            config["parameters"]["fe"]["lb"], np.ones(config["parameters"]["fe"]["length"])
        )
        config["parameters"]["fe"]["ub"] = np.multiply(
            config["parameters"]["fe"]["ub"], np.ones(config["parameters"]["fe"]["length"])
        )
        config["units"] = init_param_norm_and_shift(config)

    overall_loss = res["fun"]

    return all_weights, overall_loss, ts_fitter


def angular_adam(config, all_data, sa, batch_indices, num_batches):
    """
    This performs an SGD-like fitting routine. It is different than the SciPy routines primarily because we have to
    manage the number of iterations manually whereas the SciPy routines have their own convergence/termination process

    Args:
        config:
        all_data:
        sa:
        batch_indices:
        num_batches:

    Returns:

    """

    config["optimizer"]["batch_size"] = 1
    test_batch = {
        "e_data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "e_amps": all_data["e_amps"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
        "i_data": all_data["i_data"],
        "i_amps": all_data["i_amps"],
        "noise_e": config["other"]["PhysParams"]["noiseE"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
        "noise_i": config["other"]["PhysParams"]["noiseI"][
            config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
        ],
    }
    func_dict = get_loss_function(config, sa, test_batch)
    jaxopt_kwargs = dict(
        fun=func_dict["vg_func"], maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
    )
    opt = optax.adam(config["optimizer"]["learning_rate"])
    solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

    weights = func_dict["init_weights"]
    opt_state = solver.init_state(weights, batch=test_batch)

    # start train loop
    t1 = time.time()
    print("minimizing")
    mlflow.set_tag("status", "minimizing")

    best_loss = 1e16
    for i_epoch in (pbar := trange(config["optimizer"]["num_epochs"])):
        if config["nn"]["use"]:
            np.random.shuffle(batch_indices)
            # tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
        epoch_loss = 0.0
        weights, opt_state = solver.update(params=weights, state=opt_state, batch=test_batch)
        epoch_loss += opt_state.value
        pbar.set_description(f"Loss {epoch_loss:.2e}")

        # epoch_loss /= num_batches
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = weights

        mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)


def _1d_adam_loop_(
    config: Dict, ts_fitter: TSFitter, previous_weights: np.ndarray, batch: Dict, tbatch
) -> Tuple[float, Dict]:
    jaxopt_kwargs = dict(
        fun=ts_fitter.vg_loss, maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
    )
    opt = optax.adam(config["optimizer"]["learning_rate"])
    solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

    if previous_weights is None:
        init_weights = ts_fitter.pytree_weights["active"]
    else:
        init_weights = previous_weights

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    opt_state = solver.init_state(init_weights, batch=batch)

    best_loss = 1e16
    epoch_loss = 1e19
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
        # if config["nn"]["use"]:
        #     np.random.shuffle(batch_indices)

        init_weights, opt_state = solver.update(params=init_weights, state=opt_state, batch=batch)
        epoch_loss = opt_state.value
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = init_weights

    return best_loss, best_weights


def _1d_scipy_loop_(config: Dict, ts_fitter: TSFitter, previous_weights: np.ndarray, batch: Dict) -> Tuple[float, Dict]:
    if previous_weights is None:  # if prev, then use that, if not then use flattened weights
        init_weights = np.copy(ts_fitter.flattened_weights)
    else:
        init_weights = np.array(previous_weights)

    # if "sequential" in config["optimizer"]:
    #     if config["optimizer"]["sequential"]:
    #         if previous_weights is not None:
    #             init_weights = previous_weights

    res = spopt.minimize(
        ts_fitter.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter.loss,
        init_weights,
        args=batch,
        method=config["optimizer"]["method"],
        jac=True if config["optimizer"]["grad_method"] == "AD" else False,
        bounds=ts_fitter.bounds,
        options={"disp": True, "maxiter": 1000},
    )

    best_loss = res["fun"]
    best_weights = ts_fitter.unravel_pytree(res["x"])

    return best_loss, best_weights


def one_d_loop(
    config: Dict, all_data: Dict, sa: Tuple, batch_indices: np.ndarray, num_batches: int
) -> Tuple[List, float, TSFitter]:
    """
    This is the higher level wrapper that prepares the data and the fitting code for the 1D fits

    This function branches out into the various optimization routines for fitting.

    For now, this is either running the ADAM loop or the SciPy optimizer loop

    Args:
        config:
        all_data:
        sa:
        batch_indices:
        num_batches:

    Returns:

    """
    sample = {k: v[: config["optimizer"]["batch_size"]] for k, v in all_data.items()}
    sample = {
        "noise_e": config["other"]["PhysParams"]["noiseE"][: config["optimizer"]["batch_size"]],
        "noise_i": config["other"]["PhysParams"]["noiseI"][: config["optimizer"]["batch_size"]],
    } | sample
    ts_fitter = TSFitter(config, sa, sample)

    print("minimizing")
    mlflow.set_tag("status", "minimizing")
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
    all_weights = []
    overall_loss = 0.0
    previous_weights = None
    with trange(num_batches, unit="batch") as tbatch:
        for i_batch in tbatch:
            inds = batch_indices[i_batch]
            batch = {
                "e_data": all_data["e_data"][inds],
                "e_amps": all_data["e_amps"][inds],
                "i_data": all_data["i_data"][inds],
                "i_amps": all_data["i_amps"][inds],
                "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                "noise_i": config["other"]["PhysParams"]["noiseI"][inds],
            }

            if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
                best_loss, best_weights = _1d_adam_loop_(config, ts_fitter, previous_weights, batch, tbatch)
            else:
                # not sure why this is needed but something needs to be reset, either the weights or the bounds
                ts_fitter = TSFitter(config, sa, batch)
                best_loss, best_weights = _1d_scipy_loop_(config, ts_fitter, previous_weights, batch)

            all_weights.append(best_weights)
            mlflow.log_metrics({"batch loss": float(best_loss)}, step=i_batch)
            overall_loss += best_loss

            # ugly
            if "sequential" in config["optimizer"]:
                if config["optimizer"]["sequential"]:
                    if config["optimizer"]["method"] == "adam":
                        previous_weights = best_weights
                    else:
                        previous_weights, _ = ravel_pytree(best_weights)

    return all_weights, overall_loss, ts_fitter


def fit(config) -> Tuple[pd.DataFrame, float]:
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
    mlflow.set_tag("status", "preprocessing")
    config = _validate_inputs_(config)

    # prepare data
    all_data, sa, all_axes = prepare.prepare_data(config)
    batch_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
    num_batches = len(batch_indices) // config["optimizer"]["batch_size"] or 1
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    # perform fit
    t1 = time.time()
    mlflow.set_tag("status", "minimizing")

    if "angular" in config["other"]["extraoptions"]["spectype"]:
        fitted_weights, overall_loss, ts_fitter = scipy_angular_loop(config, all_data, sa)
    else:
        fitted_weights, overall_loss, ts_fitter = one_d_loop(config, all_data, sa, batch_indices, num_batches)

    mlflow.log_metrics({"overall loss": float(overall_loss)})
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "postprocessing")

    final_params = postprocess.postprocess(config, batch_indices, all_data, all_axes, ts_fitter, sa, fitted_weights)

    return final_params, float(overall_loss)
