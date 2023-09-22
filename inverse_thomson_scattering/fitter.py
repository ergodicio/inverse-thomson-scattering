from typing import Dict
import time
import numpy as np
import scipy.optimize as spopt

import optax, jaxopt, mlflow
from tqdm import trange

from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function
from inverse_thomson_scattering.process import prepare
from inverse_thomson_scattering.process import postprocess


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
    if config["parameters"]["fe"]["symmetric"]:
        config["velocity"] = np.linspace(0, 7, config["parameters"]["fe"]["length"])

    # get slices
    config["data"]["lineouts"]["val"] = [
        i
        for i in range(
            config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
        )
    ]

    #Warn if fe and m are fit
    if config["parameters"]["fe"]["active"] and config["parameters"]["m"]["active"]:
        print("Super-Gaussian order and distribtuion function cannot be fit simultaneously, fitting super-Gaussian order")
        config["parameters"]["fe"]["active"] = False
        
    
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
        #print(f"{batch_size=}")
        print(f"batch size = {batch_size} is not a round divisor of the number of lineouts")
        config["data"]["lineouts"]["val"] = config["data"]["lineouts"]["val"][:-(num_slices % batch_size)]
        print(f"final {num_slices % batch_size} lineouts have been removed")

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
    mlflow.set_tag("status", "preprocessing")
    config = validate_inputs(config)
    # prepare data
    all_data, sa, all_axes = prepare.prepare_data(config)

    # prepare optimizer / solver

    batch_indices = np.arange(max(len(all_data["e_data"]), len(all_data["i_data"])))
    num_batches = len(batch_indices) // config["optimizer"]["batch_size"] or 1
    mlflow.log_metrics({"setup_time": round(time.time() - t1, 2)})

    t1 = time.time()
    mlflow.set_tag("status", "minimizing")
    if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            config["optimizer"]["batch_size"] = 1
            test_batch = {
                "e_data": all_data["e_data"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
                "e_amps": all_data["e_amps"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
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

        else:
            test_batch = {k: v[config["optimizer"]["batch_size"]] for k, v in all_batches.items()}
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
                            "noise_i": config["other"]["PhysParams"]["noiseI"][inds],
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

        raw_weights = best_weights
    else:
        best_weights = {}
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            config["optimizer"]["batch_size"] = 1
            batch = {
                "e_data": all_data["e_data"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
                "e_amps": all_data["e_amps"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
                "i_data": all_data["i_data"],
                "i_amps": all_data["i_amps"],
                "noise_e": config["other"]["PhysParams"]["noiseE"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
                "noise_i": config["other"]["PhysParams"]["noiseI"][
                    config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :
                ],
            }
            for i in range(config["optimizer"]["num_mins"]):
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
                best_weights = func_dict["get_params"](res["x"], batch)
                if i == config["optimizer"]["num_mins"] - 1:
                    break
                config["parameters"]["fe"]["length"] = (
                    config["optimizer"]["refine_factor"] * config["parameters"]["fe"]["length"]
                )
                refined_v = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
                if config["parameters"]["fe"]["symmetric"]:
                    refined_v = np.linspace(0, 7, config["parameters"]["fe"]["length"])
                
                refined_fe = np.interp(refined_v, config["velocity"], np.squeeze(best_weights["fe"]))

                config["parameters"]["fe"]["val"] = refined_fe.reshape((1, -1))
                config["velocity"] = refined_v
                #config["dist_fit"]["window"]["len"] = round(config["dist_fit"]["window"]["len"]/2)
                try:
                    config["parameters"]["ne"]["val"] = best_weights["ne"].squeeze()
                except:
                    pass
                try:
                    config["parameters"]["Te"]["val"] = best_weights["Te"].squeeze()
                except:
                    pass
                try:
                    config["parameters"]["lam"]["val"] = best_weights["lam"].squeeze()
                except:
                    pass
                try:
                    config["parameters"]["amp1"]["val"] = best_weights["amp1"].squeeze()
                except:
                    pass
                try:
                    config["parameters"]["amp2"]["val"] = best_weights["amp2"].squeeze()
                except:
                    pass

                config["parameters"]["fe"]["ub"] = -0.5
                config["parameters"]["fe"]["lb"] = -50
                config["parameters"]["fe"]["lb"] = np.multiply(
                    config["parameters"]["fe"]["lb"], np.ones(config["parameters"]["fe"]["length"])
                )
                config["parameters"]["fe"]["ub"] = np.multiply(
                    config["parameters"]["fe"]["ub"], np.ones(config["parameters"]["fe"]["length"])
                )
                config["units"] = init_param_norm_and_shift(config)

            raw_weights = func_dict["unravel_pytree"](res["x"])
            overall_loss = res["fun"]
            # print(best_weights)

        else:
            batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
            overall_loss = 0.0
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
                raw_weights = best_weights
        mlflow.log_metrics({"overall loss": float(overall_loss)})

    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "postprocessing")

    final_params = postprocess.postprocess(
        config, batch_indices, all_data, all_axes, best_weights, func_dict, sa, raw_weights
    )

    return final_params
