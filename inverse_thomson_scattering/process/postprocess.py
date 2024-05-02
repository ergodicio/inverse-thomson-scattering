from typing import Dict, List

import time, tempfile, mlflow, os, copy

import numpy as np
import scipy.optimize as spopt
from jax.flatten_util import ravel_pytree

from inverse_thomson_scattering.misc import plotters
from inverse_thomson_scattering.model.TSFitter import TSFitter


def recalculate_with_chosen_weights(
    config: Dict, batch_indices, all_data: Dict, ts_fitter: TSFitter, calc_sigma: bool, fitted_weights: Dict
):
    """
    Gets parameters and the result of the full forward pass i.e. fits


    Args:
        config: Dict- configuration dictionary built from input deck
        batch_indices:
        all_data: Dict- contains the electron data, ion data, and their respective amplitudes
        ts_fitter: Instance of the TSFitter class
        fitted_weights: Dict- best values of the parameters returned by the minimizer

    Returns:

    """

    all_params = {}
    num_params = 0
    for species in config["parameters"].keys():
        all_params[species] = {}
        for key in config["parameters"][species].keys():
            if config["parameters"][species][key]["active"]:
                all_params[species][key] = np.zeros(
                    (batch_indices.flatten()[-1] + 1, np.size(config["parameters"][species][key]["val"])),
                    dtype=np.float64,
                )
                num_params += np.shape(all_params[species][key])[1]
    batch_indices.sort()
    losses = np.zeros(batch_indices.flatten()[-1] + 1, dtype=np.float64)
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))

    fits = {}
    sqdevs = {}
    fits["ion"] = np.zeros(all_data["i_data"].shape)
    sqdevs["ion"] = np.zeros(all_data["i_data"].shape)
    fits["ele"] = np.zeros(all_data["e_data"].shape)
    sqdevs["ele"] = np.zeros(all_data["e_data"].shape)

    # num_params = 0
    # for key, vec in all_params.items():
    #     num_params += np.shape(vec)[1]

    if config["other"]["extraoptions"]["load_ion_spec"]:
        sigmas = np.zeros((all_data["i_data"].shape[0], num_params))

    if config["other"]["extraoptions"]["load_ele_spec"]:
        sigmas = np.zeros((all_data["e_data"].shape[0], num_params))

    if config["other"]["extraoptions"]["spectype"] == "angular_full":
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
        losses, sqds, used_points, [ThryE, _, params] = ts_fitter.array_loss(fitted_weights, batch)
        fits["ele"] = ThryE
        sqdevs["ele"] = sqds["ele"]

        for species in all_params.keys():
            for k in all_params[species].keys():
                # all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])
                all_params[species][k] = params[species][k].reshape(-1)

        if calc_sigma:
            # this line may need to be omited since the weights may be transformed by line 77
            active_params = ts_fitter.weights_to_params(fitted_weights, return_static_params=False)
            hess = ts_fitter.h_loss_wrt_params(active_params, batch)
            sigmas = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
            print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}")

    else:
        for i_batch, inds in enumerate(batch_indices):
            batch = {
                "e_data": all_data["e_data"][inds],
                "e_amps": all_data["e_amps"][inds],
                "i_data": all_data["i_data"][inds],
                "i_amps": all_data["i_amps"][inds],
                "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                "noise_i": config["other"]["PhysParams"]["noiseI"][inds],
            }

            loss, sqds, used_points, [ThryE, ThryI, params] = ts_fitter.array_loss(fitted_weights[i_batch], batch)
            # these_params = ts_fitter.weights_to_params(fitted_weights[i_batch], return_static_params=False)

            if calc_sigma:
                hess = ts_fitter.h_loss_wrt_params(params, batch)
                print("calculating hessian finished")

            losses[inds] = loss
            sqdevs["ele"][inds] = sqds["ele"]
            sqdevs["ion"][inds] = sqds["ion"]
            if calc_sigma:
                sigmas[inds] = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
                print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}")

            fits["ele"][inds] = ThryE
            fits["ion"][inds] = ThryI

            for species in all_params.keys():
                for k in all_params[species].keys():
                    if np.size(np.shape(params[species][k])) == 3:
                        all_params[species][k][inds] = np.squeeze(params[species][k][inds])
                    else:
                        all_params[species][k][inds] = params[species][k][inds]

    return losses, sqdevs, used_points, fits, sigmas, all_params


def get_sigmas(keys: List, hess: Dict, batch_size: int) -> Dict:
    """
    Calculates the variance using the hessian with respect to the parameters and then using the hessian values
    as the inverse of the covariance matrix and then inverting that


    Args:
        keys:
        hess:
        batch_size:

    Returns:

    """
    print(keys)
    print(hess.keys())
    sizes = {key: hess[key][key].shape[1] for key in keys}
    actual_num_params = sum([v for k, v in sizes.items()])
    sigmas = np.zeros((batch_size, actual_num_params))

    for i in range(batch_size):
        temp = np.zeros((actual_num_params, actual_num_params))
        xc = 0
        for k1, param in enumerate(keys):
            yc = 0
            for k2, param2 in enumerate(keys):
                if i > 0:
                    temp[k1, k2] = np.squeeze(hess[param][param2])[i, i]
                else:
                    temp[xc : xc + sizes[param], yc : yc + sizes[param2]] = hess[param][param2][0, :, 0, :]

                yc += sizes[param2]
            xc += sizes[param]

        # print(temp)
        inv = np.linalg.inv(temp)
        # print(inv)

        for k1, param in enumerate(keys):
            sigmas[i, xc : xc + sizes[param]] = np.diag(
                np.sign(inv[xc : xc + sizes[param], xc : xc + sizes[param]])
                * np.sqrt(np.abs(inv[xc : xc + sizes[param], xc : xc + sizes[param]]))
            )
            # print(sigmas[i, k1])

    return sigmas


def postprocess(config, batch_indices, all_data: Dict, all_axes: Dict, ts_fitter, sa, fitted_weights):
    t1 = time.time()

    if config["other"]["extraoptions"]["spectype"] != "angular_full" and config["other"]["refit"]:
        losses_init, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
            config, batch_indices, all_data, ts_fitter, False, fitted_weights
        )

        # refit bad fits
        red_losses_init = losses_init / (1.1 * (used_points - len(all_params)))
        true_batch_size = config["optimizer"]["batch_size"]
        # config["optimizer"]["batch_size"] = 1
        mlflow.log_metrics({"number of fits": len(batch_indices.flatten())})
        mlflow.log_metrics({"number of refits": int(np.sum(red_losses_init > config["other"]["refit_thresh"]))})

        for i in batch_indices.flatten()[red_losses_init > config["other"]["refit_thresh"]]:
            if i == 0:
                continue

            batch = {
                "e_data": np.reshape(all_data["e_data"][i], (1, -1)),
                "e_amps": np.reshape(all_data["e_amps"][i], (1, -1)),
                "i_data": np.reshape(all_data["i_data"][i], (1, -1)),
                "i_amps": np.reshape(all_data["i_amps"][i], (1, -1)),
                "noise_e": np.reshape(config["other"]["PhysParams"]["noiseE"][i], (1, -1)),
                "noise_i": np.reshape(config["other"]["PhysParams"]["noiseI"][i], (1, -1)),
            }

            # previous_weights = {}
            temp_cfg = copy.deepcopy(config)
            temp_cfg["optimizer"]["batch_size"] = 1
            for species in fitted_weights[(i - 1) // true_batch_size].keys():
                for key in fitted_weights[(i - 1) // true_batch_size][species].keys():
                    if config["parameters"][species][key]["active"]:
                        temp_cfg["parameters"][species][key]["val"] = float(
                            fitted_weights[(i - 1) // true_batch_size][species][key][(i - 1) % true_batch_size]
                        )

            ts_fitter_refit = TSFitter(temp_cfg, sa, batch)

            # ts_fitter_refit.flattened_weights, ts_fitter_refit.unravel_pytree = ravel_pytree(previous_weights)

            res = spopt.minimize(
                ts_fitter_refit.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter_refit.loss,
                np.copy(ts_fitter_refit.flattened_weights),
                args=batch,
                method=config["optimizer"]["method"],
                jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                bounds=ts_fitter_refit.bounds,
                options={"disp": True, "maxiter": config["optimizer"]["num_epochs"]},
            )
            cur_result = ts_fitter_refit.unravel_pytree(res["x"])

            for species in cur_result.keys():
                for key in cur_result[species].keys():
                    fitted_weights[i // true_batch_size][species][key] = (
                        fitted_weights[i // true_batch_size][species][key]
                        .at[i % true_batch_size]
                        .set(cur_result[species][key][0])
                    )
                    # fitted_weights[i // true_batch_size][species][key][i % true_batch_size] = cur_result[species][key]

            # for key in fitted_weights[i // true_batch_size].keys():
            #     cur_value = cur_result[key][0, 0]
            #     new_vals = fitted_weights[i // true_batch_size][key]
            #     new_vals = new_vals.at[tuple([i % true_batch_size, 0])].set(cur_value)
            #     fitted_weights[i // true_batch_size][key] = new_vals

        # config["optimizer"]["batch_size"] = true_batch_size

    mlflow.log_metrics({"refitting time": round(time.time() - t1, 2)})

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        os.makedirs(os.path.join(td, "binary"), exist_ok=True)
        os.makedirs(os.path.join(td, "csv"), exist_ok=True)
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            best_weights_val = {}
            best_weights_std = {}
            for k, v in fitted_weights.items():
                best_weights_val[k] = np.average(v, axis=0)  # [0, :]
                best_weights_std[k] = np.std(v, axis=0)  # [0, :]
            losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
                config, batch_indices, all_data, ts_fitter, config["other"]["calc_sigmas"], best_weights_val
            )
            mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
            mlflow.set_tag("status", "plotting")
            t1 = time.time()

            final_params = plotters.get_final_params(config, best_weights_val, all_axes, td)
            if config["other"]["calc_sigmas"]:
                sigma_fe = plotters.save_sigmas_fe(final_params, best_weights_std, sigmas, td)
            else:
                sigma_fe = np.zeros_like(final_params["fe"])
            savedata = plotters.plot_data_angular(config, fits, all_data, all_axes, td)
            plotters.plot_ang_lineouts(used_points, sqdevs, losses, all_params, all_axes, savedata, td)
            plotters.plot_dist(config, final_params, sigma_fe, td)

        else:
            losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
                config, batch_indices, all_data, ts_fitter, config["other"]["calc_sigmas"], fitted_weights
            )
            mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
            mlflow.set_tag("status", "plotting")
            t1 = time.time()

            final_params = plotters.get_final_params(config, all_params, all_axes, td)
            red_losses = plotters.plot_loss_hist(config, losses, all_params, used_points, td)
            savedata = plotters.plot_ts_data(config, fits, all_data, all_axes, td)
            plotters.model_v_actual(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td)
            sigma_ds = plotters.save_sigmas_params(config, all_params, sigmas, all_axes, td)
            plotters.plot_final_params(config, all_params, sigma_ds, td)
            # plotters.plot_dist(config, final_params, sigma_fe, td)

            # final_params = plotters.plot_regular(
            #     config, losses, all_params, used_points, all_axes, fits, all_data, sqdevs, sigmas, td
            # )
        mlflow.log_artifacts(td)
    mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})

    mlflow.set_tag("status", "done plotting")

    return final_params
