from typing import Dict

import time, tempfile, mlflow, os

import numpy as np
import scipy.optimize as spopt

from inverse_thomson_scattering.misc import plotters
from inverse_thomson_scattering.model.loss_function import TSFitter


def recalculate_with_chosen_weights(
    config: Dict, batch_indices, all_data: Dict, ts_fitter: TSFitter, calc_sigma, fitted_weights
):
    """
    Gets parameters and the result of the full forward pass i.e. fits


    Args:
        config:
        batch_indices:
        all_data:
        best_weights:
        ts_fitter:

    Returns:

    """

    all_params = {}
    # print(np.shape(config["parameters"]["Z"]["val"]))
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            # all_params[key] = np.empty(np.shape(config["parameters"][key]["val"]))
            all_params[key] = np.zeros(
                (batch_indices.flatten()[-1] + 1, np.size(config["parameters"][key]["val"])), dtype=np.float64
            )
            # print(np.shape(config["parameters"][key]["val"]))
            # print(all_params)
    batch_indices.sort()
    losses = np.zeros(batch_indices.flatten()[-1] + 1, dtype=np.float64)
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))

    fits = {}
    sqdevs = {}
    fits["ion"] = np.zeros(all_data["i_data"].shape)
    sqdevs["ion"] = np.zeros(all_data["i_data"].shape)
    fits["ele"] = np.zeros(all_data["e_data"].shape)
    sqdevs["ele"] = np.zeros(all_data["e_data"].shape)

    num_params = 0
    for key, vec in all_params.items():
        num_params += np.shape(vec)[1]

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

        for k in all_params.keys():
            # all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])
            all_params[k] = params[k].reshape(-1)

        if calc_sigma:
            these_params = ts_fitter.get_active_params(fitted_weights, batch)
            hess = ts_fitter.h_loss_wrt_params(these_params, batch)
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
            these_params = ts_fitter.weights_to_params(fitted_weights[i_batch], return_static_params=False)
            if calc_sigma:
                hess = ts_fitter.h_loss_wrt_params(these_params, batch)

            losses[inds] = loss
            sqdevs["ele"][inds] = sqds["ele"]
            sqdevs["ion"][inds] = sqds["ion"]
            if calc_sigma:
                sigmas[inds] = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
                print(f"Number of 0s in sigma: {len(np.where(sigmas==0)[0])}")

            fits["ele"][inds] = ThryE
            fits["ion"][inds] = ThryI

            for k in all_params.keys():
                if np.size(np.shape(params[k])) == 3:
                    all_params[k][inds] = np.squeeze(params[k][inds])
                else:
                    all_params[k][inds] = params[k][inds]

    return losses, sqdevs, used_points, fits, sigmas, all_params


def get_sigmas(keys, hess, batch_size):
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
        config["optimizer"]["batch_size"] = 1
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

            ts_fitter_refit = TSFitter(config, sa, batch)
            new_weights = np.zeros(ts_fitter_refit["init_weights"].shape)

            for ii, key in enumerate(fitted_weights[i // true_batch_size].keys()):
                new_weights[ii] = fitted_weights[(i - 1) // true_batch_size][key][(i - 1) % true_batch_size][0]

            ts_fitter_refit["init_weights"] = new_weights

            res = spopt.minimize(
                ts_fitter_refit.vg_loss if config["optimizer"]["grad_method"] == "AD" else ts_fitter_refit.loss,
                ts_fitter_refit.flattened_weights,
                args=batch,
                method=config["optimizer"]["method"],
                jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                # hess=hess_fn if config["optimizer"]["hessian"] else None,
                bounds=ts_fitter_refit["bounds"],
                options={"disp": True},
            )
            cur_result = ts_fitter_refit["unravel_pytree"](res["x"])

            for key in fitted_weights[i // true_batch_size].keys():
                cur_value = cur_result[key][0, 0]
                new_vals = fitted_weights[i // true_batch_size][key]
                new_vals = new_vals.at[tuple([i % true_batch_size, 0])].set(cur_value)
                fitted_weights[i // true_batch_size][key] = new_vals

        config["optimizer"]["batch_size"] = true_batch_size

    mlflow.log_metrics({"refitting time": round(time.time() - t1, 2)})

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "plots"), exist_ok=True)
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            best_weights_val = {}
            best_weights_std = {}
            for k in fitted_weights[0].keys():
                best_weights_val[k] = np.average([val[k] for val in fitted_weights], axis=0)
                best_weights_std[k] = np.std([val[k] for val in fitted_weights], axis=0)
            losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
                config, batch_indices, all_data, ts_fitter, config["other"]["calc_sigmas"], fitted_weights
            )
            mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
            mlflow.set_tag("status", "plotting")
            t1 = time.time()
            final_params = plotters.plot_angular(
                config,
                losses,
                all_params,
                used_points,
                all_axes,
                fits,
                all_data,
                sqdevs,
                sigmas,
                td,
                best_weights_val,
                best_weights_std,
            )
        else:
            losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
                config, batch_indices, all_data, ts_fitter, config["other"]["calc_sigmas"], fitted_weights
            )
            mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
            mlflow.set_tag("status", "plotting")
            t1 = time.time()
            final_params = plotters.plot_regular(
                config, losses, all_params, used_points, all_axes, fits, all_data, sqdevs, sigmas, td
            )
        mlflow.log_artifacts(td)
    mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})

    mlflow.set_tag("status", "done plotting")

    return final_params
