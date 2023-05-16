from typing import Dict

import time, os, tempfile, pandas, mlflow

import numpy as np
import xarray as xr
import scipy.optimize as spopt

from matplotlib import pyplot as plt

from inverse_thomson_scattering.misc import plotters
from inverse_thomson_scattering.loss_function import get_loss_function


def recalculate_with_chosen_weights(
    config, batch_indices, all_data, best_weights, func_dict, calc_sigma, raw_weights=None
):
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

    all_params = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            all_params[key] = np.empty(0)
    batch_indices.sort()
    losses = np.zeros(batch_indices.flatten()[-1] + 1, dtype=np.float64)
    batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))

    fits = {}
    sqdevs = {}
    fits["ion"] = np.zeros(all_data["i_data"].shape)
    sqdevs["ion"] = np.zeros(all_data["i_data"].shape)
    fits["ele"] = np.zeros(all_data["e_data"].shape)
    sqdevs["ele"] = np.zeros(all_data["e_data"].shape)
    
    if config["other"]["extraoptions"]["load_ion_spec"]:
        sigmas = np.zeros((all_data["i_data"].shape[0], len(all_params.keys())))
    
    if config["other"]["extraoptions"]["load_ele_spec"]:
        sigmas = np.zeros((all_data["e_data"].shape[0], len(all_params.keys())))

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
        these_weights = best_weights
        # loss, [ThryE, _, params] = func_dict["array_loss_fn"](these_weights, batch)
        losses, sqdevs["ele"], used_points, [ThryE, _, params] = func_dict["array_loss_fn"](raw_weights, batch)
        ThryE, ThryI, lamAxisE, lamAxisI, params = func_dict["calculate_spectra"](raw_weights, batch)
        # these_params = func_dict["get_active_params"](these_weights, batch)
        # hess = func_dict["h_func"](these_params, batch)
        # losses = np.mean(loss, axis=1)

        # sigmas[inds] = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
        fits["ele"] = ThryE

        for k in all_params.keys():
            all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])

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

            if config["optimizer"]["method"] == "adam":
                these_weights = best_weights
            else:
                these_weights = best_weights[i_batch]

            loss, sqds, used_points, [ThryE, ThryI, params] = func_dict["array_loss_fn"](these_weights, batch)
            these_params = func_dict["get_active_params"](these_weights, batch)
            if calc_sigma:
                hess = func_dict["h_func"](these_params, batch)
            # print(hess)

            losses[inds] = loss
            sqdevs["ele"][inds] = sqds["ele"]
            sqdevs["ion"][inds] = sqds["ion"]
            if calc_sigma:
                sigmas[inds] = get_sigmas(all_params.keys(), hess, config["optimizer"]["batch_size"])
            fits["ele"][inds] = ThryE
            fits["ion"][inds] = ThryI

            for k in all_params.keys():
                all_params[k] = np.concatenate([all_params[k], params[k].reshape(-1)])

    return losses, sqdevs, used_points, fits, sigmas, all_params


def get_sigmas(keys, hess, batch_size):
    sigmas = np.zeros((batch_size, len(keys)))

    for i in range(batch_size):
        temp = np.zeros((len(keys), len(keys)))
        for k1, param in enumerate(keys):
            for k2, param2 in enumerate(keys):
                temp[k1, k2] = np.squeeze(hess[param][param2])[i, i]

        # print(temp)
        inv = np.linalg.inv(temp)
        # print(inv)

        for k1, param in enumerate(keys):
            sigmas[i, k1] = np.sign(inv[k1, k1]) * np.sqrt(np.abs(inv[k1, k1]))
            # print(sigmas[i, k1])

    return sigmas


def postprocess(config, batch_indices, all_data: Dict, all_axes: Dict, best_weights, func_dict, sa, raw_weights=None):
    t1 = time.time()

    if config["other"]["extraoptions"]["spectype"] != "angular_full" and config["other"]["refit"]:
        losses_init, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
            config, batch_indices, all_data, best_weights, func_dict, calc_sigma=False, raw_weights=raw_weights
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

            func_dict_refit = get_loss_function(config, sa, batch)
            new_weights = np.zeros(func_dict_refit["init_weights"].shape)

            for ii, key in enumerate(best_weights[i // true_batch_size]["ts_parameter_generator"].keys()):
                new_weights[ii] = best_weights[(i - 1) // true_batch_size]["ts_parameter_generator"][key][
                    (i - 1) % true_batch_size
                ][0]

            func_dict_refit["init_weights"] = new_weights

            res = spopt.minimize(
                func_dict_refit["vg_func"] if config["optimizer"]["grad_method"] == "AD" else func_dict_refit["v_func"],
                func_dict_refit["init_weights"],
                args=batch,
                method=config["optimizer"]["method"],
                jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                # hess=hess_fn if config["optimizer"]["hessian"] else None,
                bounds=func_dict_refit["bounds"],
                options={"disp": True},
            )
            cur_result = func_dict_refit["unravel_pytree"](res["x"])

            for key in best_weights[i // true_batch_size]["ts_parameter_generator"].keys():
                cur_value = cur_result["ts_parameter_generator"][key][0, 0]
                new_vals = best_weights[i // true_batch_size]["ts_parameter_generator"][key]
                new_vals = new_vals.at[tuple([i % true_batch_size, 0])].set(cur_value)
                best_weights[i // true_batch_size]["ts_parameter_generator"][key] = new_vals

        config["optimizer"]["batch_size"] = true_batch_size

    
    mlflow.log_metrics({"refitting time": round(time.time() - t1, 2)})
    losses, sqdevs, used_points, fits, sigmas, all_params = recalculate_with_chosen_weights(
        config, batch_indices, all_data, best_weights, func_dict, calc_sigma=True, raw_weights=raw_weights
    )

    
    mlflow.log_metrics({"postprocessing time": round(time.time() - t1, 2)})
    mlflow.set_tag("status", "plotting")
    t1 = time.time()

    if config["other"]["extraoptions"]["spectype"] == "angular_full":
        with tempfile.TemporaryDirectory() as td:
            # print(all_params)
            for key in all_params.keys():
                all_params[key] = pandas.Series(all_params[key])
            final_params = pandas.DataFrame(all_params)
            final_params.to_csv(os.path.join(td, "learned_parameters.csv"))

            # coords = ("lineout", np.array(config["data"]["lineouts"]["val"])), ("Wavelength", all_axes["epw_y"]) #replace with true wavelength axis
            dat = {
                "fit": fits,
                "data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
            }
            savedata = xr.Dataset({k: xr.DataArray(v) for k, v in dat.items()})
            savedata.to_netcdf(os.path.join(td, "fit_and_data.nc"))
            savedata["data"] = savedata["data"].T
            savedata["fit"] = savedata["fit"].T
            
            angs, wavs = np.meshgrid(all_axes["epw_x"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"]], all_axes["epw_y"])
            
            # Create fit and data image
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 21)
            ax[0].pcolormesh(angs, wavs,
                savedata["fit"],
                shading = 'nearest',
                cmap="gist_ncar",
                vmin=np.amin(savedata["data"]),
                vmax=np.amax(savedata["data"]),
            )
            ax[0].set_xlabel("Angle (degrees)")
            ax[0].set_ylabel("Wavelength (nm)")
            ax[1].pcolormesh(angs, wavs,
                savedata["data"],
                shading = 'nearest',
                cmap="gist_ncar",
                vmin=np.amin(savedata["data"]),
                vmax=np.amax(savedata["data"]),
            )
            ax[1].set_xlabel("Angle (degrees)")
            ax[1].set_ylabel("Wavelength (nm)")
            fig.savefig(os.path.join(td, "fit_and_data.png"), bbox_inches="tight")
            
            used_points = used_points * sqdevs.shape[1]
            red_losses = np.sum(losses) / (1.1 * (used_points - len(all_params)))
            mlflow.log_metrics({"Total reduced loss": float(red_losses)})

            # Create lineout images
            os.makedirs(os.path.join(td, "lineouts"))
            for i in np.linspace(0, savedata["data"].shape[1] - 1, 8, dtype="int"):
                # plot model vs actual
                titlestr = r"|Error|$^2$" + f" = {losses[i]:.2e}, line out # {i}"
                filename = f"loss={losses[i]:.2e}-lineout={i}.png"
                fig, ax = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True, sharex=True)
                ax[0].plot(all_axes["epw_y"], np.squeeze(savedata["data"][:, i]), label="Data")
                ax[0].plot(all_axes["epw_y"], np.squeeze(savedata["fit"][:, i]), label="Fit")
                ax[0].set_title(titlestr, fontsize=14)
                ax[0].set_ylabel("Amplitude (arb. units)")
                ax[0].legend(fontsize=14)
                ax[0].grid()
                ax[1].plot(all_axes["epw_y"], np.squeeze(sqdevs[i, :]), label="Residual")
                ax[1].set_ylabel("$\chi^2_i$")
                ax[1].set_xlabel("Wavelength (nm)")
                ax[1].grid()
                fig.savefig(os.path.join(td, "lineouts", filename), bbox_inches="tight")
                plt.close(fig)

            # Create fe image
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # lineouts = np.array(config["data"]["lineouts"]["val"])
            ax[0].plot(final_params["fe"])
            # ax.fill_between(
            # lineouts,
            # (final_params["ne"] - 3 * sigmas_ds.ne),
            # (final_params["ne"] + 3 * sigmas_ds.ne),
            # color="b",
            # alpha=0.1,
            # )
            ax[0].set_xlabel("v/vth (points)", fontsize=14)
            ax[0].set_ylabel("f_e (ln)")
            ax[0].grid()
            # ax.set_ylim(0.8 * np.min(final_params["ne"]), 1.2 * np.max(final_params["ne"]))
            ax[0].set_title("$f_e$", fontsize=14)
            ax[1].plot(np.log10(np.exp(final_params["fe"])))
            ax[1].set_xlabel("v/vth (points)", fontsize=14)
            ax[1].set_ylabel("f_e (log)")
            ax[1].grid()
            ax[1].set_ylim(-5, 0)
            ax[1].set_title("$f_e$", fontsize=14)
            ax[2].plot(np.exp(final_params["fe"]))
            ax[2].set_xlabel("v/vth (points)", fontsize=14)
            ax[2].set_ylabel("f_e")
            ax[2].grid()
            fig.savefig(os.path.join(td, "fe_final.png"), bbox_inches="tight")

            mlflow.log_artifacts(td)
            mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})
    else:
        num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2
        with tempfile.TemporaryDirectory() as td:
            # store fitted parameters
            final_params = pandas.DataFrame(all_params)
            final_params.to_csv(os.path.join(td, "learned_parameters.csv"))

            losses[losses>1e10]=1e10
            red_losses = losses / (1.1 * (used_points - len(all_params)))
            loss_inds = losses.flatten().argsort()[::-1]
            sorted_losses = losses[loss_inds]
            sorted_redchi = red_losses[loss_inds]
            mlflow.log_metrics({"number of fits above threshold after refit": int(np.sum(red_losses > config["other"]["refit_thresh"]))})
            
            #this wont work for ion+electron fitting (just electrons will be plotted)
            if config["other"]["extraoptions"]["load_ion_spec"]:
                coords = ("lineout", np.array(config["data"]["lineouts"]["val"])), (
                    "Wavelength",
                    all_axes["iaw_y"],
                )
                dat = {"fit": fits["ion"], "data": all_data["i_data"]}
                sorted_fits = fits["ion"][loss_inds]
                sorted_data = all_data["i_data"][loss_inds]
                sorted_sqdev = sqdevs["ion"][loss_inds]
                y_axis = all_axes["iaw_y"]
            if config["other"]["extraoptions"]["load_ele_spec"]:
                coords = ("lineout", np.array(config["data"]["lineouts"]["val"])), (
                    "Wavelength",
                    all_axes["epw_y"],
                ) 
                dat = {"fit": fits["ele"], "data": all_data["e_data"]}
                sorted_fits = fits["ele"][loss_inds]
                sorted_data = all_data["e_data"][loss_inds]
                sorted_sqdev = sqdevs["ele"][loss_inds]
                y_axis = all_axes["epw_y"]
                
            # fit vs data storage and plot
            savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in dat.items()})
            savedata.to_netcdf(os.path.join(td, "fit_and_data.nc"))

            fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 11)
            #clevs = np.linspace(0, 300, 11)
            savedata["fit"].T.plot(ax=ax[0], cmap="gist_ncar", levels=clevs)
            savedata["data"].T.plot(ax=ax[1], cmap="gist_ncar", levels=clevs)
            fig.savefig(os.path.join(td, "fit_and_data.png"), bbox_inches="tight")
            

            fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
            if "red_losses_init" not in locals():
                red_losses_init = red_losses
                losses_init = losses
            ax[0].hist([red_losses_init, red_losses], 40)
            # ax[0].hist(red_losses, 128)
            ax[0].set_yscale("log")
            ax[0].set_xlabel("$\chi^2/DOF$")
            ax[0].set_ylabel("Counts")
            ax[0].set_title("Normalized $L^2$ Norm of the Error")
            ax[0].grid()
            ax[1].hist([losses_init, losses], 40)
            # ax[1].hist(losses, 128)
            ax[1].set_yscale("log")
            ax[1].set_xlabel("$\chi^2$")
            ax[1].set_ylabel("Counts")
            ax[1].set_title("$L^2$ Norm of the Error")
            ax[1].grid()
            fig.savefig(os.path.join(td, "error_hist.png"), bbox_inches="tight")

            losses_ds = pandas.DataFrame({"initial_losses": losses_init, "losses": losses, "initial_reduced_losses": red_losses_init, "reduced_losses": red_losses})
            losses_ds.to_csv(os.path.join(td, "losses.csv"))
            
            os.makedirs(os.path.join(td, "worst"))
            os.makedirs(os.path.join(td, "best"))

            plotters.model_v_actual(
                sorted_losses,
                sorted_data,
                sorted_fits,
                num_plots,
                td,
                config,
                loss_inds,
                y_axis,
                sorted_sqdev,
                sorted_redchi,
            )

            sigmas_ds = xr.Dataset(
                {k: xr.DataArray(sigmas[:, i], coords=(coords[0],)) for i, k in enumerate(all_params.keys())}
            )
            sigmas_ds.to_netcdf(os.path.join(td, "sigmas.nc"))

            # ne_vals = pandas.Series(final_params["ne"])
            # ne_std = ne_vals.rolling(5, min_periods=1, center=True).std()
            # Te_vals = pandas.Series(final_params["Te"])
            # Te_std = Te_vals.rolling(5, min_periods=1, center=True).std()
            # m_vals = pandas.Series(final_params["m"])
            # m_std = m_vals.rolling(5, min_periods=1, center=True).std()

            # print(final_params)
            # print(sigmas_ds)
            # ne, Te, m plots with errorbars
#             fig, ax = plt.subplots(1, 3, figsize=(15, 4))
#             lineouts = np.array(config["data"]["lineouts"]["val"])
#             ax[0].plot(lineouts, final_params["ne"])
#             ax[0].fill_between(
#                 lineouts,
#                 (final_params["ne"] - 3 * sigmas_ds.ne),
#                 (final_params["ne"] + 3 * sigmas_ds.ne),
#                 color="b",
#                 alpha=0.1,
#             )
#             ax[0].fill_between(
#                 lineouts,
#                 (final_params["ne"] - 3 * ne_std.values),
#                 (final_params["ne"] + 3 * ne_std.values),
#                 color="r",
#                 alpha=0.1,
#             )
#             ax[0].set_xlabel("lineout", fontsize=14)
#             ax[0].grid()
#             ax[0].set_ylim(0.8 * np.min(final_params["ne"]), 1.2 * np.max(final_params["ne"]))
#             ax[0].set_title("$n_e$(t)", fontsize=14)

#             ax[1].plot(lineouts, final_params["Te"])
#             ax[1].fill_between(
#                 lineouts,
#                 (final_params["Te"] - 3 * sigmas_ds.Te),
#                 (final_params["Te"] + 3 * sigmas_ds.Te),
#                 color="b",
#                 alpha=0.1,
#             )
#             ax[1].fill_between(
#                 lineouts,
#                 (final_params["Te"] - 3 * Te_std.values),
#                 (final_params["Te"] + 3 * Te_std.values),
#                 color="r",
#                 alpha=0.1,
#             )
#             ax[1].set_xlabel("lineout", fontsize=14)
#             ax[1].grid()
#             ax[1].set_ylim(0.8 * np.min(final_params["Te"]), 1.2 * np.max(final_params["Te"]))
#             ax[1].set_title("$T_e$(t)", fontsize=14)

#             ax[2].plot(lineouts, final_params["m"])
#             ax[2].fill_between(
#                 lineouts,
#                 (final_params["m"] - 3 * sigmas_ds.m),
#                 (final_params["m"] + 3 * sigmas_ds.m),
#                 color="b",
#                 alpha=0.1,
#             )
#             ax[2].fill_between(
#                 lineouts,
#                 (final_params["m"] - 3 * m_std.values),
#                 (final_params["m"] + 3 * m_std.values),
#                 color="r",
#                 alpha=0.1,
#             )
#             ax[2].set_xlabel("lineout", fontsize=14)
#             ax[2].grid()
#             ax[2].set_ylim(0.8 * np.min(final_params["m"]), 1.2 * np.max(final_params["m"]))
#             ax[2].set_title("$m$(t)", fontsize=14)

#             fig.savefig(os.path.join(td, "learned_parameters.png"), bbox_inches="tight")

            mlflow.log_artifacts(td)
            mlflow.log_metrics({"plotting time": round(time.time() - t1, 2)})

    mlflow.set_tag("status", "done plotting")

    return final_params
