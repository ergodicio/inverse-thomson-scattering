import matplotlib as mpl
import mlflow, tempfile, os, pandas
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from inverse_thomson_scattering.misc.lineout_plot import lineout_plot


def get_final_params(config, best_weights, all_axes, td):
    all_params = {}
    dist = {}
    for k, v in best_weights.items():
        if k == "fe":
            dist[k] = pandas.Series(v[0])
            dist["v"] = pandas.Series(config["velocity"])
        else:
            if np.shape(v)[1] > 1:
                for i in range(np.shape(v)[1]):
                    all_params[k + str(i)] = pandas.Series(v[:, i].reshape(-1))
            else:
                all_params[k] = pandas.Series(v.reshape(-1))

    final_params = pandas.DataFrame(all_params)
    if config["other"]["extraoptions"]["load_ion_spec"]:
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelI"])
    else:
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelE"])
    final_params.to_csv(os.path.join(td, "csv", "learned_parameters.csv"))

    final_dist = pandas.DataFrame(dist)
    final_dist.to_csv(os.path.join(td, "csv", "learned_dist.csv"))

    return all_params | dist


def plot_final_params(config, all_params, sigmas_ds, td):
    for param in all_params.keys():
        vals = pandas.Series(all_params[param], dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        lineouts = np.array(config["data"]["lineouts"]["val"])
        std = vals.rolling(config["plotting"]["rolling_std_width"], min_periods=1, center=True).std()

        ax.plot(lineouts, vals)
        ax.fill_between(
            lineouts,
            (vals.values - config["plotting"]["n_sigmas"] * sigmas_ds[param].values),
            (vals.values + config["plotting"]["n_sigmas"] * sigmas_ds[param].values),
            color="b",
            alpha=0.1,
        )
        ax.fill_between(
            lineouts,
            (vals.values - config["plotting"]["n_sigmas"] * std.values),
            (vals.values + config["plotting"]["n_sigmas"] * std.values),
            color="r",
            alpha=0.1,
        )
        ax.set_xlabel("lineout", fontsize=14)
        ax.grid()
        ax.set_ylim(0.8 * np.min(vals), 1.2 * np.max(vals))
        ax.set_ylabel(param, fontsize=14)
        fig.savefig(os.path.join(td, "plots", "learned_" + param + ".png"), bbox_inches="tight")
    return


def plot_loss_hist(config, losses, all_params, used_points, td):
    losses[losses > 1e10] = 1e10
    red_losses = losses / (1.1 * (used_points - len(all_params)))
    mlflow.log_metrics(
        {"number of fits above threshold after refit": int(np.sum(red_losses > config["other"]["refit_thresh"]))}
    )

    # make histogram
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)
    if "red_losses_init" not in locals():
        red_losses_init = red_losses
        losses_init = losses
    ax[0].hist([red_losses_init, red_losses], 40)
    # ax[0].hist(red_losses, 128)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\chi^2/DOF$")
    ax[0].set_ylabel("Counts")
    ax[0].set_title("Normalized $L^2$ Norm of the Error")
    ax[0].grid()
    ax[1].hist([losses_init, losses], 40)
    # ax[1].hist(losses, 128)
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\chi^2$")
    ax[1].set_ylabel("Counts")
    ax[1].set_title("$L^2$ Norm of the Error")
    ax[1].grid()
    fig.savefig(os.path.join(td, "plots", "error_hist.png"), bbox_inches="tight")

    losses_ds = pandas.DataFrame(
        {
            "initial_losses": losses_init,
            "losses": losses,
            "initial_reduced_losses": red_losses_init,
            "reduced_losses": red_losses,
        }
    )
    losses_ds.to_csv(os.path.join(td, "csv", "losses.csv"))

    return red_losses


def plot_dist(config, final_params, sigma_fe, td):
    # Create fe image

    # lineouts = np.array(config["data"]["lineouts"]["val"])

    if config["parameters"]["fe"]["dim"] == 1:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(final_params["v"], final_params["fe"])
        ax[1].plot(np.log10(np.exp(final_params["fe"])))
        ax[2].plot(np.exp(final_params["fe"]))

        if config["other"]["calc_sigmas"]:
            ax[0].fill_between(
                final_params["v"],
                (final_params["fe"] - config["plotting"]["n_sigmas"] * sigma_fe.data),
                (final_params["fe"] + config["plotting"]["n_sigmas"] * sigma_fe.data),
                color="b",
                alpha=0.1,
            )
        ax[0].set_xlabel("v/vth (points)", fontsize=14)
        ax[0].set_ylabel("f_e (ln)")
        ax[0].grid()
        # ax.set_ylim(0.8 * np.min(final_params["ne"]), 1.2 * np.max(final_params["ne"]))
        ax[0].set_title("$f_e$", fontsize=14)
        ax[1].set_xlabel("v/vth (points)", fontsize=14)
        ax[1].set_ylabel("f_e (log)")
        ax[1].grid()
        ax[1].set_ylim(-5, 0)
        ax[1].set_title("$f_e$", fontsize=14)
        ax[2].set_xlabel("v/vth (points)", fontsize=14)
        ax[2].set_ylabel("f_e")
        ax[2].grid()
    else:
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        ax.plot_surface(
            final_params["v"][0],
            final_params["v"][1],
            final_params["fe"],
            edgecolor="royalblue",
            lw=0.5,
            rstride=16,
            cstride=16,
            alpha=0.3,
        )
        ax.set_zlim(-50, 0)
        ax.contour(
            final_params["v"][0], final_params["v"][1], final_params["fe"], zdir="x", offset=-7.5, cmap="coolwarm"
        )
        ax.contour(
            final_params["v"][0], final_params["v"][1], final_params["fe"], zdir="y", offset=7.5, cmap="coolwarm"
        )
        ax.contour(
            final_params["v"][0], final_params["v"][1], final_params["fe"], zdir="z", offset=-50, cmap="coolwarm"
        )
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e (ln)")
        ax = fig.add_subplot(1, 3, 2, projection="3d")
        ax.plot_surface(
            final_params["v"][0],
            final_params["v"][1],
            np.log10(np.exp(final_params["fe"])),
            edgecolor="royalblue",
            lw=0.5,
            rstride=16,
            cstride=16,
            alpha=0.3,
        )
        ax.set_zlim(-50, 0)
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.log10(np.exp(final_params["fe"])),
            zdir="x",
            offset=-7.5,
            cmap="coolwarm",
        )
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.log10(np.exp(final_params["fe"])),
            zdir="y",
            offset=7.5,
            cmap="coolwarm",
        )
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.log10(np.exp(final_params["fe"])),
            zdir="z",
            offset=-22,
            cmap="coolwarm",
        )
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e (log)")

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        ax.plot_surface(
            final_params["v"][0],
            final_params["v"][1],
            np.exp(final_params["fe"]),
            edgecolor="royalblue",
            lw=0.5,
            rstride=16,
            cstride=16,
            alpha=0.3,
        )
        ax.set_zlim(-50, 0)
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.exp(final_params["fe"]),
            zdir="x",
            offset=-7.5,
            cmap="coolwarm",
        )
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.exp(final_params["fe"]),
            zdir="y",
            offset=7.5,
            cmap="coolwarm",
        )
        ax.contour(
            final_params["v"][0],
            final_params["v"][1],
            np.exp(final_params["fe"]),
            zdir="z",
            offset=0.0,
            cmap="coolwarm",
        )
        ax.set_xlabel("vx/vth", fontsize=14)
        ax.set_ylabel("vy/vth", fontsize=14)
        ax.set_zlabel("f_e")

    # no rolling sigma bc we use a smoothing kernel
    fig.savefig(os.path.join(td, "plots", "fe_final.png"), bbox_inches="tight")
    return


def save_sigmas_fe(all_params, best_weights_std, sigmas, td):
    sigma_params = {}
    sizes = {key: all_params[key].shape[0] for key in all_params.keys()}
    param_ctr = 0
    for i, k in enumerate(all_params.keys()):
        val = sigmas[0, param_ctr : param_ctr + sizes[k]]
        if k == "fe":
            sigma_fe = xr.DataArray(val, coords=(("v", np.linspace(-7, 7, len(val))),))
        else:
            sigma_params[k] = xr.DataArray(val, coords=(("ind", [0]),))
        param_ctr += sizes[k]

    sigma_params = best_weights_std
    sigma_fe.to_netcdf(os.path.join(td, "binary", "sigma-fe.nc"))
    sigma_params = xr.Dataset(sigma_params)
    sigma_params.to_netcdf(os.path.join(td, "binary", "sigma-params.nc"))

    return sigma_fe


def save_sigmas_params(config, all_params, sigmas, all_axes, td):
    coords = ((all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]])),)
    sigmas_ds = xr.Dataset({k: xr.DataArray(sigmas[:, i], coords=coords) for i, k in enumerate(all_params.keys())})
    sigmas_ds.to_netcdf(os.path.join(td, "sigmas.nc"))
    return sigmas_ds


def plot_data_angular(config, fits, all_data, all_axes, td):
    dat = {
        "fit": fits["ele"],
        "data": all_data["e_data"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :],
    }
    savedata = xr.Dataset({k: xr.DataArray(v) for k, v in dat.items()})
    savedata.to_netcdf(os.path.join(td, "binary", "fit_and_data.nc"))
    savedata["data"] = savedata["data"].T
    savedata["fit"] = savedata["fit"].T

    angs, wavs = np.meshgrid(
        all_axes["epw_x"][config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"]],
        all_axes["epw_y"],
    )

    plot_2D_data_vs_fit(config, angs, wavs, savedata["data"], savedata["fit"], td, xlabel="Angle (degrees)")
    # Create fit and data image
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    # clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 21)
    # ax[0].pcolormesh(
    #     angs,
    #     wavs,
    #     savedata["fit"],
    #     shading="nearest",
    #     cmap="gist_ncar",
    #     vmin=min(np.amin(savedata["data"]), 0),
    #     vmax=max(np.amax(savedata["data"]), 1),
    # )
    # ax[0].set_xlabel("Angle (degrees)")
    # ax[0].set_ylabel("Wavelength (nm)")
    # ax[1].pcolormesh(
    #     angs,
    #     wavs,
    #     savedata["data"],
    #     shading="nearest",
    #     cmap="gist_ncar",
    #     vmin=min(np.amin(savedata["data"]), 0),
    #     vmax=max(np.amax(savedata["data"]), 1),
    # )
    # ax[1].set_xlabel("Angle (degrees)")
    # ax[1].set_ylabel("Wavelength (nm)")
    # fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

    return savedata


def plot_ts_data(config, fits, all_data, all_axes, td):
    if config["other"]["extraoptions"]["load_ion_spec"]:
        coords = (all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]])), (
            "Wavelength",
            all_axes["iaw_y"],
        )
        ion_dat = {"fit": fits["ion"], "data": all_data["i_data"]}
        # fit vs data storage and plot
        ion_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ion_dat.items()})
        ion_savedata.to_netcdf(os.path.join(td, "binary", "ion_fit_and_data.nc"))

        ion_savedata["data"] = ion_savedata["data"].T
        ion_savedata["fit"] = ion_savedata["fit"].T

        x, y = np.meshgrid(
            all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]],
            all_axes["iaw_y"],
        )

        plot_2D_data_vs_fit(config, x, y, ion_savedata["data"], ion_savedata["fit"], td, xlabel=all_axes["x_label"])

    if config["other"]["extraoptions"]["load_ele_spec"]:
        coords = (all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]])), (
            "Wavelength",
            all_axes["epw_y"],
        )
        ele_dat = {"fit": fits["ele"], "data": all_data["e_data"]}
        # fit vs data storage and plot
        ele_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ele_dat.items()})
        ele_savedata.to_netcdf(os.path.join(td, "binary", "ele_fit_and_data.nc"))

        ele_savedata["data"] = ele_savedata["data"].T
        ele_savedata["fit"] = ele_savedata["fit"].T

        x, y = np.meshgrid(
            all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]],
            all_axes["epw_y"],
        )

        plot_2D_data_vs_fit(config, x, y, ele_savedata["data"], ele_savedata["fit"], td, xlabel=all_axes["x_label"])


def plot_2D_data_vs_fit(
    config, x, y, data, fit, td, xlabel="Time (ps)", ylabel="Wavelength (nm)", name="fit_and_data.png"
):
    # Create fit and data image
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    pc = ax[0].pcolormesh(
        x,
        y,
        fit,
        shading="nearest",
        cmap="gist_ncar",
        vmin=np.amin(data) if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
        vmax=np.amax(data) if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"],
    )
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].pcolormesh(
        x,
        y,
        data,
        shading="nearest",
        cmap="gist_ncar",
        vmin=np.amin(data) if config["plotting"]["data_cbar_l"] == "data" else config["plotting"]["data_cbar_l"],
        vmax=np.amax(data) if config["plotting"]["data_cbar_u"] == "data" else config["plotting"]["data_cbar_u"],
    )
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    fig.colorbar(pc)
    fig.savefig(os.path.join(td, "plots", name), bbox_inches="tight")


def plot_ang_lineouts(used_points, sqdevs, losses, all_params, all_axes, savedata, td):
    used_points = used_points * sqdevs["ele"].shape[1]
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
        ax[1].plot(all_axes["epw_y"], np.squeeze(sqdevs["ele"][i, :]), label="Residual")
        ax[1].set_ylabel(r"$\chi^2_i$")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].grid()
        fig.savefig(os.path.join(td, "lineouts", filename), bbox_inches="tight")
        plt.close(fig)
    return


def model_v_actual(config, all_data, all_axes, fits, losses, red_losses, sqdevs, td):
    num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

    os.makedirs(os.path.join(td, "worst"))
    os.makedirs(os.path.join(td, "best"))

    loss_inds = losses.flatten().argsort()[::-1]
    sorted_losses = losses[loss_inds]
    sorted_red_losses = red_losses[loss_inds]
    s_ind = []
    e_ind = []
    sorted_data = []
    sorted_fits = []
    sorted_sqdev = []
    yaxis = []

    if config["other"]["extraoptions"]["load_ele_spec"]:
        s_ind.append(np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_start"])))
        e_ind.append(np.argmin(np.abs(all_axes["epw_y"] - config["plotting"]["ele_window_end"])))
        sorted_fits.append(fits["ele"][loss_inds])
        sorted_data.append(all_data["e_data"][loss_inds])
        sorted_sqdev.append(sqdevs["ele"][loss_inds])
        yaxis.append(all_axes["epw_y"])

    if config["other"]["extraoptions"]["load_ion_spec"]:
        s_ind.append(np.argmin(np.abs(all_axes["iaw_y"] - config["plotting"]["ion_window_start"])))
        e_ind.append(np.argmin(np.abs(all_axes["iaw_y"] - config["plotting"]["ion_window_end"])))
        sorted_fits.append(fits["ion"][loss_inds])
        sorted_data.append(all_data["i_data"][loss_inds])
        sorted_sqdev.append(sqdevs["ion"][loss_inds])
        yaxis.append(all_axes["iaw_y"])

    for i in range(num_plots):
        # plot model vs actual
        titlestr = (
            r"|Error|$^2$" + f" = {sorted_losses[i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[i]]}"
        )
        filename = f"loss={sorted_losses[i]:.2e}-reduced_loss={sorted_red_losses[i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[i]]}.png"

        lineout_plot(
            np.array(sorted_data)[:, i, :],
            np.array(sorted_fits)[:, i, :],
            np.array(sorted_sqdev)[:, i, :],
            yaxis,
            s_ind,
            e_ind,
            titlestr,
            filename,
            td,
            "worst",
        )

        titlestr = (
            r"|Error|$^2$"
            + f" = {sorted_losses[-1 - i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[-1 - i]]}"
        )
        filename = f"loss={sorted_losses[-1 - i]:.2e}-reduced_loss={sorted_red_losses[-1 - i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[-1 - i]]}.png"

        lineout_plot(
            np.array(sorted_data)[:, -1 - i, :],
            np.array(sorted_fits)[:, -1 - i, :],
            np.array(sorted_sqdev)[:, -1 - i, :],
            yaxis,
            s_ind,
            e_ind,
            titlestr,
            filename,
            td,
            "best",
        )


def TScmap():
    # Define jet colormap with 0=white (this might be moved and just loaded here)
    upper = mpl.cm.jet(np.arange(256))
    lower = np.ones((int(256 / 16), 4))
    # - modify the first three columns (RGB):
    #   range linearly between white (1,1,1) and the first color of the upper colormap
    for i in range(3):
        lower[:, i] = np.linspace(1, upper[0, i], lower.shape[0])

    # combine parts of colormap
    cmap = np.vstack((lower, upper))

    # convert to matplotlib colormap
    cmap = mpl.colors.ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])

    return cmap
