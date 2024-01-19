import matplotlib as mpl
import mlflow, tempfile, os, pandas
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.model.physics.generate_spectra import FitModel
from inverse_thomson_scattering.misc.lineout_plot import lineout_plot


def plot_angular(
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
):
    all_params = {}
    dist = {}
    for k, v in best_weights_val.items():
        if k == "fe":
            dist[k] = pandas.Series(v[0])
            dist["v"] = pandas.Series(config["velocity"])
        else:
            all_params[k] = pandas.Series(v.reshape(-1))

    final_params = pandas.DataFrame(all_params)
    final_params.to_csv(os.path.join(td, "csv", "learned_parameters.csv"))

    final_dist = pandas.DataFrame(dist)
    final_dist.to_csv(os.path.join(td, "csv", "learned_dist.csv"))

    sigma_params = {}
    sizes = {key: all_params[key].shape[0] for key in all_params.keys()}
    param_ctr = 0
    if config["other"]["calc_sigmas"]:
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

    # Create fit and data image
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 21)
    ax[0].pcolormesh(
        angs,
        wavs,
        savedata["fit"],
        shading="nearest",
        cmap="gist_ncar",
        vmin=np.amin(savedata["data"]),
        vmax=np.amax(savedata["data"]),
    )
    ax[0].set_xlabel("Angle (degrees)")
    ax[0].set_ylabel("Wavelength (nm)")
    ax[1].pcolormesh(
        angs,
        wavs,
        savedata["data"],
        shading="nearest",
        cmap="gist_ncar",
        vmin=np.amin(savedata["data"]),
        vmax=np.amax(savedata["data"]),
    )
    ax[1].set_xlabel("Angle (degrees)")
    ax[1].set_ylabel("Wavelength (nm)")
    fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

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

    # Create fe image
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # lineouts = np.array(config["data"]["lineouts"]["val"])
    ax[0].plot(xie := np.linspace(-7, 7, config["parameters"]["fe"]["length"]), final_dist["fe"])

    if config["other"]["calc_sigmas"]:
        ax[0].fill_between(
            xie,
            (final_params["fe"] - config["plotting"]["n_sigmas"] * sigma_fe.data),
            (final_params["fe"] + config["plotting"]["n_sigmas"] * sigma_fe.data),
            color="b",
            alpha=0.1,
        )

    # no rolling sigma bc we use a smoothing kernel
    ax[0].set_xlabel("v/vth (points)", fontsize=14)
    ax[0].set_ylabel("f_e (ln)")
    ax[0].grid()
    # ax.set_ylim(0.8 * np.min(final_params["ne"]), 1.2 * np.max(final_params["ne"]))
    ax[0].set_title("$f_e$", fontsize=14)
    ax[1].plot(np.log10(np.exp(final_dist["fe"])))
    ax[1].set_xlabel("v/vth (points)", fontsize=14)
    ax[1].set_ylabel("f_e (log)")
    ax[1].grid()
    ax[1].set_ylim(-5, 0)
    ax[1].set_title("$f_e$", fontsize=14)
    ax[2].plot(np.exp(final_dist["fe"]))
    ax[2].set_xlabel("v/vth (points)", fontsize=14)
    ax[2].set_ylabel("f_e")
    ax[2].grid()
    fig.savefig(os.path.join(td, "plots", "fe_final.png"), bbox_inches="tight")

    return all_params | dist


def plot_regular(config, losses, all_params, used_points, all_axes, fits, all_data, sqdevs, sigmas, td):
    num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

    # store fitted parameters
    reshaped_params = {}
    for key in all_params.keys():
        # cur_ind = 1
        if np.shape(all_params[key])[1] > 1:
            for i in range(np.shape(all_params[key])[1]):
                reshaped_params[key + str(i)] = all_params[key][:, i]
        else:
            reshaped_params[key] = all_params[key][:, 0]
        # all_params[key] = all_params[key].tolist()
    final_params = pandas.DataFrame(reshaped_params)
    if config["other"]["extraoptions"]["load_ion_spec"]:
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelI"])
    else:
        final_params.insert(0, all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]]))
        final_params.insert(0, "lineout pixel", config["data"]["lineouts"]["pixelE"])
    final_params.to_csv(os.path.join(td, "csv", "learned_parameters.csv"))

    losses[losses > 1e10] = 1e10
    red_losses = losses / (1.1 * (used_points - len(all_params)))
    loss_inds = losses.flatten().argsort()[::-1]
    sorted_losses = losses[loss_inds]
    sorted_redchi = red_losses[loss_inds]
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

    os.makedirs(os.path.join(td, "worst"))
    os.makedirs(os.path.join(td, "best"))

    if config["other"]["extraoptions"]["load_ion_spec"]:
        coords = (all_axes["x_label"], np.array(all_axes["iaw_x"][config["data"]["lineouts"]["pixelI"]])), (
            "Wavelength",
            all_axes["iaw_y"],
        )
        ion_dat = {"fit": fits["ion"], "data": all_data["i_data"]}
        ion_sorted_fits = fits["ion"][loss_inds]
        ion_sorted_data = all_data["i_data"][loss_inds]
        ion_sorted_sqdev = sqdevs["ion"][loss_inds]
        ion_y_axis = all_axes["iaw_y"]
        # fit vs data storage and plot
        ion_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ion_dat.items()})
        ion_savedata.to_netcdf(os.path.join(td, "binary", "ion_fit_and_data.nc"))
    if config["other"]["extraoptions"]["load_ele_spec"]:
        coords = (all_axes["x_label"], np.array(all_axes["epw_x"][config["data"]["lineouts"]["pixelE"]])), (
            "Wavelength",
            all_axes["epw_y"],
        )
        ele_dat = {"fit": fits["ele"], "data": all_data["e_data"]}
        ele_sorted_fits = fits["ele"][loss_inds]
        ele_sorted_data = all_data["e_data"][loss_inds]
        ele_sorted_sqdev = sqdevs["ele"][loss_inds]
        ele_y_axis = all_axes["epw_y"]
        # fit vs data storage and plot
        ele_savedata = xr.Dataset({k: xr.DataArray(v, coords=coords) for k, v in ele_dat.items()})
        ele_savedata.to_netcdf(os.path.join(td, "binary", "ele_fit_and_data.nc"))

    if config["other"]["extraoptions"]["load_ion_spec"] and config["other"]["extraoptions"]["load_ele_spec"]:
        fig, ax = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)
        ion_clevs = np.linspace(
            np.amin(ion_dat["data"])
            if config["plotting"]["data_cbar_l"] == "data"
            else config["plotting"]["data_cbar_l"],
            np.amax(ion_dat["data"])
            if config["plotting"]["data_cbar_u"] == "data"
            else config["plotting"]["data_cbar_u"],
            11,
        )
        ele_clevs = np.linspace(
            np.amin(ele_dat["data"])
            if config["plotting"]["data_cbar_l"] == "data"
            else config["plotting"]["data_cbar_l"],
            np.amax(ele_dat["data"])
            if config["plotting"]["data_cbar_u"] == "data"
            else config["plotting"]["data_cbar_u"],
            11,
        )
        # clevs = np.linspace(0, 300, 11)
        ele_savedata["fit"].T.plot(ax=ax[0][0], cmap="gist_ncar", levels=ele_clevs)
        ele_savedata["data"].T.plot(ax=ax[0][1], cmap="gist_ncar", levels=ele_clevs)
        ion_savedata["fit"].T.plot(ax=ax[1][0], cmap="gist_ncar", levels=ion_clevs)
        ion_savedata["data"].T.plot(ax=ax[1][1], cmap="gist_ncar", levels=ion_clevs)
        fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

        model_v_actual(
            sorted_losses,
            [ele_sorted_data, ion_sorted_data],
            [ele_sorted_fits, ion_sorted_fits],
            num_plots,
            td,
            config,
            loss_inds,
            [ele_y_axis, ion_y_axis],
            [ele_sorted_sqdev, ion_sorted_sqdev],
            sorted_redchi,
        )

    elif config["other"]["extraoptions"]["load_ion_spec"]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        clevs = np.linspace(
            np.amin(ion_savedata["data"])
            if config["plotting"]["data_cbar_l"] == "data"
            else config["plotting"]["data_cbar_l"],
            np.amax(ion_savedata["data"])
            if config["plotting"]["data_cbar_u"] == "data"
            else config["plotting"]["data_cbar_u"],
            11,
        )
        # clevs = np.linspace(0, 300, 11)
        ion_savedata["fit"].T.plot(ax=ax[0], cmap="gist_ncar", levels=clevs)
        ion_savedata["data"].T.plot(ax=ax[1], cmap="gist_ncar", levels=clevs)
        fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

        model_v_actual(
            sorted_losses,
            ion_sorted_data,
            ion_sorted_fits,
            num_plots,
            td,
            config,
            loss_inds,
            np.array([ion_y_axis]),
            ion_sorted_sqdev,
            sorted_redchi,
        )

    elif config["other"]["extraoptions"]["load_ele_spec"]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
        clevs = np.linspace(
            np.amin(ele_savedata["data"])
            if config["plotting"]["data_cbar_l"] == "data"
            else config["plotting"]["data_cbar_l"],
            np.amax(ele_savedata["data"])
            if config["plotting"]["data_cbar_u"] == "data"
            else config["plotting"]["data_cbar_u"],
            11,
        )
        # clevs = np.linspace(0, 300, 11)
        ele_savedata["fit"].T.plot(ax=ax[0], cmap="gist_ncar", levels=clevs)
        ele_savedata["data"].T.plot(ax=ax[1], cmap="gist_ncar", levels=clevs)
        fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

        model_v_actual(
            sorted_losses,
            ele_sorted_data,
            ele_sorted_fits,
            num_plots,
            td,
            config,
            loss_inds,
            np.array([ele_y_axis]),
            ele_sorted_sqdev,
            sorted_redchi,
        )

    sigmas_ds = xr.Dataset(
        {k: xr.DataArray(sigmas[:, i], coords=(coords[0],)) for i, k in enumerate(reshaped_params.keys())}
    )
    sigmas_ds.to_netcdf(os.path.join(td, "sigmas.nc"))

    for param in reshaped_params.keys():
        vals = pandas.Series(final_params[param], dtype=float)
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

    return final_params


def plotinput(config, sa):
    parameters = config["parameters"]

    # Setup x0
    xie = np.linspace(-7, 7, parameters["fe"]["length"])

    NumDistFunc = get_num_dist_func(parameters["fe"]["type"], xie)
    parameters["fe"]["val"] = np.log(NumDistFunc(parameters["m"]["val"]))
    parameters["fe"]["lb"] = np.multiply(parameters["fe"]["lb"], np.ones(parameters["fe"]["length"]))
    parameters["fe"]["ub"] = np.multiply(parameters["fe"]["ub"], np.ones(parameters["fe"]["length"]))

    x0 = []
    lb = []
    ub = []
    xiter = []
    for i, _ in enumerate(config["data"]["lineouts"]["val"]):
        for key in parameters.keys():
            if parameters[key]["active"]:
                if np.size(parameters[key]["val"]) > 1:
                    x0.append(parameters[key]["val"][i])
                elif isinstance(parameters[key]["val"], list):
                    x0.append(parameters[key]["val"][0])
                else:
                    x0.append(parameters[key]["val"])
                lb.append(parameters[key]["lb"])
                ub.append(parameters[key]["ub"])

    x0 = np.array(x0)
    fit_model = FitModel(config, xie, sa)

    print("plotting")
    mlflow.set_tag("status", "plotting")

    fig = plt.figure(figsize=(14, 6))
    with tempfile.TemporaryDirectory() as td:
        fig.clf()
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        fig, ax = plotState(
            x0,
            config,
            [1, 1],
            xie,
            sa,
            [],
            fitModel2=fit_model,
            fig=fig,
            ax=[ax, ax2],
        )
        fig.savefig(os.path.join(td, "simulated_spectrum.png"), bbox_inches="tight")
        mlflow.log_artifacts(td, artifact_path="plots")
    return


def model_v_actual(
    sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds, yaxis, sorted_sqdev, sorted_red_losses
):
    if config["other"]["extraoptions"]["load_ion_spec"] and config["other"]["extraoptions"]["load_ele_spec"]:
        ele_s_ind = np.argmin(np.abs(yaxis[0] - config["other"]["ele_window_start"]))
        ele_e_ind = np.argmin(np.abs(yaxis[0] - config["other"]["ele_window_end"]))
        ion_s_ind = np.argmin(np.abs(yaxis[1] - config["other"]["ion_window_start"]))
        ion_e_ind = np.argmin(np.abs(yaxis[1] - config["other"]["ion_window_end"]))
        s_ind = [ele_s_ind, ion_s_ind]
        e_ind = [ele_e_ind, ion_e_ind]
    elif config["other"]["extraoptions"]["load_ion_spec"]:
        s_ind = [np.argmin(np.abs(yaxis - config["other"]["ion_window_start"]))]
        e_ind = [np.argmin(np.abs(yaxis - config["other"]["ion_window_end"]))]
        sorted_data = [sorted_data]
        sorted_fits = [sorted_fits]
        sorted_sqdev = [sorted_sqdev]

    elif config["other"]["extraoptions"]["load_ele_spec"]:
        s_ind = [np.argmin(np.abs(yaxis - config["other"]["ele_window_start"]))]
        e_ind = [np.argmin(np.abs(yaxis - config["other"]["ele_window_end"]))]
        sorted_data = [sorted_data]
        sorted_fits = [sorted_fits]
        sorted_sqdev = [sorted_sqdev]

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


def LinePlots(
    x,
    y,
    fig,
    ax,
    f=[],
    XLabel="v/v_{th}",
    YLabel="Amplitude (arb.)",
    CurveNames=[],
    Residuals=[],
    title="Simulated Thomson Spectrum",
):
    """
    This function plots a set of lines on a specified axis and formats the figure with specified labels. A set of resuduals can also be supplied to be plotted below the primary image
    Args:
        x: x-axis data array
        y: y data can include multiple lines
        fig: figure handle
        ax: axis handle
        f: Distribtuion function to be overplotted on a log scale
        XLabel: x-axis label
        YLabel: y-axis label
        CurveNames: list of strings containing names of curvs being plotted
        Residuals: y data of residuals to be plotted
        title: figure title

    Returns:
    """

    if np.shape(x)[0] == 0:
        x = np.arange(max(np.shape(y)))

    if np.shape(x)[0] != np.shape(y)[0]:
        # This occurs if multiple curves are submitted as part of y
        y = y.transpose()

    if Residuals:
        ax[0].plot(x, y)
        # possibly change to stem
        ax[1].plot(x, Residuals)

    if f:
        ax.plot(x, y, "b")
        ax2 = ax.twinx()
        ax2.plot(x, f, "h")

        ax2.set_yscale("log")

        ax.tick_params(axis="y", color="k", labelcolor="k")
        ax2.tick_params(axis="y", color="g", labelcolor="g")

        ax2.set_ylabel("Amplitude (arb.)", color="g")

    else:
        ax.plot(x, y)

    ax.set_title(title)
    ax.set_ylabel(YLabel)
    ax.set_xlabel(XLabel)

    if CurveNames:
        ax.legend(CurveNames)


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


def ColorPlots(
    x,
    y,
    C,
    Line=[],
    vmin=0,
    vmax=None,
    logplot=False,
    kaxis=[],
    XLabel="r$\lambda_s (nm)$",
    YLabel=r"$\theta (\circ)$",
    CurveNames=[],
    Residuals=[],
    title=r"EPW feature vs $\theta$",
):
    """
    This function plots a 2D color image, with the default behaviour based on ARTS data. Optional argmuents rescale to a log scale, add a residual plot, overplot lines, or add a second y-axis with k-vector information.

    Args:
        x: x-axis data array
        y: y-axis data array
        C: color data
        Line: Curves to be plotted over the color data [[x1],[y1],[x2],[y2],...]
        vmin: color scale minimum
        vmax: color scale maximum
        logplot: boolean determining if the color scale is linear or log
        kaxis: list of density, temperature, and wavelength used to calculate a k-vector axis
        XLabel: x-axis label
        YLabel: y-axis label
        CurveNames: list of strings containing names of curvs being plotted
        Residuals: y data of residuals to be plotted
        title: figure title

    Returns:
    """

    if Residuals:
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[2, 1], figsize=(16, 4))
        ax0 = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax0 = ax

    cmap = TScmap()

    if logplot:
        C = np.log(C)

    im = ax0.imshow(
        C, cmap, interpolation="none", extent=[x[0], x[-1], y[-1], y[0]], aspect="auto", vmin=vmin, vmax=vmax
    )
    ax0.set_title(
        title,
        fontdict={"fontsize": 10, "fontweight": "bold"},
    )
    ax0.set_xlabel(XLabel)
    ax0.set_ylabel(YLabel)
    plt.colorbar(im, ax=ax0)

    if Line:
        line_colors = iter(["r", "k", "g", "b"])
        for i in range(0, len(Line), 2):
            ax0.plot(Line[i], Line[i + 1], next(line_colors))

    if kaxis:
        ax2 = ax0.secondary_yaxis("right", fucntions=(forward_kaxis, backward_kaxis))
        secax_y2.set_ylabel(r"$~v_p/v_{th}$")

        def forward_kaxis(y):
            c = 2.99792458e10
            omgL = 2 * np.pi * 1e7 * c / kaxis[2]
            omgpe = 5.64 * 10**4 * np.sqrt(kaxis[0])
            ko = np.sqrt((omgL**2 - omgpe**2) / c**2)
            newy = np.sqrt(kaxis[0] / (1000 * kaxis[1])) * 1.0 / (1486 * ko * np.sin(y / 360 * np.pi))
            return newy

        def backward_kaxis(y):
            c = 2.99792458e10
            omgL = 2 * np.pi * 1e7 * c / kaxis[2]
            omgpe = 5.64 * 10**4 * np.sqrt(kaxis[0])
            ko = np.sqrt((omgL**2 - omgpe**2) / c**2)
            newy = 360 / np.pi * np.arcsin(np.sqrt(kaxis[0] / (1000 * kaxis[1])) * 1.0 / (1486 * ko * y))
            return newy

    if Residuals:
        # possibly change to stem
        ax[1].plot(x, Residuals)

    if CurveNames:
        ax.legend(CurveNames)

    plt.show()


def plotState(x, config, amps, xie, sas, data, noiseE, nosieI, fitModel2, fig, ax):
    [modlE, modlI, lamAxisE, lamAxisI, tsdict] = fitModel2(x, sas["weights"])

    lam = config["parameters"]["lam"]["val"]
    amp1 = config["parameters"]["amp1"]["val"]
    amp2 = config["parameters"]["amp2"]["val"]
    amp3 = config["parameters"]["amp3"]["val"]

    stddev = config["other"]["PhysParams"]["widIRF"]

    if config["other"]["extraoptions"]["load_ion_spec"]:
        stddevI = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
        originI = (np.amax(lamAxisI) + np.amin(lamAxisI)) / 2.0
        inst_funcI = np.squeeze(
            (1.0 / (stddevI * np.sqrt(2.0 * np.pi))) * np.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
        )  # Gaussian
        ThryI = np.convolve(modlI, inst_funcI, "same")
        ThryI = (np.amax(modlI) / np.amax(ThryI)) * ThryI
        ThryI = np.average(ThryI.reshape(1024, -1), axis=1)

        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisI = np.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = amp3 * amps[1] * ThryI / max(ThryI)

        ThryI = ThryI + np.array(noiseI)

    if config["other"]["extraoptions"]["load_ele_spec"]:
        stddevE = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
        # Conceptual_origin so the convolution doesn't shift the signal
        originE = (np.amax(lamAxisE) + np.amin(lamAxisE)) / 2.0
        inst_funcE = np.squeeze(
            (1.0 / (stddevE * np.sqrt(2.0 * np.pi))) * np.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
        )  # Gaussian
        ThryE = np.convolve(modlE, inst_funcE, "same")
        ThryE = (np.amax(modlE) / np.amax(ThryE)) * ThryE

        if config["other"]["PhysParams"]["norm"] > 0:
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam] / max(ThryE[lamAxisE < lam]))
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam] / max(ThryE[lamAxisE > lam]))

        ThryE = np.average(ThryE.reshape(1024, -1), axis=1)
        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisE = np.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = amps[0] * ThryE / max(ThryE)
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam])
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam])

        ThryE = ThryE + np.array(noiseE)

    if config["other"]["extraoptions"]["spectype"] == 0:
        print("colorplot still needs to be written")
        # Write Colorplot
        # Thryinit=ArtemisModel(config["parameters"],xie,scaterangs,x0,weightMatrix,...
    #    spectralFWHM,angularFWHM,lamAxis,xax,D,norm2B);
    # if ~norm2B
    #    Thryinit=Thryinit./max(Thryinit(470:900,:));
    #    Thryinit=Thryinit.*max(data(470:900,:));
    #    Thryinit=config["parameters"].amp1.Value*Thryinit;
    # end
    # chisq = sum(sum((data([40:330 470:900],90:1015)-Thryinit([40:330 470:900],90:1015)).^2));
    # Thryinit(330:470,:)=0;
    #
    # ColorPlots(yax,xax,rot90(Thryinit),'Kaxis',[config["parameters"].ne.Value*1E20,config["parameters"].Te.Value,526.5],...
    #    'Title','Starting point','Name','Initial Spectrum');
    # ColorPlots(yax,xax,rot90(data-Thryinit),'Title',...
    #    ['Initial difference: \chi^2 =' num2str(chisq)],'Name','Initial Difference');
    # load('diffcmap.mat','diffcmap');
    # colormap(diffcmap);

    # if norm2B
    #    caxis([-1 1]);
    # else
    #    caxis([-8000 8000]);
    # end
    else:
        if config["other"]["extraoptions"]["load_ion_spec"]:
            LinePlots(
                lamAxisI,
                np.vstack((data[1, :], ThryI)),
                fig,
                ax[0],
                CurveNames=["Data", "Fit"],
                XLabel="Wavelength (nm)",
            )
            ax[0].set_xlim([525, 528])

        if config["other"]["extraoptions"]["load_ele_spec"]:
            LinePlots(
                lamAxisE,
                np.vstack((data[0, :], ThryE)),
                fig,
                ax[1],
                CurveNames=["Data", "Fit"],
                XLabel="Wavelength (nm)",
            )
            ax[1].set_xlim([400, 630])

    chisq = 0
    if config["other"]["extraoptions"]["fit_IAW"]:
        #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
        chisq = chisq + sum((data[1, :] - ThryI) ** 2)

    if config["other"]["extraoptions"]["fit_EPWb"]:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
        )

    if config["other"]["extraoptions"]["fit_EPWr"]:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
        )

    loss = 0
    if config["other"]["extraoptions"]["fit_IAW"]:
        loss = loss + np.sum(np.square(data[1, :] - ThryI) / i_data)

    if config["other"]["extraoptions"]["fit_EPWb"]:
        sqdev = np.square(data[0, :] - ThryE) / ThryE
        sqdev = np.where(
            (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"]),
            sqdev,
            0.0,
        )

        loss = loss + np.sum(sqdev)

    if config["other"]["extraoptions"]["fit_EPWr"]:
        sqdev = np.square(data[0, :] - ThryE) / ThryE
        sqdev = np.where(
            (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"]),
            sqdev,
            0.0,
        )

        loss = loss + np.sum(sqdev)

    print(loss)

    return fig, ax
