import matplotlib as mpl
import mlflow, tempfile, os, pandas
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.mplot3d import axes3d


from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.model.physics.generate_spectra import FitModel
from inverse_thomson_scattering.misc.lineout_plot import lineout_plot


def get_final_params(config, best_weights_val, td):
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

    return all_params | dist


def plot_final_params():
    return


def plot_dist(config, final_params, sigma_fe, td):
    # Create fe image
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # lineouts = np.array(config["data"]["lineouts"]["val"])

    if config["parameters"]["fe"]["dim"] == 1:
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
    else:
        ax[0].plot_surface(X, Y, Z, edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)

        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        ax[0].contour(X, Y, Z, zdir="z", cmap="coolwarm")
        ax[0].contour(X, Y, Z, zdir="x", cmap="coolwarm")
        ax[0].contour(X, Y, Z, zdir="y", cmap="coolwarm")

    # no rolling sigma bc we use a smoothing kernel
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
    fig.savefig(os.path.join(td, "plots", "fe_final.png"), bbox_inches="tight")
    return


def plot_sigmas(config, all_params, best_weights_std, sigmas, td):
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

    # Create fit and data image
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    clevs = np.linspace(np.amin(savedata["data"]), np.amax(savedata["data"]), 21)
    ax[0].pcolormesh(
        angs,
        wavs,
        savedata["fit"],
        shading="nearest",
        cmap="gist_ncar",
        vmin=min(np.amin(savedata["data"]), 0),
        vmax=max(np.amax(savedata["data"]), 1),
    )
    ax[0].set_xlabel("Angle (degrees)")
    ax[0].set_ylabel("Wavelength (nm)")
    ax[1].pcolormesh(
        angs,
        wavs,
        savedata["data"],
        shading="nearest",
        cmap="gist_ncar",
        vmin=min(np.amin(savedata["data"]), 0),
        vmax=max(np.amax(savedata["data"]), 1),
    )
    ax[1].set_xlabel("Angle (degrees)")
    ax[1].set_ylabel("Wavelength (nm)")
    fig.savefig(os.path.join(td, "plots", "fit_and_data.png"), bbox_inches="tight")

    return savedata


def plot_lineouts(used_points, sqdevs, losses, all_params, all_axes, savedata, td):
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


def model_v_actual(
    sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds, yaxis, sorted_sqdev, sorted_red_losses
):
    if config["other"]["extraoptions"]["load_ion_spec"] and config["other"]["extraoptions"]["load_ele_spec"]:
        ele_s_ind = np.argmin(np.abs(yaxis[0] - config["plotting"]["ele_window_start"]))
        ele_e_ind = np.argmin(np.abs(yaxis[0] - config["plottting"]["ele_window_end"]))
        ion_s_ind = np.argmin(np.abs(yaxis[1] - config["plottting"]["ion_window_start"]))
        ion_e_ind = np.argmin(np.abs(yaxis[1] - config["plottting"]["ion_window_end"]))
        s_ind = [ele_s_ind, ion_s_ind]
        e_ind = [ele_e_ind, ion_e_ind]
    elif config["other"]["extraoptions"]["load_ion_spec"]:
        s_ind = [np.argmin(np.abs(yaxis - config["plottting"]["ion_window_start"]))]
        e_ind = [np.argmin(np.abs(yaxis - config["plottting"]["ion_window_end"]))]
        sorted_data = [sorted_data]
        sorted_fits = [sorted_fits]
        sorted_sqdev = [sorted_sqdev]

    elif config["other"]["extraoptions"]["load_ele_spec"]:
        s_ind = [np.argmin(np.abs(yaxis - config["plottting"]["ele_window_start"]))]
        e_ind = [np.argmin(np.abs(yaxis - config["plottting"]["ele_window_end"]))]
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
        C,
        cmap,
        interpolation="none",
        extent=[x[0], x[-1], y[-1], y[0]],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
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
