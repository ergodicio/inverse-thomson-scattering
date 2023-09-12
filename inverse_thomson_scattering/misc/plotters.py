import matplotlib as mpl
import mlflow, tempfile, os
import numpy as np
import matplotlib.pyplot as plt


from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.model.physics.generate_spectra import get_fit_model


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
    fit_model = get_fit_model(config, xie, sa)

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


def model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds, yaxis, sorted_sqdev, sorted_red_losses):
    # make plots
    for i in range(num_plots):
        # plot model vs actual
        titlestr = (
            r"|Error|$^2$" + f" = {sorted_losses[i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[i]]}"
        )
        filename = f"loss={sorted_losses[i]:.2e}-reduced_loss={sorted_red_losses[i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[-1 - i]]}.png"
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True, sharex = True)
        ax[0].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_data[i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Data")
        ax[0].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_fits[i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Fit")
        ax[0].set_title(titlestr, fontsize=14)
        ax[0].set_ylabel("Amp (arb. units)")
        ax[0].legend(fontsize=14)
        ax[0].grid()
        ax[1].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_sqdev[i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Residual")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].set_ylabel("$\chi_i^2$")
        fig.savefig(os.path.join(td, "worst", filename), bbox_inches="tight")
        plt.close(fig)

        titlestr = (
            r"|Error|$^2$"
            + f" = {sorted_losses[-1 - i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[-1 - i]]}"
        )
        filename = f"loss={sorted_losses[-1 - i]:.2e}-reduced_loss={sorted_red_losses[-1 - i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[-1 - i]]}.png"
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True, sharex = True)
        ax[0].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_data[-1 - i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Data")
        ax[0].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_fits[-1 - i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Fit")
        ax[0].set_title(titlestr, fontsize=14)
        ax[0].set_ylabel("Amp (arb. units)")
        ax[0].legend(fontsize=14)
        ax[0].grid()
        ax[1].plot(yaxis[config["other"]["crop_window"]:-config["other"]["crop_window"]], np.squeeze(sorted_sqdev[-1 - i, config["other"]["crop_window"]:-config["other"]["crop_window"]]), label="Residual")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].set_ylabel("$\chi_i^2$")
        fig.savefig(os.path.join(td, "best", filename), bbox_inches="tight")
        plt.close(fig)



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
    XLabel="\lambda_s (nm)",
    YLabel="\theta (\circ)",
    CurveNames=[],
    Residuals=[],
    title="EPW feature vs \theta",
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
        C = log(C)

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
