import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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
    ax0.set_title(title,
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
        ax2 = ax0.secondary_yaxis('right', fucntions=(forward_kaxis, backward_kaxis))
        secax_y2.set_ylabel(r'$~v_p/v_{th}$')

        def forward_kaxis(y):
            c = 2.99792458e10
            omgL = 2 * np.pi * 1e7 * c / kaxis[2]
            omgpe = 5.64 * 10 ** 4 * np.sqrt(kaxis[0])
            ko = np.sqrt((omgL ** 2 - omgpe ** 2) / c ** 2)
            newy = np.sqrt(kaxis[0] / (1000 * kaxis[1])) * 1. / (1486 * ko * np.sin(y / 360 * np.pi))
            return newy

        def backward_kaxis(y):
            c = 2.99792458e10
            omgL = 2 * np.pi * 1e7 * c / kaxis[2]
            omgpe = 5.64 * 10 ** 4 * np.sqrt(kaxis[0])
            ko = np.sqrt((omgL ** 2 - omgpe ** 2) / c ** 2)
            newy = 360 / np.pi * np.arcsin(np.sqrt(kaxis[0] / (1000 * kaxis[1])) * 1. / (1486 * ko * y))
            return newy

    if Residuals:
        # possibly change to stem
        ax[1].plot(x, Residuals)

    if CurveNames:
        ax.legend(CurveNames)

def plotState(x, config, amps, xie, sas, data, noiseE, nosieI, fitModel2, fig, ax):
    [modlE, modlI, lamAxisE, lamAxisI, tsdict] = fitModel2(x, sas["weights"])

    lam = config["parameters"]["lam"]["val"]
    amp1 = config["parameters"]["amp1"]["val"]
    amp2 = config["parameters"]["amp2"]["val"]
    amp3 = config["parameters"]["amp3"]["val"]

    stddev = config["D"]["PhysParams"]["widIRF"]

    if config["D"]["extraoptions"]["load_ion_spec"]:
        originI = (max(lamAxisI) + min(lamAxisI)) / 2  # Conceptual_origin so the convolution donsn't shift the signal
        inst_funcI = np.squeeze(
            (1 / (stddev[1] * np.sqrt(2 * np.pi))) * np.exp(-((lamAxisI - originI) ** 2) / (2 * (stddev[1]) ** 2))
        )  # Gaussian
        ThryI = np.convolve(modlI, inst_funcI, "same")
        ThryI = (max(modlI) / max(ThryI)) * ThryI
        ThryI = np.average(ThryI.reshape(1024, -1), axis=1)

        if config["D"]["PhysParams"]["norm"] == 0:
            lamAxisI = np.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = amp3 * amps[1] * ThryI / max(ThryI)
            
        ThryI = ThryI + np.array(noiseI)

    if config["D"]["extraoptions"]["load_ele_spec"]:
        originE = (max(lamAxisE) + min(lamAxisE)) / 2  # Conceptual_origin so the convolution donsn't shift the signal
        inst_funcE = np.squeeze(
            (1 / (stddev[0] * np.sqrt(2 * np.pi))) * np.exp(-((lamAxisE - originE) ** 2) / (2 * (stddev[0]) ** 2))
        )  # Gaussian
        ThryE = np.convolve(modlE, inst_funcE, "same")
        ThryE = (max(modlE) / max(ThryE)) * ThryE

        if config["D"]["PhysParams"]["norm"] > 0:
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam] / max(ThryE[lamAxisE < lam]))
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam] / max(ThryE[lamAxisE > lam]))

        ThryE = np.average(ThryE.reshape(1024, -1), axis=1)
        if config["D"]["PhysParams"]["norm"] == 0:
            lamAxisE = np.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = amps[0] * ThryE / max(ThryE)
            ThryE[lamAxisE < lam] = amp1 * (ThryE[lamAxisE < lam])
            ThryE[lamAxisE > lam] = amp2 * (ThryE[lamAxisE > lam])
            
        ThryE = ThryE + np.array(noiseE)

    if config["D"]["extraoptions"]["spectype"] == 0:
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
        if config["D"]["extraoptions"]["load_ion_spec"]:
            LinePlots(
                lamAxisI,
                np.vstack((data[1, :], ThryI)),
                fig,
                ax[0],
                CurveNames=["Data", "Fit"],
                XLabel="Wavelength (nm)",
            )
            ax[0].set_xlim([525, 528])

        if config["D"]["extraoptions"]["load_ele_spec"]:
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
    if config["D"]["extraoptions"]["fit_IAW"]:
        #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
        chisq = chisq + sum((data[1, :] - ThryI) ** 2)

    if config["D"]["extraoptions"]["fit_EPWb"]:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
        )

    if config["D"]["extraoptions"]["fit_EPWr"]:
        chisq = chisq + sum(
            (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
        )

    return fig, ax
