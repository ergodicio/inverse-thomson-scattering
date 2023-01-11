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
        fig, ax = plt.subplots(2, 1, sharex = True, height_ratios=[2, 1], figsize=(16, 4))
        ax0 = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax0 = ax
    
    cmap = TScmap()
    
    if logplot:
        C=log(C)
    
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
        line_colors= iter(["r","k","g","b"])
        for i in range(0,len(Line),2):
            ax0.plot(Line[i], Line[i+1], next(line_colors))
    

    if kaxis:
        ax2 = ax0.secondary_yaxis('right', fucntions=(forward_kaxis,backward_kaxis))
        secax_y2.set_ylabel(r'$~v_p/v_{th}$')

                                    
        def forward_kaxis(y):
            c=2.99792458e10
            omgL=2*np.pi*1e7*c/kaxis[2]
            omgpe = 5.64*10**4 *np.sqrt(kaxis[0])
            ko = np.sqrt((omgL**2 - omgpe**2)/c**2)
            newy=np.sqrt(kaxis[0]/(1000*kaxis[1]))*1./(1486*ko*np.sin(y/360 *np.pi))
            return newy
        
        def backward_kaxis(y):
            c=2.99792458e10
            omgL=2*np.pi*1e7*c/kaxis[2]
            omgpe = 5.64*10**4 *np.sqrt(kaxis[0])
            ko = np.sqrt((omgL**2 - omgpe**2)/c**2)
            newy=360/np.pi * np.arcsin(np.sqrt(kaxis[0]/(1000*kaxis[1]))*1./(1486*ko*y))
            return newy
                                    
    if Residuals:
        # possibly change to stem
        ax[1].plot(x, Residuals)

    if CurveNames:
        ax.legend(CurveNames)