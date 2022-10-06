import matplotlib.pyplot as plt
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
