# Contains two functions one which quantifies the background for full images and one whcih quantifies the background for individual lineouts
import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2

from inverse_thomson_scattering.misc.load_ts_data import loadData
from inverse_thomson_scattering.process.correct_throughput import correctThroughput


def get_shot_bg(config, axisyE, elecData):
    if config["data"]["background"]["type"] == "Shot":
        [BGele, BGion, _, _] = loadData(
            config["data"]["background"]["slice"], config["data"]["shotDay"], config["other"]["extraoptions"]
        )
        if config["other"]["extraoptions"]["load_ion_spec"]:
            BGion = conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        else:
            BGion = 0
        if config["other"]["extraoptions"]["load_ele_spec"]:
            BGele = correctThroughput(
                BGele, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"]
            )
            if config["other"]["extraoptions"]["spectype"] == "angular":
                BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")  # 1/27 for H2 and 1/24 for kr
            else:
                BGele = conv2(BGele, np.ones([5, 3]) / 15, mode="same")
        else:
            BGele = 0
    elif config["other"]["extraoptions"]["spectype"] == "angular" and config["data"]["background"]["type"] == "Fit":
        [BGele, _, _, _] = loadData(
            config["data"]["background"]["slice"], config["data"]["shotDay"], config["other"]["extraoptions"]
        )

        BGele = correctThroughput(BGele, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"])

        BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")  # 1/27 for H2 and 1/24 for kr
        xx = np.arange(1024)

        def quadbg(x):
            res = np.sum(
                (elecData[1000, :] - ((x[0] * (xx - x[3]) ** 2 + x[1] * (xx - x[3]) + x[2]) * BGele[1000, :])) ** 2
            )
            return res

        corrfactor = spopt.minimize(quadbg, [0.1, 0.1, 1.15, 300])
        newBG = (
            corrfactor.x[0] * (xx - corrfactor.x[3]) ** 2 + corrfactor.x[1] * (xx - corrfactor.x[3]) + corrfactor.x[2]
        ) * BGele
        BGele = newBG

        print("Angular background corrected with polynomial model")
        print(corrfactor.x)
        BGion = 0
    else:
        BGele = 0
        BGion = 0

    return BGele, BGion


def get_lineout_bg(
    config, elecData, ionData, BGele, BGion, LineoutTSE_smooth, BackgroundPixel, IAWtime, LineoutPixelE, LineoutPixelI
):
    span = 2 * config["data"]["dpixel"] + 1  # (span must be odd)

    # Fit a model to the data edges to interperet the background
    if config["data"]["background"]["type"] == "Fit" and config["other"]["extraoptions"]["spectype"] != "angular":
        if config["other"]["extraoptions"]["load_ele_spec"]:
            # exp2 bg seems to be the best for some imaging data while rat11 is better in other cases but should be checked in more situations
            bgfitx = np.hstack([np.arange(100, 200), np.arange(800, 1023)])

            def exp2(x, a, b, c, d):
                return a * np.exp(b * x) + c * np.exp(d * x)

            # [expbg, _] = spopt.curve_fit(exp2,bgfitx,LineoutTSE_smooth[bgfitx])

            def power2(x, a, b, c):
                return a * x**b + c

            # [pwerbg, _] = spopt.curve_fit(power2,bgfitx,LineoutTSE_smooth[bgfitx])

            def rat21(x, a, b, c, d):
                return (a * x**2 + b * x + c) / (x + d)

            # [ratbg, _] = spopt.curve_fit(rat21,bgfitx,LineoutTSE_smooth[bgfitx])

            def rat11(x, a, b, c):
                return (a * x + b) / (x + c)

            LineoutBGE = []
            for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                [rat1bg, _] = spopt.curve_fit(rat11, bgfitx, LineoutTSE_smooth[i][bgfitx], [-16, 200000, 170])
                if config["data"]["background"]["show"]:
                    plt.plot(rat11(np.arange(1024), *rat1bg))
                    plt.plot(LineoutTSE_smooth[i])
                    plt.show()
                # LineoutTSE_smooth[i] = LineoutTSE_smooth[i] - rat11(np.arange(1024), *rat1bg)
                # the behaviour of this fit is different now when a BG shot is included (no effect without a BG shot
                LineoutBGE.append(rat11(np.arange(1024), *rat1bg))
            # print(np.shape(LineoutBGE))

    if config["other"]["extraoptions"]["load_ion_spec"]:
        # quantify a uniform background
        noiseI = np.mean(
            (ionData - BGion)[
                :, BackgroundPixel - config["data"]["dpixel"] : BackgroundPixel + config["data"]["dpixel"]
            ],
            1,
        )
        noiseI = np.convolve(noiseI, np.ones(span) / span, "same")
        bgfitx = np.hstack([np.arange(200, 400), np.arange(700, 850)])
        noiseI = np.mean(noiseI[bgfitx])
        noiseI = np.ones(1024) * config["data"]["bgscaleI"] * noiseI

        # add the uniform background to the background from the background shot
        if np.shape(BGion) == tuple(config["other"]["CCDsize"]):
            LineoutBGI = [
                np.mean(
                    BGion[:, a - IAWtime - config["data"]["dpixel"] : a - IAWtime + config["data"]["dpixel"]], axis=1
                )
                for a in LineoutPixelI
            ]
            noiseI = noiseI + LineoutBGI
        else:
            noiseI = noiseI * np.ones((len(LineoutPixelI), 1))
    else:
        noiseI = 0

    if config["other"]["extraoptions"]["load_ele_spec"] and config["data"]["background"]["type"] != "Fit":
        # This model is in conflict with the fitted background since they both attempt to quantify the background from the data effectively attempting to remove the same information

        # quantify a background lineout
        noiseE = np.mean(
            (elecData - BGele)[
                :, BackgroundPixel - config["data"]["dpixel"] : BackgroundPixel + config["data"]["dpixel"]
            ],
            1,
        )
        noiseE = np.convolve(noiseE, np.ones(span) / span, "same")

        # replace background lineout with double exponential for extra smoothing
        if config["other"]["extraoptions"]["spectype"] != "angular":

            def exp2(x, a, b, c, d):
                return a * np.exp(-b * x) + c * np.exp(-d * x)

            bgfitx = np.hstack(
                [np.arange(250, 480), np.arange(540, 900)]
            )  # this is specificaly targeted at streaked data, removes the fiducials at top and bottom and notch filter
            bgfitx2 = np.hstack([np.arange(250, 300), np.arange(700, 900)])
            plt.plot(bgfitx, noiseE[bgfitx])
            # [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[1000, 0.001, 1000, 0.001])
            [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[200, 0.001, 200, 0.001])
            noiseE = config["data"]["bgscaleE"] * exp2(np.arange(1024), *expbg)

            # rescale background exponential using the edge of each data lineout
            noiseE_rescaled = []
            for i, _ in enumerate(config["data"]["lineouts"]["val"]):
                scale = spopt.minimize_scalar(
                    lambda a: np.sum(abs(LineoutTSE_smooth[i][bgfitx2] - a * noiseE[bgfitx2]))
                )

                noiseE_rescaled.append(scale.x * noiseE)

            noiseE = np.array(noiseE_rescaled)
            # if config["data"]["background"]["show"]:
            #     plt.plot(bgfitx, noiseE[bgfitx])
            #     plt.plot(bgfitx, scale.x * noiseE[bgfitx])
            #     plt.plot(bgfitx, exp2(bgfitx, 200, 0.001, 200, 0.001))
            #     lin = np.mean((elecData - BGele)[:, 480 - config["data"]["dpixel"] : 480 + config["data"]["dpixel"]], 1)
            #     plt.plot(bgfitx, lin[bgfitx])
            #     plt.show()

        # constant addition to the background
        noiseE += config["other"]["flatbg"]
    else:
        noiseE = 0
    if config["other"]["extraoptions"]["load_ele_spec"]:
        if np.shape(BGele) == tuple(config["other"]["CCDsize"]):
            LineoutBGE2 = [
                np.mean(BGele[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
                for a in LineoutPixelE
            ]
            noiseE = noiseE + np.array(LineoutBGE2)
        else:
            noiseE = noiseE * np.ones((len(LineoutPixelE), 1))

        if "LineoutBGE" in locals():
            # print(np.shape(noiseE))
            noiseE = noiseE + np.array(LineoutBGE)
            # print(np.shape(noiseE))
    else:
        noiseE = 0

    # print(np.shape(noiseE))

    return noiseE, noiseI
