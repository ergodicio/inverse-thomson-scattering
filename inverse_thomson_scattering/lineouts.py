from collections import defaultdict

import numpy as np
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc.plotters import ColorPlots
from inverse_thomson_scattering.evaluate_background import get_lineout_bg


def get_lineouts(
    elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, shift_zero, IAWtime, xlab, sa, config
):
    # Convert lineout locations to pixel
    if config["data"]["lineouts"]["type"] == "ps" or config["data"]["lineouts"]["type"] == "um":
        LineoutPixelE = [np.argmin(abs(axisxE - loc - shift_zero)) for loc in config["data"]["lineouts"]["val"]]
    elif config["data"]["lineouts"]["type"] == "pixel":
        LineoutPixelE = config["data"]["lineouts"]["val"]
    else:
        raise NotImplementedError

    LineoutPixelI = LineoutPixelE

    if config["data"]["background"]["type"] == "ps":
        BackgroundPixel = np.argmin(abs(axisxE - config["data"]["background"]["slice"]))
    elif config["data"]["background"]["type"] == "pixel":
        BackgroundPixel = config["data"]["background"]["slice"]
    elif config["data"]["background"]["type"] == "auto":
        BackgroundPixel = LineoutPixelE + 100
    else:
        BackgroundPixel = []

    span = 2 * config["data"]["dpixel"] + 1  # (span must be odd)

    # extract lineouts
    if config["other"]["extraoptions"]["load_ele_spec"]:
        LineoutTSE = [
            np.sum(elecData[:, a - config["data"]["dpixel"] : a + config["data"]["dpixel"]], axis=1)
            for a in LineoutPixelE
        ]
        LineoutTSE_smooth = [
            np.convolve(LineoutTSE[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]
        if config["other"]["extraoptions"]["spectype"] == "angular":
            # print(np.shape(sa["weights"]))
            sa["weights"] = np.array(
                [
                    np.mean(sa["weights"][a - config["data"]["dpixel"] : a + config["data"]["dpixel"], :], axis=0)
                    for a in LineoutPixelE
                ]
            )
            # print(np.shape(sa["weights"]))
            sa["weights"] = sa["weights"][:, np.newaxis, :]
            # print(np.shape(sa["weights"]))
        else:
            # print(np.shape(sa["weights"]))
            sa["weights"] = sa["weights"] * np.ones([len(LineoutPixelE), len(sa["sa"])])
            # print(np.shape(sa["weights"]))

    if config["other"]["extraoptions"]["load_ion_spec"]:
        LineoutTSI = [
            np.mean(ionData[:, a - IAWtime - config["data"]["dpixel"] : a - IAWtime + config["data"]["dpixel"]], axis=1)
            for a in LineoutPixelI
        ]
        LineoutTSI_smooth = [
            np.convolve(LineoutTSI[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]  # was divided by 10 for some reason (removed 8-9-22)

    # Find background signal combining information from a background shot and background lineout
    [noiseE, noiseI] = get_lineout_bg(
        config,
        elecData,
        ionData,
        BGele,
        BGion,
        LineoutTSE_smooth,
        BackgroundPixel,
        IAWtime,
        LineoutPixelE,
        LineoutPixelI,
    )

    # Plot Data
    # if config["other"]["extraoptions"]["plot_raw_data"]:
    #     if config["other"]["extraoptions"]["load_ion_spec"]:
    #         ColorPlots(
    #             axisxI - shift_zero,
    #             axisyI,
    #             conv2(ionData - BGion, np.ones([5, 3]) / 15, mode="same"),
    #             Line=[
    #                 [axisxI[LineoutPixelI] - shift_zero, axisxI[LineoutPixelI] - shift_zero],
    #                 [axisyI[0], axisyI[-1]],
    #                 [axisxI[BackgroundPixel] - shift_zero, axisxI[BackgroundPixel] - shift_zero],
    #                 [axisyI[0], axisyI[-1]],
    #             ],
    #             vmin=0,
    #             XLabel=xlab,
    #             YLabel="Wavelength (nm)",
    #             title="Shot : " + str(config["data"]["shotnum"]) + " : " + "TS : Corrected and background subtracted",
    #         )
    #
    #     if config["other"]["extraoptions"]["load_ele_spec"]:
    #         ColorPlots(
    #             axisxE - shift_zero,
    #             axisyE,
    #             conv2(elecData - BGele, np.ones([5, 3]) / 15, mode="same"),
    #             Line=[
    #                 [axisxE[LineoutPixelE] - shift_zero, axisxE[LineoutPixelE] - shift_zero],
    #                 [axisyE[0], axisyE[-1]],
    #                 [axisxE[BackgroundPixel] - shift_zero, axisxE[BackgroundPixel] - shift_zero],
    #                 [axisyE[0], axisyE[-1]],
    #             ],
    #             vmin=0,
    #             XLabel=xlab,
    #             YLabel="Wavelength (nm)",
    #             title="Shot : " + str(config["data"]["shotnum"]) + " : " + "TS : Corrected and background subtracted",
    #         )

    # Find data amplitudes
    gain = config["other"]["gain"]
    if config["other"]["extraoptions"]["load_ion_spec"]:
        noiseI = noiseI / gain
        LineoutTSI_norm = [LineoutTSI_smooth[i] / gain for i, _ in enumerate(LineoutPixelI)]
        LineoutTSI_norm = np.array(LineoutTSI_norm)
        ampI = np.amax(LineoutTSI_norm - noiseI, axis=1)
    else:
        ampI = 1

    if config["other"]["extraoptions"]["load_ele_spec"]:
        noiseE = noiseE / gain
        LineoutTSE_norm = [LineoutTSE_smooth[i] / gain for i, _ in enumerate(LineoutPixelE)]
        LineoutTSE_norm = np.array(LineoutTSE_norm)
        ampE = np.amax(LineoutTSE_norm[:, 100:-1] - noiseE[:, 100:-1], axis=1)  # attempts to ignore 3w comtamination
    else:
        ampE = 1

    config["other"]["PhysParams"]["noiseI"] = noiseI
    config["other"]["PhysParams"]["noiseE"] = noiseE

    all_data = defaultdict(list)

    #for i, _ in enumerate(config["data"]["lineouts"]["val"]):
    #    # this probably needs to be done differently
    #    if config["other"]["extraoptions"]["load_ion_spec"] and config["other"]["extraoptions"]["load_ele_spec"]:
    #        data = np.vstack((LineoutTSE_norm[i], LineoutTSI_norm[i]))
    #        amps = [ampE[i], ampI[i]]
    #    elif config["other"]["extraoptions"]["load_ion_spec"]:
    #        data = np.vstack((LineoutTSI_norm[i], LineoutTSI_norm[i]))
    #        amps = [ampE, ampI[i]]
    #    elif config["other"]["extraoptions"]["load_ele_spec"]:
    #        data = np.vstack((LineoutTSE_norm[i], LineoutTSE_norm[i]))
    #        amps = [ampE[i], ampI]
    #    else:
    #        raise NotImplementedError("This spectrum does not exist")

    #    all_data["data"].append(data[None, :])
    #    all_data["amps"].append(np.array(amps)[None, :])

    #all_data = {k: np.concatenate(v) for k, v in all_data.items()}
    
    if config["other"]["extraoptions"]["load_ion_spec"]:
        all_data["i_data"] = LineoutTSI_norm
        all_data["i_amps"] = ampI
    else:
        all_data["i_data"] = all_data["i_amps"] = np.zeros(len(config["data"]["lineouts"]["val"]))
    if config["other"]["extraoptions"]["load_ele_spec"]:
        all_data["e_data"] = LineoutTSE_norm
        all_data["e_amps"] = ampE
    else:
        all_data["e_data"] = all_data["e_amps"] = np.zeros(len(config["data"]["lineouts"]["val"]))
    
    #all_data = {k: np.concatenate(v) for k, v in all_data.items()}

    return all_data
