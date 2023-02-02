import numpy as np
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc.plotters import ColorPlots
from inverse_thomson_scattering.evaluate_background import get_lineout_bg

def get_lineouts(elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, shift_zero, IAWtime, xlab, sa, config):
    # Convert lineout locations to pixel
    if config["lineoutloc"]["type"] == "ps" or config["lineoutloc"]["type"] == "um":
        LineoutPixelE = [np.argmin(abs(axisxE - loc - shift_zero)) for loc in config["lineoutloc"]["val"]]
    elif config["lineoutloc"]["type"] == "pixel":
        LineoutPixelE = config["lineoutloc"]["val"]
    LineoutPixelI = LineoutPixelE


    if config["bgloc"]["type"] == "ps":
        BackgroundPixel = np.argmin(abs(axisxE - config["bgloc"]["val"]))
    elif config["bgloc"]["type"] == "pixel":
        BackgroundPixel = config["bgloc"]["val"]
    elif config["bgloc"]["type"] == "auto":
        BackgroundPixel = LineoutPixelE + 100

    span = 2 * config["dpixel"] + 1  # (span must be odd)
    
    #extract lineouts
    if config["D"]["extraoptions"]["load_ele_spec"]:
        LineoutTSE = [
            np.mean(elecData[:, a - config["dpixel"] : a + config["dpixel"]], axis=1)
            for a in LineoutPixelE
        ]
        LineoutTSE_smooth = [
            np.convolve(LineoutTSE[i], np.ones(span) / span, "same") 
            for i, _ in enumerate(LineoutPixelE)
        ]
        if config["D"]["extraoptions"]["spectype"] == "angular":
            #print(np.shape(sa["weights"]))
            sa["weights"]=np.array([
                np.mean(sa["weights"][a - config["dpixel"] : a + config["dpixel"],:], axis=0)
                for a in LineoutPixelE
                ])
            #print(np.shape(sa["weights"]))
            sa["weights"]=sa["weights"][:,np.newaxis,:]
            #print(np.shape(sa["weights"]))
        else:
            #print(np.shape(sa["weights"]))
            sa["weights"]=sa["weights"]*np.ones([len(LineoutPixelE),len(sa["sa"])])
            #print(np.shape(sa["weights"]))

    if config["D"]["extraoptions"]["load_ion_spec"]:
        LineoutTSI = [
            np.mean(ionData[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1)
            for a in LineoutPixelI
        ]
        LineoutTSI_smooth = [
            np.convolve(LineoutTSI[i], np.ones(span) / span, "same") 
            for i, _ in enumerate(LineoutPixelE)
        ]  # was divided by 10 for some reason (removed 8-9-22)
        
        
    #Find background signal combining information from a background shot and background lineout
    [noiseE, noiseI] = get_lineout_bg(config, elecData, ionData, BGele, BGion, LineoutTSE_smooth, BackgroundPixel, IAWtime, LineoutPixelE, LineoutPixelI)
    
    #Plot Data
    if config["D"]["extraoptions"]["plot_raw_data"]:
        if config["D"]["extraoptions"]["load_ion_spec"]:
            ColorPlots(
                axisxI - shift_zero,
                axisyI,
                conv2(ionData-BGion, np.ones([5, 3]) / 15, mode="same"),
                Line=[[axisxI[LineoutPixelI] - shift_zero, axisxI[LineoutPixelI] - shift_zero],
                      [axisyI[0], axisyI[-1]], 
                      [axisxI[BackgroundPixel] - shift_zero, axisxI[BackgroundPixel] - shift_zero],
                      [axisyI[0], axisyI[-1]]],
                vmin=0,
                XLabel=xlab,
                YLabel="Wavelength (nm)",
                title="Shot : " + str(config["shotnum"]) + " : " + "TS : Corrected and background subtracted")

        if config["D"]["extraoptions"]["load_ele_spec"]:
            ColorPlots(
                axisxE - shift_zero,
                axisyE,
                conv2(elecData-BGele, np.ones([5, 3]) / 15, mode="same"),
                Line=[[axisxE[LineoutPixelE] - shift_zero, axisxE[LineoutPixelE] - shift_zero],
                      [axisyE[0], axisyE[-1]], 
                      [axisxE[BackgroundPixel] - shift_zero, axisxE[BackgroundPixel] - shift_zero],
                      [axisyE[0], axisyE[-1]]],
                vmin=0,
                XLabel=xlab,
                YLabel="Wavelength (nm)",
                title="Shot : " + str(config["shotnum"]) + " : " + "TS : Corrected and background subtracted")
        
    #Find data amplitudes
    gain = config["D"]["gain"]
    if config["D"]["extraoptions"]["load_ion_spec"]:
        noiseI = noiseI / gain
        LineoutTSI_norm = [LineoutTSI_smooth[i] / gain for i, _ in enumerate(LineoutPixelI)]
        LineoutTSI_norm = np.array(LineoutTSI_norm)
        ampI = np.amax(LineoutTSI_norm-noiseI, axis=1)
    else:
        ampI = 1

    if config["D"]["extraoptions"]["load_ele_spec"]:
        noiseE = noiseE / gain
        LineoutTSE_norm = [LineoutTSE_smooth[i] / gain for i, _ in enumerate(LineoutPixelE)]
        LineoutTSE_norm = np.array(LineoutTSE_norm)
        ampE = np.amax(LineoutTSE_norm[:, 100:-1]-noiseE[:, 100:-1], axis=1)  # attempts to ignore 3w comtamination
    else:
        ampE = 1
    
    config["D"]["PhysParams"]["noiseI"] = noiseI
    config["D"]["PhysParams"]["noiseE"] = noiseE
    
    all_data = []
    config["D"]["PhysParams"]["amps"] = []
    # run fitting code for each lineout
    for i, _ in enumerate(config["lineoutloc"]["val"]):
        # this probably needs to be done differently
        if config["D"]["extraoptions"]["load_ion_spec"] and config["D"]["extraoptions"]["load_ele_spec"]:
            data = np.vstack((LineoutTSE_norm[i], LineoutTSI_norm[i]))
            amps = [ampE[i], ampI[i]]
        elif config["D"]["extraoptions"]["load_ion_spec"]:
            data = np.vstack((LineoutTSI_norm[i], LineoutTSI_norm[i]))
            amps = [ampE, ampI[i]]
        elif config["D"]["extraoptions"]["load_ele_spec"]:
            data = np.vstack((LineoutTSE_norm[i], LineoutTSE_norm[i]))
            amps = [ampE[i], ampI]
        else:
            raise NotImplementedError("This spectrum does not exist")

        all_data.append(data[None, :])
        config["D"]["PhysParams"]["amps"].append(np.array(amps)[None, :])
        
    all_data = np.concatenate(all_data)
    
    return all_data