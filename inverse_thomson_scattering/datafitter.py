import tempfile, os
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.optimize as spopt
import time
import mlflow, jax
import yaml

from scipy.signal import convolve2d as conv2
from numpy.linalg import inv
from inverse_thomson_scattering.loadTSdata import loadData
from inverse_thomson_scattering.correctThroughput import correctThroughput
from inverse_thomson_scattering.getCalibrations import getCalibrations
from inverse_thomson_scattering.numDistFunc import get_num_dist_func
from inverse_thomson_scattering.plotstate import plotState
from inverse_thomson_scattering.fitmodl import get_fit_model
from inverse_thomson_scattering.loss_function import get_loss_function


def unnumpy_dict(this_dict: Dict):
    new_dict = {}
    for k, v in this_dict.items():
        if isinstance(v, Dict):
            new_v = unnumpy_dict(v)
        elif isinstance(v, np.ndarray):
            new_v = [float(val) for val in v]
        elif isinstance(v, jax.numpy.ndarray):
            new_v = [float(val) for val in v]
        else:
            new_v = v

        new_dict[k] = new_v

    return new_dict


def fit(config):
    """
    # function description [from Ang](update once complete)
    This function takes the inputs from the ANGTSDATAFITTERGUI and preforms
    the data corrections then fits the data returning the fit result

    The inputs from the GUI are Shot number, lineout locations, background shot
    number, probe wavelength, electron temperature, electron density, m, amp1
    and amp2, ionization state, starting distribution function
    type and the number of distribution function points to use in numerical
    distribution function fitting.

    Summary of additional needs:
          A wrapper to allow for multiple lineouts or shots to be analyzed and gradients to be handled
          A way to store shot data from one call to the next (this code is frequently used on the same shot repeatedly)
          Better way to handle data finding since the location may change with computer or on a shot day
          Better way to hadnle shots with multiple types of data
          Way to handle calibrations which change from one to shot day to the next and have to be recalculated frequently (adding a new function to attempt this 8/8/22)
          Potentially move the default values, especially the calibration into the input file
          A way to handle the expanded ion calculation when colapsing the spectrum to pixel resolution
          A way to handle different numbers of points

    Depreciated functions that need to be restored:
       Streaked EPW warp correction
       Time axis alignment with fiducials
       persistents
       persistends in numDistFunc
       interactive confirmation of new table creation
       ability to generate different table names without the default values


    Args:
        config:

    Returns:

    """

    t0 = time.time()
    ## Hard code toggles, locations and values
    # these should only be changed if something changes in the experimental setup or data is moved around

    tstype = config["D"]["extraoptions"]["spectype"]  # 1 for ARTS, 2 for TRTS, 3 for SRTS

    # lines 75 through 85 can likely be moved to the input decks
    CCDsize = [1024, 1024]  # dimensions of the CCD chip as read
    shift_zero = 0

    # need a better way to do this
    shotDay = 0  # turn on to switch file retrieval to shot day location

    gain = 1
    bgscalingE = config["bgscale"]  # multiplicitive factor on the EPW BG lineout
    bgscalingI = 0.1  # multiplicitive factor on the IAW BG lineout
    bgshotmult = 1
    flatbg = 0

    # Scattering angle in degrees for TIM6 TS
    if tstype > 1:
        sa = dict(
            sa=np.linspace(53.637560, 66.1191, 10),
            weights=np.array(
                [
                    0.00702671050853565,
                    0.0391423809738300,
                    0.0917976667717670,
                    0.150308544660150,
                    0.189541011666141,
                    0.195351560740507,
                    0.164271879645061,
                    0.106526733030044,
                    0.0474753389486960,
                    0.00855817305526778,
                ]
            ),
        )
    else:
        sa = dict(sa=np.arange(19, 139, 0.5), weights=np.vstack(np.loadtxt("files/angleWghtsFredfine.txt")))

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

    # Retrieve calibrated axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = getCalibrations(config["shotnum"], tstype, CCDsize)

    [elecData, ionData, xlab, shift_zero] = loadData(
        config["shotnum"], shotDay, tstype, magE, config["D"]["extraoptions"]
    )

    # turn off ion or electron fitting if the corresponding spectrum was not loaded
    if not config["D"]["extraoptions"]["load_ion_spec"]:
        config["D"]["extraoptions"]["fit_IAW"] = 0
        print("IAW data not loaded, omitting IAW fit")
    if not config["D"]["extraoptions"]["load_ele_spec"]:
        config["D"]["extraoptions"]["fit_EPWb"] = 0
        config["D"]["extraoptions"]["fit_EPWr"] = 0
        print("EPW data not loaded, omitting EPW fit")

    #If no data is loaded or being fit plot the input and quit
    if config["lineoutloc"]["val"]==[] or (config["D"]["extraoptions"]["fit_IAW"]+config["D"]["extraoptions"]["fit_EPWb"]+config["D"]["extraoptions"]["fit_EPWr"]) == 0 or (elecData==[] and ionData==[]):
        print("No data loaded, plotting input")
        plotinput(config, sa)
        return []
    
    
    if config["D"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(elecData, tstype, axisyE)

    # Background Shot
    if config["bgshot"]["type"] == "Shot":
        [BGele, BGion, _, _] = loadData(config["bgshot"]["val"], shotDay, tstype, magE, config["D"]["extraoptions"])
        if config["D"]["extraoptions"]["load_ion_spec"]:
            BGion = conv2(BGion, np.ones([5, 3]) / 15, mode="same")
            ionData_bsub = ionData - conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        if config["D"]["extraoptions"]["load_ele_spec"]:
            BGele = correctThroughput(BGele, tstype, axisyE)
            if tstype == 1:
                BGele = conv2(BGele, np.ones([5, 5]) / 25, mode="same")
                elecData_bsub = elecData - bgshotmult * conv2(BGele, np.ones([5, 5]) / 25, mode="same")
            else:
                BGele = conv2(BGele, np.ones([5, 3]) / 15, mode="same")
                elecData_bsub = elecData - bgshotmult * conv2(BGele, np.ones([5, 3]) / 15, mode="same")

    else:
        elecData_bsub = elecData
        ionData_bsub = ionData

    # Assign lineout locations
    if config["lineoutloc"]["type"] == "ps":
        LineoutPixelE = [np.argmin(abs(axisxE - loc - shift_zero)) for loc in config["lineoutloc"]["val"]]
        LineoutPixelI = LineoutPixelE

    elif config["lineoutloc"]["type"] == "um":  # [char(hex2dec('03bc')) 'm']:
        LineoutPixelE = [np.argmin(abs(axisxE - loc)) for loc in config["lineoutloc"]["val"]]
        LineoutPixelI = LineoutPixelE

    elif config["lineoutloc"]["type"] == "pixel":
        LineoutPixelE = config["lineoutloc"]["val"]
        LineoutPixelI = LineoutPixelE

    if config["bgloc"]["type"] == "ps":
        BackgroundPixel = np.argmin(abs(axisxE - config["bgloc"]["val"]))

    elif config["bgloc"]["type"] == "pixel":
        BackgroundPixel = config["bgloc"]["val"]

    elif config["bgloc"]["type"] == "auto":
        BackgroundPixel = LineoutPixelE + 100

    span = 2 * config["dpixel"] + 1
    # (span must be odd)

    if config["D"]["extraoptions"]["load_ele_spec"]:
        LineoutTSE = [
            np.mean(elecData[:, a - config["dpixel"] : a + config["dpixel"]], axis=1) for a in LineoutPixelE
            #np.mean(elecData_bsub[:, a - config["dpixel"] : a + config["dpixel"]], axis=1) for a in LineoutPixelE
        ]
        LineoutTSE_smooth = [
            np.convolve(LineoutTSE[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]

    if config["D"]["extraoptions"]["load_ion_spec"]:
        LineoutTSI = [
            np.mean(ionData[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1)
            #np.mean(ionData_bsub[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1)
            for a in LineoutPixelI
        ]
        LineoutTSI_smooth = [
            np.convolve(LineoutTSI[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]  # was divided by 10 for some reason (removed 8-9-22)

    if config["bgshot"]["type"] == "Fit":
        if config["D"]["extraoptions"]["load_ele_spec"]:
            if tstype == 1:
                [BGele, _, _, _] = loadData(
                    config["bgshot"]["val"], shotDay, tstype, magE, config["D"]["extraoptions"]
                )
                xx = np.arange(1024)

                def qaudbg(x):
                    np.sum(
                        (elecData[1000, :] - ((x[0] * (xx - x[3]) ** 2 + x[1] * (xx - x[3]) + x[2]) * BGele[1000, :]))
                        ** 2
                    )

                corrfactor = spopt.minimize(quadbg, [0.1, 0.1, 1.15, 300])
                newBG = (
                    corrfactor.x[0] * (xx - corrfactor.x[3]) ** 2
                    + corrfactor.x[1] * (xx - corrfactor.x[3])
                    + corrfactor[2]
                ) * BGele
                BGele = newBG
                #elecData_bsub = elecData - newBG
            else:
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

                LineoutBGE=[]
                for i, _ in enumerate(config["lineoutloc"]["val"]):
                    [rat1bg, _] = spopt.curve_fit(rat11, bgfitx, LineoutTSE_smooth[i][bgfitx],[-16,200000,170])
                    #plt.plot(rat11(np.arange(1024), *rat1bg))
                    #plt.plot(LineoutTSE_smooth[i])
                    #plt.show()
                    #LineoutTSE_smooth[i] = LineoutTSE_smooth[i] - rat11(np.arange(1024), *rat1bg)
                    #the behaviour of this fit is different now when a BG shot is included (no effect without a BG shot
                    LineoutBGE.append(rat11(np.arange(1024), *rat1bg)[None,:])
                print(np.shape(LineoutBGE))

    # Attempt to quantify any residual background
    # this has been switched from mean of elecData to mean of elecData_bsub 8-9-22
    if config["D"]["extraoptions"]["load_ion_spec"]:
        noiseI = np.mean(ionData_bsub[:, BackgroundPixel - config["dpixel"] : BackgroundPixel + config["dpixel"]], 1)
        noiseI = np.convolve(noiseI, np.ones(span) / span, "same")
        bgfitx = np.hstack([np.arange(200, 400), np.arange(700, 850)])
        noiseI = np.mean(noiseI[bgfitx])
        noiseI = np.ones(1024) * bgscalingI * noiseI
        
        if 'BGion' in locals():
            LineoutBGI = [
            np.mean(BGion[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1) for a in LineoutPixelI]
            noiseI = noiseI + LineoutBGI
        else:
            noiseI = noiseI * np.ones((len(LineoutPixelI),1))

    if config["D"]["extraoptions"]["load_ele_spec"]:
        noiseE = np.mean(elecData_bsub[:, BackgroundPixel - config["dpixel"] : BackgroundPixel + config["dpixel"]], 1)
        noiseE = np.convolve(noiseE, np.ones(span) / span, "same")
        #This model is in conflict with the fitted background since they both affect the data but are performed on different data
        def exp2(x, a, b, c, d):
            return a * np.exp(-b * x) + c * np.exp(-d * x)

        bgfitx = np.hstack(
            [np.arange(250, 480), np.arange(540, 900)]
        )  # this is specificaly targeted at streaked data, removes the fiducials at top and bottom and notch filter
        plt.plot(bgfitx,noiseE[bgfitx])
        #[expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[1000, 0.001, 1000, 0.001])
        [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[200, 0.001, 200, 0.001])
        noiseE = bgscalingE * exp2(np.arange(1024), *expbg)
        plt.plot(bgfitx,noiseE[bgfitx])
        plt.plot(bgfitx,exp2(bgfitx,200,0.001,200,0.001))
        plt.show()

        # temporary constant addition to the background
        noiseE = noiseE + flatbg
        
        if 'BGele' in locals():
            LineoutBGE2 = [
            np.mean(BGele[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1) for a in LineoutPixelI]
            noiseE = noiseE + LineoutBGE2
        else:
            noiseE = noiseE * np.ones((len(LineoutPixelE),1))
            
        if 'LineoutBGE' in locals():
            noiseE = noiseE + LineoutBGE

    ## Plot data
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    if config["D"]["extraoptions"]["load_ion_spec"]:
        imI = ax[1].imshow(
            conv2(ionData_bsub, np.ones([5, 3]) / 15, mode="same"),
            cmap,
            interpolation="none",
            extent=[axisxI[0] - shift_zero, axisxI[-1] - shift_zero, axisyI[-1], axisyI[0]],
            aspect="auto",
            vmin=0,
        )
        ax[1].set_title(
            "Shot : " + str(config["shotnum"]) + " : " + "TS : Corrected and background subtracted",
            fontdict={"fontsize": 10, "fontweight": "bold"},
        )
        ax[1].set_xlabel(xlab)
        ax[1].set_ylabel("Wavelength (nm)")
        plt.colorbar(imI, ax=ax[1])
        ax[1].plot(
            [axisxI[LineoutPixelI] - shift_zero, axisxI[LineoutPixelI] - shift_zero], [axisyI[0], axisyI[-1]], "r"
        )
        ax[1].plot(
            [axisxI[BackgroundPixel] - shift_zero, axisxI[BackgroundPixel] - shift_zero], [axisyI[0], axisyI[-1]], "k"
        )

    if config["D"]["extraoptions"]["load_ele_spec"]:
        imE = ax[0].imshow(
            conv2(elecData_bsub, np.ones([5, 3]) / 15, mode="same"),
            cmap,
            interpolation="none",
            extent=[axisxE[0] - shift_zero, axisxE[-1] - shift_zero, axisyE[-1], axisyE[0]],
            aspect="auto",
            vmin=0,
        )
        ax[0].set_title(
            "Shot : " + str(config["shotnum"]) + " : " + "TS : Corrected and background subtracted",
            fontdict={"fontsize": 10, "fontweight": "bold"},
        )
        ax[0].set_xlabel(xlab)
        ax[0].set_ylabel("Wavelength (nm)")
        plt.colorbar(imE, ax=ax[0])
        ax[0].plot(
            [axisxE[LineoutPixelE] - shift_zero, axisxE[LineoutPixelE] - shift_zero], [axisyE[0], axisyE[-1]], "r"
        )
        ax[0].plot(
            [axisxE[BackgroundPixel] - shift_zero, axisxE[BackgroundPixel] - shift_zero], [axisyE[0], axisyE[-1]], "k"
        )

    # Normalize Data before fitting
    if config["D"]["extraoptions"]["load_ion_spec"]:
        noiseI = noiseI / gain
        LineoutTSI_norm = [LineoutTSI_smooth[i] / gain for i, _ in enumerate(LineoutPixelI)]
        LineoutTSI_norm = np.array(LineoutTSI_norm)
        #LineoutTSI_norm = LineoutTSI_norm - noiseI  # new 6-29-20
        ampI = np.amax(LineoutTSI_norm-noiseI, axis=1)
    else:
        ampI = 1
        noiseI = []

    if config["D"]["extraoptions"]["load_ele_spec"]:
        noiseE = noiseE / gain
        LineoutTSE_norm = [LineoutTSE_smooth[i] / gain for i, _ in enumerate(LineoutPixelE)]
        LineoutTSE_norm = np.array(LineoutTSE_norm)
        #LineoutTSE_norm = LineoutTSE_norm - noiseE  # new 6-29-20
        ampE = np.amax(LineoutTSE_norm[:, 100:-1]-noiseE[:, 100:-1], axis=1)  # attempts to ignore 3w comtamination
    else:
        ampE = 1
        noiseE = []

    config["D"]["PhysParams"]["widIRF"] = stddev
    config["D"]["lamrangE"] = [axisyE[0], axisyE[-2]]
    config["D"]["lamrangI"] = [axisyI[0], axisyI[-2]]
    config["D"]["npts"] = (len(LineoutTSE_norm) - 1) * 20
    config["D"]["PhysParams"]["noiseI"] = noiseI
    config["D"]["PhysParams"]["noiseE"] = noiseE

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
    for i, _ in enumerate(config["lineoutloc"]["val"]):
        for key in parameters.keys():
            if parameters[key]["active"]:
                if np.size(parameters[key]["val"])>1:
                    x0.append(parameters[key]["val"][i])
                elif isinstance(parameters[key]["val"], list):
                    x0.append(parameters[key]["val"][0])
                else:
                    x0.append(parameters[key]["val"])
                lb.append(parameters[key]["lb"])
                ub.append(parameters[key]["ub"])

    x0=np.array(x0)
    lb=np.array(lb)
    ub=np.array(ub)
    #print(x0)
    
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

    #x0 = np.repeat(np.array(x0)[None, :], repeats=len(all_data), axis=0).flatten()
    #lb = np.repeat(np.array(lb)[None, :], repeats=len(all_data), axis=0).flatten()
    #ub = np.repeat(np.array(ub)[None, :], repeats=len(all_data), axis=0).flatten()
    if config["optimizer"]["x_norm"]:
        norms = 2 * (ub - lb)
        shifts = lb
    else:
        norms = np.ones_like(x0)
        shifts = np.zeros_like(x0)

    x0 = (x0 - shifts) / norms
    lb = (lb - shifts) / norms
    ub = (ub - shifts) / norms
    #print(x0)
    #print(shifts)
    #print(norms)
    bnds= list(zip(lb,ub))
    #testa, testb = zip(*bnds)
    #print(bnds)

    loss_fn, vg_loss_fn, hess_fn = get_loss_function(config, xie, sa, np.concatenate(all_data), norms, shifts)

    t1 = time.time()
    print("minimizing")
    mlflow.set_tag("status", "minimizing")
    # Perform fit
    if np.shape(x0)[0] != 0:
        if config["optimizer"]["method"]=="basinhopping":
            res = spopt.basinhopping(vg_loss_fn, x0, T=5e6, niter= 10, stepsize=0.004, disp=True,
                                     minimizer_kwargs = {"method": "L-BFGS-B", "jac": True,
                                                         "bounds": bnds,
                                                         "options": {"disp": True}})
        elif config["optimizer"]["method"]=="shgo":
            res = spopt.shgo(vg_loss_fn, bounds=bnds,
                             minimizer_kwargs = {"method": "L-BFGS-B", "jac": True},
                             options={"disp": True, "jac": True})
        elif config["optimizer"]["method"]=="dual_annealing":
            res = spopt.dual_annealing(loss_fn, bounds=bnds, maxiter = 20, initial_temp=500, maxfun = 1e5,
                             minimizer_kwargs = {"method": "L-BFGS-B", "jac": vg_loss_fn(1), "bounds": bnds, "options": {"disp": True, "maxiter": 100}},
                             x0=x0)
        else:
            res = spopt.minimize(
                vg_loss_fn if config["optimizer"]["grad_method"] == "AD" else loss_fn,
                x0,
                method=config["optimizer"]["method"],
                jac=True if config["optimizer"]["grad_method"] == "AD" else False,
                hess=hess_fn if config["optimizer"]["hessian"] else None,
                bounds=zip(lb, ub),
                options={"disp": True},
            )
    else:
        x = x0

    #print(res.status)
    #print(res.message)
    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})

    fit_model = get_fit_model(config, xie, sa)
    init_x = (x0 * norms + shifts).reshape((len(all_data), -1))
    final_x = (res.x * norms + shifts).reshape((len(all_data), -1))
    #print(final_x)
    #print(loss_fn(res.x))
    print(vg_loss_fn(res.x))
    #print(hess_fn(np.array(res.x.reshape((len(all_data), -1)))))
    hess_val = hess_fn(np.array(res.x.reshape((len(all_data), -1))))
    hess_val = hess_val.reshape(len(res.x),len(res.x))
    #print(np.shape(res.x))
    #print(np.shape(res.x.reshape((len(all_data), -1))))
    print(np.shape(hess_val))
    print(hess_val)
    
    cov_mat = 2.*inv(hess_val)
    print(cov_mat)
    sigmas = np.sqrt(np.diag(cov_mat))
    print(sigmas)
    sigmas = sigmas.reshape((len(all_data), -1))
    print(sigmas)
    

    print("plotting")
    mlflow.set_tag("status", "plotting")
    if len(config["lineoutloc"]["val"]) > 4:
        plot_inds = np.random.choice(len(config["lineoutloc"]["val"]), 2, replace=False)
    else:
        #plot_inds = config["lineoutloc"]["val"]
        plot_inds = np.arange(len(config["lineoutloc"]["val"]))

    t1 = time.time()
    fig = plt.figure(figsize=(14, 6))
    with tempfile.TemporaryDirectory() as td:
        for i in plot_inds:
            curline=config["lineoutloc"]["val"][i]
            fig.clf()
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            # Plot initial guess
            fig, ax = plotState(
                init_x[i],
                config,
                config["D"]["PhysParams"]["amps"][i][0],
                xie,
                sa,
                all_data[i][0],
                fitModel2=fit_model,
                fig=fig,
                ax=[ax, ax2],
            )
            fig.savefig(os.path.join(td, f"before-{curline}.png"), bbox_inches="tight")

            fig.clf()
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            fig, ax = plotState(
                final_x[i],
                config,
                config["D"]["PhysParams"]["amps"][i][0],
                xie,
                sa,
                all_data[i][0],
                fitModel2=fit_model,
                fig=fig,
                ax=[ax, ax2],
            )
            fig.savefig(os.path.join(td, f"after-{curline}.png"), bbox_inches="tight")
        mlflow.log_artifacts(td, artifact_path="plots")

    print(res)
    metrics_dict = {"loss": res.fun, "num_iterations": res.nit, "num_fun_eval": res.nfev}
    mlflow.log_metrics({"plot_time": round(time.time() - t1, 2)})
    mlflow.log_metrics(metrics_dict)

    mlflow.set_tag("status", "done plotting")

    result = config["parameters"]
    count = 0
    final_x.reshape((len(config["lineoutloc"]["val"]), -1))

    outputs = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            # config["parameters"][key]["val"] = [float(val) for val in list(final_x[:, count])]
            outputs[key] = [float(val) for val in list(final_x[:, count])]
            outputs[key]["uncertainty"] = [float(val) for val in list(sigmas[:, count])]
            count = count + 1

    # needs to be fixed
    # if result["fe"]["active"]:
    #    result["fe"]["val"] = res.x[-result["fe"]["length"] : :]
    # elif result["m"]["active"]:
    #    TSinputs["fe"]["val"] = np.log(NumDistFunc(TSinputs["m"]["val"]))  # initFe(result, xie)
    
    #mlflow.log_params(config["parameters"])
    #result = config["parameters"]

    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "ts_parameters.yaml"), "w") as fi:
            yaml.dump(outputs, fi)

        mlflow.log_artifacts(td)
    result = config["parameters"]

    def plotinput(config, sa, fit_model):
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
        for i, _ in enumerate(config["lineoutloc"]["val"]):
            for key in parameters.keys():
                if parameters[key]["active"]:
                    if np.size(parameters[key]["val"])>1:
                        x0.append(parameters[key]["val"][i])
                    elif isinstance(parameters[key]["val"], list):
                        x0.append(parameters[key]["val"][0])
                    else:
                        x0.append(parameters[key]["val"])
                    lb.append(parameters[key]["lb"])
                    ub.append(parameters[key]["ub"])

        x0=np.array(x0)
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
    
    return result