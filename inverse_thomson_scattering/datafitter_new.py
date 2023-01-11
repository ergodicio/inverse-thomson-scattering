import time, mlflow, os, tempfile, jax, yaml

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from typing import Dict

from scipy.signal import convolve2d as conv2
from os.path import join,exists
from inverse_thomson_scattering.additional_functions import get_scattering_angles, plotinput
from inverse_thomson_scattering.evaluate_background import get_shot_bg, get_lineout_bg
from inverse_thomson_scattering.loadTSdata import loadData
from inverse_thomson_scattering.correctThroughput import correctThroughput
from inverse_thomson_scattering.getCalibrations import getCalibrations
from inverse_thomson_scattering.plotters import ColorPlots
from inverse_thomson_scattering.numDistFunc import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function
from inverse_thomson_scattering.fitmodl import get_fit_model
from inverse_thomson_scattering.plotstate import plotState

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
    #This function fits the Thomson scattering spectral dnesity fucntion to experimental data, or plots specified spectra. All inputs are derived from the input dictionary config.

    Summary of additional needs:
          A wrapper to allow for multiple lineouts or shots to be analyzed and gradients to be handled
          Better way to handle data finding since the location may change with computer or on a shot day
          Better way to handle shots with multiple types of data
          Way to handle calibrations which change from one to shot day to the next and have to be recalculated frequently (adding a new function to attempt this 8/8/22)
          A way to handle the expanded ion calculation when colapsing the spectrum to pixel resolution
          A way to handle different numbers of points

    Depreciated functions that need to be restored:
       Time axis alignment with fiducials
       interactive confirmation of new table creation
       ability to generate different table names without the default values


    Args:
        config:

    Returns:

    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    #initialize timer
    t0 = time.time()
    
    #get scattering angles and weights
    sa = get_scattering_angles(config["D"]["extraoptions"]["spectype"])
    
    #Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = getCalibrations(
        config["shotnum"], config["D"]["extraoptions"]["spectype"], config["D"]["CCDsize"])
    
    #load data
    [elecData, ionData, xlab, shift_zero] = loadData(
        config["shotnum"], config["D"]["shotDay"], config["D"]["extraoptions"]["spectype"],
        magE, config["D"]["extraoptions"])
    
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
    
    #Correct for spectral throughput
    if config["D"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(elecData, config["D"]["extraoptions"]["spectype"], axisyE, config["shotnum"])
        
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
        if config["D"]["extraoptions"]["spectype"] == 1:
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
    [BGele, BGion] = get_shot_bg(config, magE, axisyE, elecData)
    [noiseE, noiseI] = get_lineout_bg(config, elecData, ionData, BGele, BGion, LineoutTSE_smooth,
                                      BackgroundPixel, IAWtime, LineoutPixelE, LineoutPixelI)
    
    #Plot Data
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

    config["D"]["PhysParams"]["widIRF"] = stddev
    config["D"]["lamrangE"] = [axisyE[0], axisyE[-2]]
    config["D"]["lamrangI"] = [axisyI[0], axisyI[-2]]
    #config["D"]["npts"] = (len(LineoutTSE_norm) - 1) * 20
    config["D"]["npts"] = np.shape(LineoutTSE_norm)[1] * 10
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

    if config["optimizer"]["x_norm"]:
        norms = 2 * (ub - lb)
        shifts = lb
    else:
        norms = np.ones_like(x0)
        shifts = np.zeros_like(x0)

    x0 = (x0 - shifts) / norms
    lb = (lb - shifts) / norms
    ub = (ub - shifts) / norms
    bnds= list(zip(lb,ub))

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

    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})

    fit_model = get_fit_model(config, xie, sa)
    init_x = (x0 * norms + shifts).reshape((len(all_data), -1))
    final_x = (res.x * norms + shifts).reshape((len(all_data), -1))

    print("plotting")
    mlflow.set_tag("status", "plotting")
    if len(config["lineoutloc"]["val"]) > 4:
        plot_inds = np.random.choice(len(config["lineoutloc"]["val"]), 2, replace=False)
    else:
        plot_inds = np.arange(len(config["lineoutloc"]["val"]))

    t1 = time.time()
    fig = plt.figure(figsize=(14, 6))
    with tempfile.TemporaryDirectory() as td:
        for i in plot_inds:
            curline = config["lineoutloc"]["val"][i]
            cur_sa = dict(sa=sa["sa"], weights=sa["weights"][i])
            fig.clf()
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            # Plot initial guess
            fig, ax = plotState(
                init_x[i],
                config,
                config["D"]["PhysParams"]["amps"][i][0],
                xie,
                cur_sa,
                all_data[i][0],
                config["D"]["PhysParams"]["noiseE"][i] if config["D"]["extraoptions"]["load_ele_spec"] else config["D"]["PhysParams"]["noiseE"],
                config["D"]["PhysParams"]["noiseI"][i] if config["D"]["extraoptions"]["load_ion_spec"] else config["D"]["PhysParams"]["noiseI"],
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
                cur_sa,
                all_data[i][0],
                config["D"]["PhysParams"]["noiseE"][i] if config["D"]["extraoptions"]["load_ele_spec"] else config["D"]["PhysParams"]["noiseE"],
                config["D"]["PhysParams"]["noiseI"][i] if config["D"]["extraoptions"]["load_ion_spec"] else config["D"]["PhysParams"]["noiseI"],
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