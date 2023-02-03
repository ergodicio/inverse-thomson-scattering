import time, mlflow, os, tempfile, yaml

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc.additional_functions import get_scattering_angles, plotinput, initialize_parameters
from inverse_thomson_scattering.evaluate_background import get_shot_bg
from inverse_thomson_scattering.misc.load_ts_data import loadData
from inverse_thomson_scattering.process.correct_throughput import correctThroughput
from inverse_thomson_scattering.misc.calibration import get_calibrations
from inverse_thomson_scattering.lineouts import get_lineouts
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function
from inverse_thomson_scattering.generate_spectra import get_fit_model
from inverse_thomson_scattering.misc.plotters import plotState

#def unnumpy_dict(this_dict: Dict):
#    new_dict = {}
#    for k, v in this_dict.items():
#        if isinstance(v, Dict):
#            new_v = unnumpy_dict(v)
#        elif isinstance(v, np.ndarray):
#            new_v = [float(val) for val in v]
#        elif isinstance(v, jax.numpy.ndarray):
#            new_v = [float(val) for val in v]
#        else:
#            new_v = v
#
#        new_dict[k] = new_v
#
#    return new_dict

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
    #initialize timer
    t0 = time.time()
    
    #load data
    [elecData, ionData, xlab, config["D"]["extraoptions"]["spectype"]] = loadData(
        config["shotnum"], config["D"]["shotDay"], config["D"]["extraoptions"])
    
    #get scattering angles and weights
    sa = get_scattering_angles(config["D"]["extraoptions"]["spectype"])
    
    #Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = get_calibrations(
        config["shotnum"], config["D"]["extraoptions"]["spectype"], config["D"]["CCDsize"])
    
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
    
    
    #load and correct background
    [BGele, BGion] = get_shot_bg(config, axisyE, elecData)
    
    #extract ARTS section
    if (config["lineoutloc"]["type"] == "range") & (config["D"]["extraoptions"]["spectype"] == "angular"):
        
        config["D"]["extraoptions"]["spectype"] = "angular_full"
        config["D"]["PhysParams"]["amps"] = np.array([np.amax(elecData), 1])
        sa["angAxis"] = axisxE
        
        if config["D"]["extraoptions"]["plot_raw_data"]:
            ColorPlots(axisxE, axisyE,
                conv2(elecData-BGele, np.ones([5, 5]) / 25, mode="same"),
                vmin=0,
                XLabel=xlab,
                YLabel="Wavelength (nm)",
                title="Shot : " + str(config["shotnum"]) + " : " + "TS : Corrected and background subtracted")
            
        #down sample image to resolution units by summation
        ang_res_unit = 10 #in pixels
        lam_res_unit = 5 #in pixels
        
        data_res_unit = np.array([np.average(elecData[i:i+lam_res_unit,:], axis=0) for i in range(0, elecData.shape[0], lam_res_unit)])
        #print("data shape after 1 resize", np.shape(data_res_unit))
        data_res_unit = np.array([np.average(data_res_unit[:,i:i+ang_res_unit], axis=1) for i in range(0, data_res_unit.shape[1], ang_res_unit)])
        #print("data shape after 2 resize", np.shape(data_res_unit))
        
        all_data = data_res_unit
        config["D"]["PhysParams"]["noiseI"] = 0
        config["D"]["PhysParams"]["noiseE"] = BGele
        
    else: 
        all_data = get_lineouts(elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, 0, IAWtime, xlab, sa, config)

    config["D"]["PhysParams"]["widIRF"] = stddev
    config["D"]["lamrangE"] = [axisyE[0], axisyE[-1]]
    config["D"]["lamrangI"] = [axisyI[0], axisyI[-1]]
    config["D"]["npts"] = config["D"]["CCDsize"][0] * config["D"]["points_per_pixel"]

    parameters = config["parameters"]

    # Setup x0
    xie = np.linspace(-7, 7, parameters["fe"]["length"])

    # Initialize fe
    NumDistFunc = get_num_dist_func(parameters["fe"]["type"], xie)
    parameters["fe"]["val"] = np.log(NumDistFunc(parameters["m"]["val"]))
    parameters["fe"]["lb"] = np.multiply(parameters["fe"]["lb"], np.ones(parameters["fe"]["length"]))
    parameters["fe"]["ub"] = np.multiply(parameters["fe"]["ub"], np.ones(parameters["fe"]["length"]))

    #[x0, bnds, norms, shifts] = init_params(config, parameters)
    fitting_params = initialize_parameters(config)
    
    
    #loss_fn, vg_loss_fn, hess_fn = get_loss_function(config, xie, sa, all_data, norms, shifts)
    loss_fn, vg_loss_fn, hess_fn = get_loss_function(
        config, xie, sa, all_data, fitting_params["norms"], fitting_params["shifts"], backend="haiku"
    )
    x0 = fitting_params["array"]["init_params"]
    bnds = list(zip(fitting_params["array"]["lb"], fitting_params["array"]["ub"]))

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
                bounds=bnds,
                options={"disp": True},
            )
    else:
        x = fitting_params["pytree"]["init_params"]

    mlflow.log_metrics({"fit_time": round(time.time() - t1, 2)})

    i = 0
    final_x = []
    init_x = []
    for k, v in fitting_params["pytree"]["init_params"].items():
        init_x.append((x0[i] * fitting_params["norms"][k] + fitting_params["shifts"][k]).reshape((len(all_data), -1)))
        final_x.append((res.x[i] * fitting_params["norms"][k] + fitting_params["shifts"][k]).reshape((len(all_data), -1)))
        i += 1
    
    final_x = np.concatenate(final_x, axis=-1)
    init_x = np.concatenate(init_x, axis=-1)
    
    count = 0
    #final_x.reshape((len(config["lineoutloc"]["val"]), -1))

    outputs = {}
    inits = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            # config["parameters"][key]["val"] = [float(val) for val in list(final_x[:, count])]
            outputs[key] = np.array([float(val) for val in list(final_x[:, count])])
            inits[key] = np.array([float(val) for val in list(init_x[:, count])])
            count = count + 1
    
    
    fit_model = get_fit_model(config, xie, sa)
    
    #init_x = (fitting_params["pytree"]["init_params"] * fitting_params["norms"] + fitting_params["shifts"]).reshape((len(all_data), -1))
    #final_x = (res.x * fitting_params["norms"] + fitting_params["shifts"]).reshape((len(all_data), -1))

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
                inits,
                config,
                config["D"]["PhysParams"]["amps"][i][0],
                xie,
                cur_sa,
                all_data[i],
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
                outputs,
                config,
                config["D"]["PhysParams"]["amps"][i][0],
                xie,
                cur_sa,
                all_data[i],
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
    #final_x.reshape((len(config["lineoutloc"]["val"]), -1))

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

    return outputs