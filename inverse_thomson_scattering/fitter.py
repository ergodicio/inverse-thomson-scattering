from typing import Dict
import time, mlflow, os, tempfile, yaml

import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt

import optax, jaxopt, pandas, mlflow
from tqdm import trange
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc.additional_functions import (
    get_scattering_angles,
    plotinput,
)  # , initialize_parameters
from inverse_thomson_scattering.evaluate_background import get_shot_bg
from inverse_thomson_scattering.misc.load_ts_data import loadData
from inverse_thomson_scattering.process.correct_throughput import correctThroughput
from inverse_thomson_scattering.misc.calibration import get_calibrations
from inverse_thomson_scattering.lineouts import get_lineouts
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function
from inverse_thomson_scattering.generate_spectra import get_fit_model
from inverse_thomson_scattering.misc.plotters import plotState


def initialize_parameters(config: Dict) -> Dict:
    # init_params = {}
    lb = {}
    ub = {}
    parameters = config["parameters"]
    active_params = []
    for key in parameters.keys():
        if parameters[key]["active"]:
            active_params.append(key)
            # init_params[key] = []
            lb[key] = []
            ub[key] = []
            lb[key].append(parameters[key]["lb"])
            ub[key].append(parameters[key]["ub"])
            # for i, _ in enumerate(config["data"]["lineouts"]["val"]):
            #     if np.size(parameters[key]["val"]) > 1:
            #         init_params[key].append(parameters[key]["val"][i])
            #     elif isinstance(parameters[key]["val"], list):
            #         init_params[key].append(parameters[key]["val"][0])
            #     else:
            #         init_params[key].append(parameters[key]["val"])

    # init_params = {k: np.array(v) for k, v in init_params.items()}
    lb = {k: np.array(v) for k, v in lb.items()}
    ub = {k: np.array(v) for k, v in ub.items()}

    norms = {}
    shifts = {}
    if config["optimizer"]["parameter_norm"]:
        for k in active_params:
            norms[k] = ub[k] - lb[k]
            shifts[k] = lb[k]
    else:
        for k, v in active_params:
            norms[k] = np.ones_like(lb[k])
            shifts[k] = np.zeros_like(lb[k])

    # init_params = {k: (v - shifts[k]) / norms[k] for k, v in init_params.items()}
    lower_bound = {k: (v - shifts[k]) / norms[k] for k, v in lb.items()}
    upper_bound = {k: (v - shifts[k]) / norms[k] for k, v in ub.items()}

    # init_params_arr = np.array([v for k, v in init_params.items()])
    lb_arr = np.array([v for k, v in lower_bound.items()])
    ub_arr = np.array([v for k, v in upper_bound.items()])

    return {
        "pytree": {"lb": lb, "rb": ub},
        # "array": {"lb": lb_arr, "ub": ub_arr},
        "norms": norms,
        "shifts": shifts,
    }


def validate_inputs(config):
    # get derived quantities
    config["velocity"] = np.linspace(-7, 7, config["parameters"]["fe"]["length"])

    # get slices
    config["data"]["lineouts"]["val"] = [
        i
        for i in range(
            config["data"]["lineouts"]["start"], config["data"]["lineouts"]["end"], config["data"]["lineouts"]["skip"]
        )
    ]

    # create fes
    NumDistFunc = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])
    config["parameters"]["fe"]["val"] = np.log(NumDistFunc(config["parameters"]["m"]["val"]))
    config["parameters"]["fe"]["lb"] = np.multiply(
        config["parameters"]["fe"]["lb"], np.ones(config["parameters"]["fe"]["length"])
    )
    config["parameters"]["fe"]["ub"] = np.multiply(
        config["parameters"]["fe"]["ub"], np.ones(config["parameters"]["fe"]["length"])
    )

    if config["nn"]["use"]:
        if not len(config["data"]["lineouts"]["val"]) % config["optimizer"]["batch_size"] == 0:
            num_slices = len(config["data"]["lineouts"]["val"])
            batch_size = config["optimizer"]["batch_size"]
            print(f"total slices: {num_slices}")
            print(f"{batch_size=}")
            print(f"batch size = {config['optimizer']['batch_size']} is not a round divisor of the number of lineouts")
            num_batches = np.ceil(len(config["data"]["lineouts"]["val"]) / config["optimizer"]["batch_size"])
            config["optimizer"]["batch_size"] = int(len(config["data"]["lineouts"]["val"]) // num_batches)
            print(f"new batch size = {config['optimizer']['batch_size']}")
    else:
        if len(config["data"]["lineouts"]["val"]) != config["optimizer"]["batch_size"]:
            print(f"setting batch size to the number of lineouts")
            config["optimizer"]["batch_size"] = len(config["data"]["lineouts"]["val"])

        for param_name, param_config in config["parameters"].items():
            if param_config["active"]:
                pass
            else:
                param_config["val"] = [
                    np.array(param_config["val"]).reshape(1, -1) for _ in range(config["optimizer"]["batch_size"])
                ]

    config["units"] = initialize_parameters(config)

    return config


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
    config = validate_inputs(config)

    # prepare data
    all_data, sa = prepare_data(config)
    test_batch = {
        "data": all_data["data"][: config["optimizer"]["batch_size"]],
        "amps": all_data["amps"][: config["optimizer"]["batch_size"]],
        "noise_e": config["other"]["PhysParams"]["noiseE"][: config["optimizer"]["batch_size"]],
        # "noise_i": config["other"]["PhysParams"]["noiseI"][: config["optimizer"]["batch_size"]],
    }

    # prepare optimizer / solver
    loss_dict = get_loss_function(config, sa, test_batch)
    batch_indices = np.arange(len(all_data["data"]))

    if config["optimizer"]["method"] == "adam":  # Stochastic Gradient Descent
        jaxopt_kwargs = dict(
            fun=loss_dict["vg_func"], maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
        )
        opt = optax.adam(config["optimizer"]["learning_rate"])
        solver = jaxopt.OptaxSolver(opt=opt, **jaxopt_kwargs)

        weights = loss_dict["init_weights"]
        opt_state = solver.init_state(weights, batch=test_batch)

        # start train loop
        t1 = time.time()
        print("minimizing")
        mlflow.set_tag("status", "minimizing")

        epoch_loss = 1e19
        best_loss = 1e16
        num_batches = len(batch_indices) // config["optimizer"]["batch_size"]
        for i_epoch in range(config["optimizer"]["num_epochs"]):
            if config["nn"]["use"]:
                np.random.shuffle(batch_indices)
            batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
            with trange(num_batches, unit="batch") as tbatch:
                tbatch.set_description(f"Epoch {i_epoch + 1}, Prev Epoch Loss {epoch_loss:.2e}")
                epoch_loss = 0.0
                for i_batch in tbatch:
                    inds = batch_indices[i_batch]
                    batch = {
                        "data": all_data["data"][inds],
                        "amps": all_data["amps"][inds],
                        "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
                    }
                    weights, opt_state = solver.update(params=weights, state=opt_state, batch=batch)
                    epoch_loss += opt_state.value
                    tbatch.set_postfix({"Prev Batch Loss": opt_state.value})

                epoch_loss /= num_batches
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_weights = weights

                mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)
            batch_indices = batch_indices.flatten()

    elif config["optimizer"]["method"] == "l-bfgs-b":
        lb = config["units"]["array"]["lb"]
        ub = config["units"]["array"]["ub"]
        # bounds =

        bounds = zip(lb, ub)
        res = spopt.minimize(
            loss_dict["vg_func"] if config["optimizer"]["grad_method"] == "AD" else loss_dict["v_func"],
            init_array,
            method=config["optimizer"]["method"],
            jac=True if config["optimizer"]["grad_method"] == "AD" else False,
            # hess=hess_fn if config["optimizer"]["hessian"] else None,
            bounds=bounds,
            options={"disp": True},
        )

    else:
        raise NotImplementedError

    mlflow.log_metrics({"fit time": round(time.time() - t1, 2)})
    final_params = postprocess(config, batch_indices, all_data, best_weights, loss_dict["array_loss_fn"])
    return final_params


def postprocess(config, batch_indices, all_data: Dict, best_weights, array_loss_fn):
    all_params = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            all_params[key] = np.empty(0)

    with tempfile.TemporaryDirectory() as td:
        t1 = time.time()
        batch_indices.sort()
        batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
        losses = np.zeros_like(batch_indices, dtype=np.float64)
        fits = np.zeros((all_data["data"].shape[0], all_data["data"].shape[2]))
        for i_batch, inds in enumerate(batch_indices):
            batch = {
                "data": all_data["data"][inds],
                "amps": all_data["amps"][inds],
                "noise_e": config["other"]["PhysParams"]["noiseE"][inds],
            }
            loss, [ThryE, _, params] = array_loss_fn(best_weights, batch)
            losses[i_batch] = np.mean(loss, axis=1)
            fits[inds] = ThryE
            for k in all_params.keys():
                all_params[k] = np.concatenate([all_params[k], np.squeeze(params[k])])

        mlflow.log_metrics({"inference time": round(time.time() - t1, 2)})

        losses = losses.flatten() / np.amax(all_data["data"][:, 0, :], axis=-1)
        loss_inds = losses.argsort()[::-1]

        sorted_losses = losses[loss_inds]
        sorted_fits = fits[loss_inds]
        sorted_data = all_data["data"][loss_inds]

        num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

        t1 = time.time()
        os.makedirs(os.path.join(td, "worst"))
        os.makedirs(os.path.join(td, "best"))

        model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds)

        mlflow.log_metrics({"plot time": round(time.time() - t1, 2)})

        all_params["lineout"] = config["data"]["lineouts"]["val"]
        final_params = pandas.DataFrame(all_params)
        final_params.to_csv(os.path.join(td, "learned_parameters.csv"))
        mlflow.set_tag("status", "done plotting")
        mlflow.log_artifacts(td)

    return final_params


def model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds):
    # make plots
    for i in range(num_plots):
        # plot model vs actual
        titlestr = (
            r"|Error|$^2$" + f" = {sorted_losses[i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[i]]}"
        )
        filename = f"loss={sorted_losses[i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[i]]}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        ax.plot(np.squeeze(sorted_data[i, 0, 256:-256]), label="Data")
        ax.plot(np.squeeze(sorted_fits[i, 256:-256]), label="Fit")
        ax.set_title(titlestr, fontsize=14)
        ax.legend(fontsize=14)
        ax.grid()
        fig.savefig(os.path.join(td, "worst", filename), bbox_inches="tight")
        plt.close(fig)

        titlestr = (
            r"|Error|$^2$"
            + f" = {sorted_losses[-1 - i]:.2e}, line out # {config['data']['lineouts']['val'][loss_inds[-1 - i]]}"
        )
        filename = (
            f"loss={sorted_losses[-1 - i]:.2e}-lineout={config['data']['lineouts']['val'][loss_inds[-1 - i]]}.png"
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        ax.plot(np.squeeze(sorted_data[-1 - i, 0, 256:-256]), label="Data")
        ax.plot(np.squeeze(sorted_fits[-1 - i, 256:-256]), label="Fit")
        ax.set_title(titlestr, fontsize=14)
        ax.legend(fontsize=14)
        ax.grid()
        fig.savefig(os.path.join(td, "best", filename), bbox_inches="tight")
        plt.close(fig)


def prepare_data(config: Dict) -> Dict:
    """
    Loads and preprocesses the data for fitting

    Args:
        config:

    Returns:

    """
    # load data
    [elecData, ionData, xlab, config["other"]["extraoptions"]["spectype"]] = loadData(
        config["data"]["shotnum"], config["data"]["shotDay"], config["other"]["extraoptions"]
    )

    # get scattering angles and weights
    sa = get_scattering_angles(config["other"]["extraoptions"]["spectype"])

    # Calibrate axes
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = get_calibrations(
        config["data"]["shotnum"], config["other"]["extraoptions"]["spectype"], config["other"]["CCDsize"]
    )

    # turn off ion or electron fitting if the corresponding spectrum was not loaded
    if not config["other"]["extraoptions"]["load_ion_spec"]:
        config["other"]["extraoptions"]["fit_IAW"] = 0
        print("IAW data not loaded, omitting IAW fit")
    if not config["other"]["extraoptions"]["load_ele_spec"]:
        config["other"]["extraoptions"]["fit_EPWb"] = 0
        config["other"]["extraoptions"]["fit_EPWr"] = 0
        print("EPW data not loaded, omitting EPW fit")

    # Correct for spectral throughput
    if config["other"]["extraoptions"]["load_ele_spec"]:
        elecData = correctThroughput(
            elecData, config["other"]["extraoptions"]["spectype"], axisyE, config["data"]["shotnum"]
        )

    # load and correct background
    [BGele, BGion] = get_shot_bg(config, axisyE, elecData)

    # extract ARTS section
    if (config["data"]["lineouts"]["type"] == "range") & (config["other"]["extraoptions"]["spectype"] == "angular"):
        config["other"]["extraoptions"]["spectype"] = "angular_full"
        config["other"]["PhysParams"]["amps"] = np.array([np.amax(elecData), 1])
        sa["angAxis"] = axisxE

        if config["other"]["extraoptions"]["plot_raw_data"]:
            ColorPlots(
                axisxE,
                axisyE,
                conv2(elecData - BGele, np.ones([5, 5]) / 25, mode="same"),
                vmin=0,
                XLabel=xlab,
                YLabel="Wavelength (nm)",
                title="Shot : " + str(config["data"]["shotnum"]) + " : " + "TS : Corrected and background subtracted",
            )

        # down sample image to resolution units by summation
        ang_res_unit = 10  # in pixels
        lam_res_unit = 5  # in pixels

        data_res_unit = np.array(
            [np.average(elecData[i : i + lam_res_unit, :], axis=0) for i in range(0, elecData.shape[0], lam_res_unit)]
        )
        # print("data shape after 1 resize", np.shape(data_res_unit))
        data_res_unit = np.array(
            [
                np.average(data_res_unit[:, i : i + ang_res_unit], axis=1)
                for i in range(0, data_res_unit.shape[1], ang_res_unit)
            ]
        )
        # print("data shape after 2 resize", np.shape(data_res_unit))

        all_data = data_res_unit
        config["other"]["PhysParams"]["noiseI"] = 0
        config["other"]["PhysParams"]["noiseE"] = BGele

    else:
        all_data = get_lineouts(
            elecData, ionData, BGele, BGion, axisxE, axisxI, axisyE, axisyI, 0, IAWtime, xlab, sa, config
        )

    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [axisyE[0], axisyE[-1]]
    config["other"]["lamrangI"] = [axisyI[0], axisyI[-1]]
    config["other"]["npts"] = config["other"]["CCDsize"][0] * config["other"]["points_per_pixel"]

    return all_data, sa
