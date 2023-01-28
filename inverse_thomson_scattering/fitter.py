import tempfile, os, time, yaml
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas
import scipy.optimize as spopt
import optax
from jaxopt import OptaxSolver


import mlflow

from tqdm import trange
from scipy.signal import convolve2d as conv2
from inverse_thomson_scattering.misc.load_ts_data import load_data
from inverse_thomson_scattering.process.correct_throughput import correct_throughput
from inverse_thomson_scattering.misc.calibration import get_calibrations
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.loss_function import get_loss_function


def initialize_parameters(config: Dict) -> Dict:
    init_params = {}
    lb = {}
    ub = {}
    parameters = config["parameters"]
    for i, _ in enumerate(config["lineoutloc"]["val"]):
        for key in parameters.keys():
            if parameters[key]["active"]:
                init_params[key] = []
                lb[key] = []
                ub[key] = []
                if np.size(parameters[key]["val"]) > 1:
                    init_params[key].append(parameters[key]["val"][i])
                elif isinstance(parameters[key]["val"], list):
                    init_params[key].append(parameters[key]["val"][0])
                else:
                    init_params[key].append(parameters[key]["val"])
                lb[key].append(parameters[key]["lb"])
                ub[key].append(parameters[key]["ub"])

    init_params = {k: np.array(v) for k, v in init_params.items()}
    lb = {k: np.array(v) for k, v in lb.items()}
    ub = {k: np.array(v) for k, v in ub.items()}

    norms = {}
    shifts = {}
    if config["optimizer"]["x_norm"]:
        for k, v in init_params.items():
            norms[k] = ub[k] - lb[k]
            shifts[k] = lb[k]
    else:
        for k, v in init_params.items():
            norms[k] = np.ones_like(init_params)
            shifts[k] = np.zeros_like(init_params)

    init_params = {k: (v - shifts[k]) / norms[k] for k, v in init_params.items()}
    lower_bound = {k: (v - shifts[k]) / norms[k] for k, v in lb.items()}
    upper_bound = {k: (v - shifts[k]) / norms[k] for k, v in ub.items()}

    init_params_arr = np.array([v for k, v in init_params.items()])
    lb_arr = np.array([v for k, v in lower_bound.items()])
    ub_arr = np.array([v for k, v in upper_bound.items()])

    return {
        "pytree": {"init_params": init_params, "lb": lb, "rb": ub},
        "array": {"init_params": init_params_arr, "lb": lb_arr, "ub": ub_arr},
        "norms": norms,
        "shifts": shifts,
    }


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

    ## Persistents
    # used to prevent reloading and one time analysis (not sure if there is a way to do this in python, omitted for now)
    # persistent prevShot
    ## Hard code toggles, locations and values
    # these should only be changed if something changes in the experimental setup or data is moved around

    tstype = config["other"]["extraoptions"]["spectype"]  # 1 for ARTS, 2 for TRTS, 3 for SRTS

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
    [axisxE, axisxI, axisyE, axisyI, magE, IAWtime, stddev] = get_calibrations(config["shotnum"], tstype, CCDsize)

    # Data loading and corrections
    # Open data stored from the previous run (inactivated for now)
    # if isfield(prevShot, 'config["shotnum"]') & & prevShot.config["shotnum"] == config["shotnum"]
    #    elecData = prevShot.elecData;
    #    ionData = prevShot.ionData;
    #    xlab = prevShot.xlab;
    #    shift_zero = prevShot.shift_zero;

    [elecData, ionData, xlab, shift_zero] = load_data(
        config["shotnum"], shotDay, tstype, magE, config["other"]["extraoptions"]
    )

    # turn off ion or electron fitting if the corresponding spectrum was not loaded
    if not config["other"]["extraoptions"]["load_ion_spec"]:
        config["other"]["extraoptions"]["fit_IAW"] = 0
        print("IAW data not loaded, omitting IAW fit")
    if not config["other"]["extraoptions"]["load_ele_spec"]:
        config["other"]["extraoptions"]["fit_EPWb"] = 0
        config["other"]["extraoptions"]["fit_EPWr"] = 0
        print("EPW data not loaded, omitting EPW fit")

    if config["other"]["extraoptions"]["load_ele_spec"]:
        elecData = correct_throughput(elecData, tstype, axisyE)
    # prevShot.config["shotnum"] = config["shotnum"];
    # prevShot.elecData = elecData;
    # prevShot.ionData = ionData;
    # prevShot.xlab = xlab;
    # prevShot.shift_zero = shift_zero;

    # Background Shot subtraction
    if config["bgshot"]["type"] == "Shot":
        [BGele, BGion, _, _] = load_data(
            config["bgshot"]["val"], shotDay, tstype, magE, config["other"]["extraoptions"]
        )
        if config["other"]["extraoptions"]["load_ion_spec"]:
            ionData_bsub = ionData - conv2(BGion, np.ones([5, 3]) / 15, mode="same")
        if config["other"]["extraoptions"]["load_ele_spec"]:
            BGele = correct_throughput(BGele, tstype, axisyE)
            if tstype == 1:
                elecData_bsub = elecData - bgshotmult * conv2(BGele, np.ones([5, 5]) / 25, mode="same")
            else:
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

    if config["other"]["extraoptions"]["load_ele_spec"]:
        LineoutTSE = [
            np.mean(elecData_bsub[:, a - config["dpixel"] : a + config["dpixel"]], axis=1) for a in LineoutPixelE
        ]
        LineoutTSE_smooth = [
            np.convolve(LineoutTSE[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]

    if config["other"]["extraoptions"]["load_ion_spec"]:
        LineoutTSI = [
            np.mean(ionData_bsub[:, a - IAWtime - config["dpixel"] : a - IAWtime + config["dpixel"]], axis=1)
            for a in LineoutPixelI
        ]
        LineoutTSI_smooth = [
            np.convolve(LineoutTSI[i], np.ones(span) / span, "same") for i, _ in enumerate(LineoutPixelE)
        ]  # was divided by 10 for some reason (removed 8-9-22)

    if config["bgshot"]["type"] == "Fit":
        if config["other"]["extraoptions"]["load_ele_spec"]:
            if tstype == 1:
                [BGele, _, _, _] = load_data(
                    config["bgshot"]["val"], shotDay, tstype, magE, config["other"]["extraoptions"]
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
                elecData_bsub = elecData - newBG
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

                for i, _ in enumerate(config["lineoutloc"]["val"]):
                    [rat1bg, _] = spopt.curve_fit(rat11, bgfitx, LineoutTSE_smooth[i][bgfitx], [-16, 200000, 170])
                    # plt.plot(rat11(np.arange(1024), *rat1bg))
                    # plt.plot(LineoutTSE_smooth[i])
                    # plt.show()
                    LineoutTSE_smooth[i] = LineoutTSE_smooth[i] - rat11(np.arange(1024), *rat1bg)

    # Attempt to quantify any residual background
    # this has been switched from mean of elecData to mean of elecData_bsub 8-9-22
    if config["other"]["extraoptions"]["load_ion_spec"]:
        noiseI = np.mean(ionData_bsub[:, BackgroundPixel - config["dpixel"] : BackgroundPixel + config["dpixel"]], 1)
        noiseI = np.convolve(noiseI, np.ones(span) / span, "same")
        bgfitx = np.hstack([np.arange(200, 400), np.arange(700, 850)])
        noiseI = np.mean(noiseI[bgfitx])
        noiseI = np.ones(1024) * bgscalingI * noiseI

    if config["other"]["extraoptions"]["load_ele_spec"]:
        noiseE = np.mean(elecData_bsub[:, BackgroundPixel - config["dpixel"] : BackgroundPixel + config["dpixel"]], 1)
        noiseE = np.convolve(noiseE, np.ones(span) / span, "same")
        # print(noiseE)
        def exp2(x, a, b, c, d):
            return a * np.exp(-b * x) + c * np.exp(-d * x)

        bgfitx = np.hstack(
            [np.arange(250, 480), np.arange(540, 900)]
        )  # this is specificaly targeted at streaked data, removes the fiducials at top and bottom and notch filter
        # plt.plot(bgfitx, noiseE[bgfitx])
        # [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[1000, 0.001, 1000, 0.001])
        [expbg, _] = spopt.curve_fit(exp2, bgfitx, noiseE[bgfitx], p0=[200, 0.001, 200, 0.001])
        noiseE = bgscalingE * exp2(np.arange(1024), *expbg)
        # plt.plot(bgfitx, noiseE[bgfitx])
        # plt.plot(bgfitx, exp2(bgfitx, 200, 0.001, 200, 0.001))
        # plt.show()

        # temporary constant addition to the background
        noiseE = noiseE + flatbg

    ## Plot data
    # fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    # if config["other"]["extraoptions"]["load_ion_spec"]:
    #     imI = ax[1].imshow(
    #         conv2(ionData_bsub, np.ones([5, 3]) / 15, mode="same"),
    #         cmap,
    #         interpolation="none",
    #         extent=[axisxI[0] - shift_zero, axisxI[-1] - shift_zero, axisyI[-1], axisyI[0]],
    #         aspect="auto",
    #         vmin=0,
    #     )
    #     ax[1].set_title(
    #         "Shot : " + str(config["shotnum"]) + " : " + "TS : Thruput corrected",
    #         fontdict={"fontsize": 10, "fontweight": "bold"},
    #     )
    #     ax[1].set_xlabel(xlab)
    #     ax[1].set_ylabel("Wavelength (nm)")
    #     plt.colorbar(imI, ax=ax[1])
    #     ax[1].plot(
    #         [axisxI[LineoutPixelI] - shift_zero, axisxI[LineoutPixelI] - shift_zero], [axisyI[0], axisyI[-1]], "r"
    #     )
    #     ax[1].plot(
    #         [axisxI[BackgroundPixel] - shift_zero, axisxI[BackgroundPixel] - shift_zero], [axisyI[0], axisyI[-1]], "k"
    #     )
    #
    # if config["other"]["extraoptions"]["load_ele_spec"]:
    #     imE = ax[0].imshow(
    #         conv2(elecData_bsub, np.ones([5, 3]) / 15, mode="same"),
    #         cmap,
    #         interpolation="none",
    #         extent=[axisxE[0] - shift_zero, axisxE[-1] - shift_zero, axisyE[-1], axisyE[0]],
    #         aspect="auto",
    #         vmin=0,
    #     )
    #     ax[0].set_title(
    #         "Shot : " + str(config["shotnum"]) + " : " + "TS : Thruput corrected",
    #         fontdict={"fontsize": 10, "fontweight": "bold"},
    #     )
    #     ax[0].set_xlabel(xlab)
    #     ax[0].set_ylabel("Wavelength (nm)")
    #     plt.colorbar(imE, ax=ax[0])
    #     ax[0].plot(
    #         [axisxE[LineoutPixelE] - shift_zero, axisxE[LineoutPixelE] - shift_zero], [axisyE[0], axisyE[-1]], "r"
    #     )
    #     ax[0].plot(
    #         [axisxE[BackgroundPixel] - shift_zero, axisxE[BackgroundPixel] - shift_zero], [axisyE[0], axisyE[-1]], "k"
    #     )

    # Normalize Data before fitting
    if config["other"]["extraoptions"]["load_ion_spec"]:
        noiseI = noiseI / gain
        LineoutTSI_norm = [LineoutTSI_smooth[i] / gain for i, _ in enumerate(LineoutPixelI)]
        LineoutTSI_norm = LineoutTSI_norm - noiseI  # new 6-29-20
        ampI = np.amax(LineoutTSI_norm, axis=1)
    else:
        ampI = 1

    if config["other"]["extraoptions"]["load_ele_spec"]:
        noiseE = noiseE / gain
        LineoutTSE_norm = [LineoutTSE_smooth[i] / gain for i, _ in enumerate(LineoutPixelE)]
        LineoutTSE_norm = LineoutTSE_norm - noiseE  # new 6-29-20
        ampE = np.amax(LineoutTSE_norm[:, 100:-1], axis=1)  # attempts to ignore 3w contamination
    else:
        ampE = 1

    config["other"]["PhysParams"]["widIRF"] = stddev
    config["other"]["lamrangE"] = [axisyE[0], axisyE[-2]]
    config["other"]["lamrangI"] = [axisyI[0], axisyI[-2]]
    config["other"]["npts"] = (len(LineoutTSE_norm) - 1) * 20

    parameters = config["parameters"]

    # Setup x0
    xie = np.linspace(-7, 7, parameters["fe"]["length"])

    NumDistFunc = get_num_dist_func(parameters["fe"]["type"], xie)
    parameters["fe"]["val"] = np.log(NumDistFunc(parameters["m"]["val"]))
    parameters["fe"]["lb"] = np.multiply(parameters["fe"]["lb"], np.ones(parameters["fe"]["length"]))
    parameters["fe"]["ub"] = np.multiply(parameters["fe"]["ub"], np.ones(parameters["fe"]["length"]))

    units = initialize_parameters(config)

    all_data = []
    amps_list = []
    # run fitting code for each lineout
    for i, _ in enumerate(config["lineoutloc"]["val"]):
        # this probably needs to be done differently
        if config["other"]["extraoptions"]["load_ion_spec"] and config["other"]["extraoptions"]["load_ele_spec"]:
            data = np.vstack((LineoutTSE_norm[i], LineoutTSI_norm[i]))
            amps = [ampE[i], ampI[i]]
        elif config["other"]["extraoptions"]["load_ion_spec"]:
            data = np.vstack((LineoutTSI_norm[i], LineoutTSI_norm[i]))
            amps = [ampE, ampI[i]]
        elif config["other"]["extraoptions"]["load_ele_spec"]:
            data = np.vstack((LineoutTSE_norm[i], LineoutTSE_norm[i]))
            amps = [ampE[i], ampI]
        else:
            raise NotImplementedError("This spectrum does not exist")

        # if data.shape
        all_data.append(data[None, :])
        amps_list.append(np.array(amps)[None, :])

    all_data = np.concatenate(all_data)
    amps_list = np.concatenate(amps_list)
    test_batch = {
        "data": all_data[: config["optimizer"]["batch_size"]],
        "amps": amps_list[: config["optimizer"]["batch_size"]],
    }

    vg_loss_fn, array_loss_fn, init_weights, get_params = get_loss_function(
        config, xie, sa, test_batch, units["norms"], units["shifts"]
    )

    # opt_init, opt_update = optax.chain(
    #     # Set the parameters of Adam. Note the learning_rate is not here.
    #     optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    #     # Put a minus sign to *minimise* the loss.
    #     optax.scale(-config["optimizer"]["learning_rate"]),
    # )
    # opt_state = opt_init(weights)

    if config["optimizer"]["method"] == "adam":
        opt = optax.adam(config["optimizer"]["learning_rate"])
        solver = OptaxSolver(
            opt=opt, fun=vg_loss_fn, maxiter=config["optimizer"]["num_epochs"], value_and_grad=True, has_aux=True
        )
    else:
        raise NotImplementedError

    batch_indices = np.arange(len(all_data))
    weights = init_weights
    opt_state = solver.init_state(weights, batch=test_batch)

    t1 = time.time()
    print("minimizing")
    mlflow.set_tag("status", "minimizing")

    epoch_loss = 1e19
    best_loss = 1e16
    for i_epoch in range(config["optimizer"]["num_epochs"]):
        num_batches = len(batch_indices) // config["optimizer"]["batch_size"]
        np.random.shuffle(batch_indices)
        batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
        with trange(num_batches, unit="batch") as tbatch:
            tbatch.set_description(f"Epoch {i_epoch+1}, Prev Epoch Loss {round(epoch_loss)}")
            epoch_loss = 0.0
            for i_batch in tbatch:
                inds = batch_indices[i_batch]
                batch = {"data": all_data[inds], "amps": amps_list[inds]}
                weights, opt_state = solver.update(params=weights, state=opt_state, batch=batch)

                epoch_loss += opt_state.value
                tbatch.set_postfix({"Prev Batch Loss": round(opt_state.value)})

            epoch_loss /= num_batches
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_weights = weights

            mlflow.log_metrics({"epoch loss": float(epoch_loss)}, step=i_epoch)
        batch_indices = batch_indices.flatten()

    all_params = {}
    for key in config["parameters"].keys():
        if config["parameters"][key]["active"]:
            all_params[key] = np.empty(0)

    mlflow.log_metrics({"fit time": round(time.time() - t1, 2)})
    final_params = postprocess(config, batch_indices, all_data, all_params, amps_list, best_weights, array_loss_fn)
    return final_params


def postprocess(config, batch_indices, all_data, all_params, amps_list, best_weights, array_loss_fn):
    with tempfile.TemporaryDirectory() as td:
        t1 = time.time()
        batch_indices.sort()
        batch_indices = np.reshape(batch_indices, (-1, config["optimizer"]["batch_size"]))
        losses = np.zeros_like(batch_indices)
        fits = np.zeros((all_data.shape[0], all_data.shape[2]))
        for i_batch, inds in enumerate(batch_indices):
            batch = {"data": all_data[inds], "amps": amps_list[inds]}
            loss, [ThryE, _, params] = array_loss_fn(best_weights, batch)
            losses[i_batch] = np.mean(loss, axis=1)
            fits[inds] = ThryE
            for k in all_params.keys():
                all_params[k] = np.concatenate([all_params[k], np.squeeze(params[k])])

        mlflow.log_metrics({"inference time": round(time.time() - t1, 2)})

        losses = losses.flatten() / np.amax(all_data[:, 0, :], axis=-1)
        loss_inds = losses.argsort()[::-1]

        sorted_losses = losses[loss_inds]
        sorted_fits = fits[loss_inds]
        sorted_data = all_data[loss_inds]

        num_plots = 8 if 8 < len(losses) // 2 else len(losses) // 2

        t1 = time.time()
        os.makedirs(os.path.join(td, "worst"))
        os.makedirs(os.path.join(td, "best"))

        model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds)

        mlflow.log_metrics({"plot time": round(time.time() - t1, 2)})

        all_params["lineout"] = config["lineoutloc"]["val"]
        final_params = pandas.DataFrame(all_params)
        final_params.to_csv(os.path.join(td, "learned_parameters.csv"))
        mlflow.set_tag("status", "done plotting")
        mlflow.log_artifacts(td)

    return final_params


def model_v_actual(sorted_losses, sorted_data, sorted_fits, num_plots, td, config, loss_inds):
    # make plots
    for i in range(num_plots):
        # plot model vs actual
        titlestr = r"|Error|$^2$" + f" = {sorted_losses[i]}, line out # {config['lineoutloc']['val'][loss_inds[i]]}"
        filename = f"loss={round(sorted_losses[i])}-lineout={config['lineoutloc']['val'][loss_inds[i]]}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        ax.plot(np.squeeze(sorted_data[i, 0, 256:-256]), label="Data")
        ax.plot(np.squeeze(sorted_fits[i, 256:-256]), label="Fit")
        ax.set_title(titlestr, fontsize=14)
        ax.legend(fontsize=14)
        ax.grid()
        fig.savefig(os.path.join(td, "worst", filename), bbox_inches="tight")
        plt.close(fig)

        titlestr = (
            r"|Error|$^2$" + f" = {sorted_losses[-1 - i]}, line out # {config['lineoutloc']['val'][loss_inds[-1 - i]]}"
        )
        filename = f"loss={round(sorted_losses[-1 - i])}-lineout={config['lineoutloc']['val'][loss_inds[-1 - i]]}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
        ax.plot(np.squeeze(sorted_data[-1 - i, 0, 256:-256]), label="Data")
        ax.plot(np.squeeze(sorted_fits[-1 - i, 256:-256]), label="Fit")
        ax.set_title(titlestr, fontsize=14)
        ax.legend(fontsize=14)
        ax.grid()
        fig.savefig(os.path.join(td, "best", filename), bbox_inches="tight")
        plt.close(fig)
