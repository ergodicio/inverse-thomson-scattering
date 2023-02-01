from typing import Dict
from functools import partial

import jax
from jax import numpy as jnp

from jax import jit, value_and_grad
import haiku as hk
import numpy as np
from inverse_thomson_scattering.generate_spectra import get_fit_model


def get_loss_function(config: Dict, xie, sas, data: np.ndarray, norms: Dict, shifts: Dict, backend="jax"):
    """

    Args:
        config: Dictionary containing all parameter and static values
        xie: normalized electron velocity
        sas: dictionary of angles and relative weights
        data: Data to be compared against
        norms: noramlization values for fitting parameters, in combination with shifts set all parameter to range[0,1]
        shifts: shift values for parameters, in combination with norms set all parameter to range[0,1]

    Returns:

    """
    forward_pass = get_fit_model(config, xie, sas)
    lam = config["parameters"]["lam"]["val"]

    if backend == "jax":
        vmap = jax.vmap
    else:
        vmap = partial(hk.vmap, split_rng=False)

    def add_ion_IRF(lamAxisI, modlI, lamAxisE, amps, TSins):
        stddevI = config["D"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if config["D"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["amp3"]["val"] * amps[1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def add_electron_IRF(lamAxisE, modlE, amps, TSins):
        stddevE = config["D"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
        # Conceptual_origin so the convolution doesn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
        )  # Gaussian
        ThryE = jnp.convolve(modlE, inst_funcE, "same")
        ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

        if config["D"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
        if config["D"]["PhysParams"]["norm"] == 0:
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = amps[0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)

        return lamAxisE, ThryE

    def add_ATS_IRF(lamAxisE, modlE, amps, TSins):
        stddev_lam = config["D"]["PhysParams"]["widIRF"]["spect_FWHM_ele"] / 2.3548
        stddev_ang = config["D"]["PhysParams"]["widIRF"]["ang_FWHM_ele"] / 2.3548
        # Conceptual_origin so the convolution donsn't shift the signal
        origin_lam = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        origin_ang = (jnp.amax(sas["angAxis"]) + jnp.amin(sas["angAxis"])) / 2.0
        inst_func_lam = jnp.squeeze(
            (1.0 / (stddev_lam * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - origin_lam) ** 2.0) / (2.0 * (stddev_lam) ** 2.0))
        )  # Gaussian
        inst_func_ang = jnp.squeeze(
            (1.0 / (stddev_ang * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((sas["angAxis"] - origin_ang) ** 2.0) / (2.0 * (stddev_ang) ** 2.0))
        )  # Gaussian
        print("inst ang shape ", jnp.shape(inst_func_ang))
        print("inst lam shape ", jnp.shape(inst_func_lam))
        #apply 2d convolution
        print("modlE shape ", jnp.shape(modlE))
        ThryE = jnp.array([jnp.convolve(modlE[:,i], inst_func_ang, "same")
                 for i in range(modlE.shape[1])])
        print("ThryE shape after conv1 ", jnp.shape(ThryE))
        ThryE = jnp.array([jnp.convolve(ThryE[:,i], inst_func_lam, "same")
                 for i in range(ThryE.shape[1])])
        #renorm (not sure why this is needed)
        ThryE = jnp.array([(jnp.amax(modlE[:,i]) / jnp.amax(ThryE[:,i])) * ThryE[:,i] for i in range(modlE.shape[1])])
        ThryE = ThryE.transpose()

        print("ThryE shape after conv2 ", jnp.shape(ThryE))

        if config["D"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        print("ThryE shape after amps", jnp.shape(ThryE))
        lam_step = round(ThryE.shape[1]/data.shape[1])
        ang_step = round(ThryE.shape[0]/data.shape[0])

        ThryE = jnp.array([jnp.average(ThryE[:,i:i+lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
        print("ThryE shape after 1 resize", jnp.shape(ThryE))
        ThryE = jnp.array([jnp.average(ThryE[:,i:i+ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])
        print("ThryE shape after 2 resize", jnp.shape(ThryE))

        #ThryE = ThryE.transpose()
        if config["D"]["PhysParams"]["norm"] == 0:
            #lamAxisE = jnp.average(lamAxisE.reshape(data.shape[0], -1), axis=1)
            lamAxisE = jnp.array([jnp.average(lamAxisE[i:i+lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)])
            ThryE = amps[0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)
        print("ThryE shape after norm ", jnp.shape(ThryE))
        #ThryE = ThryE.transpose()

        return lamAxisE, ThryE

    @jit
    def postprocess_thry(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):

        if config["D"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = add_ion_IRF(lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan

        if config["D"]["extraoptions"]["load_ele_spec"] & (config["D"]["extraoptions"]["spectype"] == "angular_full"):
            lamAxisE, ThryE = add_ATS_IRF(lamAxisE, modlE, amps, TSins)
        elif config["D"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = add_electron_IRF(lamAxisE, modlE, amps, TSins)
        else:
            lamAxisE = jnp.nan
            ThryE = jnp.nan

        return ThryE, ThryI, lamAxisE, lamAxisI

    if config["D"]["extraoptions"]["spectype"] == "angular_full":
        # ATS data can't be vmaped
        vmap_forward_pass = forward_pass
        vmap_postprocess_thry = postprocess_thry
    else:
        vmap_forward_pass = vmap(forward_pass)
        vmap_postprocess_thry = vmap(postprocess_thry)

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(data[:, 1, :])
        e_norm = np.amax(data[:, 0, :])
    else:
        i_norm = e_norm = 1.0

    if backend == "jax":

        def loss_fn(x: jnp.ndarray):
            these_params = {}
            i = 0
            for param_name, param_config in config["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = x[0, i].reshape((1, -1)) * norms[param_name] + shifts[param_name]
                    i += 1
                else:
                    these_params[param_name] = jnp.array(param_config["val"]).reshape((1, -1))

            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_forward_pass(these_params, sas["weights"])
            ThryE, ThryI, lamAxisE, lamAxisI = vmap_postprocess_thry(
                modlE, modlI, lamAxisE, lamAxisI, jnp.concatenate(config["D"]["PhysParams"]["amps"]), live_TSinputs
            )

            ThryE = ThryE / e_norm
            ThryI = ThryI / i_norm

            loss = 0
            if config["D"]["extraoptions"]["spectype"] == "angular_full":
                e_data = data
                i_data = 0
            else:
                i_data = data[:, 1, :] / i_norm
                e_data = data[:, 0, :] / e_norm
            if config["D"]["extraoptions"]["fit_IAW"]:
                loss = loss + jnp.sum(jnp.square(i_data - ThryI) /i_data)

            if config["D"]["extraoptions"]["fit_EPWb"]:
                sqdev = jnp.square(e_data - ThryE) / ThryE
                sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["blue_min"])
                                  & (lamAxisE < config["D"]["fit_rng"]["blue_max"]),
                                  sqdev, 0.0)

                loss = loss + jnp.sum(sqdev)

            if config["D"]["extraoptions"]["fit_EPWr"]:
                sqdev = jnp.square(e_data - ThryE) / ThryE
                sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["red_min"])
                                  & (lamAxisE < config["D"]["fit_rng"]["red_max"]),
                                  sqdev, 0.0)

                loss = loss + jnp.sum(sqdev)

            return loss

        vg_func = jit(value_and_grad(loss_fn))
        loss_func = jit(loss_fn)
        hess_func = jit(jax.hessian(loss_fn))

        def val_and_grad_loss(x: np.ndarray):
            reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
            value, grad = vg_func(reshaped_x)
            return value, np.array(grad).flatten()

    else:

        class TSSpectraGenerator(hk.Module):
            def __init__(self, cfg):
                super(TSSpectraGenerator, self).__init__()
                self.cfg = cfg

            def initialize_params(self):
                these_params = {}
                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name] = hk.get_parameter(
                            param_name,
                            shape=[1, 1],
                            init=hk.initializers.RandomUniform(minval=param_config["lb"], maxval=param_config["ub"]),
                        )
                    else:
                        these_params[param_name] = jnp.array(param_config["val"]).reshape((1, -1))

                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name] = these_params[param_name] * norms[param_name] + shifts[param_name]

                return these_params

            def __call__(self, batch):
                params = self.initialize_params()
                # params = self.neural_network_parameterizer(batch)
                modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_forward_pass(params)
                ThryE, ThryI, lamAxisE, lamAxisI = vmap_postprocess_thry(
                    modlE,
                    modlI,
                    lamAxisE,
                    lamAxisI,
                    jnp.concatenate(self.cfg["D"]["PhysParams"]["amps"]),
                    live_TSinputs,
                )

                ThryE = ThryE / e_norm
                ThryI = ThryI / i_norm

                i_data = batch[:, 1, :] / i_norm
                e_data = batch[:, 0, :] / e_norm

                return e_data, i_data, ThryE, ThryI, lamAxisE, lamAxisI

        def loss_fn(batch):
            loss = 0.0
            Spectrumator = TSSpectraGenerator(config)
            e_data, i_data, ThryE, ThryI, lamAxisE, lamAxisI = Spectrumator(batch)

            if config["D"]["extraoptions"]["fit_IAW"]:
                loss = loss + jnp.sum(jnp.square(i_data - ThryI) / i_data)

            if config["D"]["extraoptions"]["fit_EPWb"]:
                sqdev = jnp.square(e_data - ThryE) / ThryE
                sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["blue_min"])
                                  & (lamAxisE < config["D"]["fit_rng"]["blue_max"]),
                                  sqdev, 0.0)

                loss = loss + jnp.sum(sqdev)

            if config["D"]["extraoptions"]["fit_EPWr"]:
                sqdev = jnp.square(e_data - ThryE) / ThryE
                sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["red_min"])
                                  & (lamAxisE < config["D"]["fit_rng"]["red_max"]),
                                  sqdev, 0.0)
                loss = loss + jnp.sum(sqdev)

            return loss

        loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
        vg_func = jit(value_and_grad(loss_fn.apply))
        loss_func = jit(loss_fn.apply)
        hess_func = jit(jax.hessian(loss_fn.apply))

        def val_and_grad_loss(x: np.ndarray):
            pytree_weights = {"ts_spectra_generator": {}}
            i = 0
            for key in config["parameters"].keys():
                if config["parameters"][key]["active"]:
                    pytree_weights["ts_spectra_generator"][key] = jnp.array(x[i].reshape((data.shape[0], -1)))
                    i += 1

            value, grad = vg_func(pytree_weights, data)
            grads = []
            for key in config["parameters"].keys():
                if config["parameters"][key]["active"]:
                    grads.append(grad["ts_spectra_generator"][key])
            return value, np.array(np.concatenate(grads)).flatten()

    def value(x: np.ndarray):
        x = x * norms + shifts
        reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
        val = loss_func(reshaped_x)

        return val

    return value, val_and_grad_loss, hess_func
