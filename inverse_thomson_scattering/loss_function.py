import copy
from typing import Dict
from functools import partial
from collections import defaultdict

import jax
from jax import numpy as jnp


from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import haiku as hk
import numpy as np
from inverse_thomson_scattering.generate_spectra import get_fit_model


def get_loss_function(config: Dict, sas, dummy_batch: Dict):
    """

    Args:
        config: Dictionary containing all parameter and static values
        sas: dictionary of angles and relative weights
        dummy_batch: Data to be compared against

    Returns:

    """
    forward_pass = get_fit_model(config, sas, backend="haiku")
    lam = config["parameters"]["lam"]["val"]
    vmap = partial(hk.vmap, split_rng=False)

    def add_ion_IRF(lamAxisI, modlI, lamAxisE, amps, TSins):
        stddevI = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["amp3"]["val"] * amps[1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def add_electron_IRF(lamAxisE, modlE, amps, TSins):
        stddevE = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
        # Conceptual_origin so the convolution doesn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
        )  # Gaussian
        ThryE = jnp.convolve(modlE, inst_funcE, "same")
        ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

        if config["other"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = amps[0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)

        return lamAxisE, ThryE

    def add_ATS_IRF(lamAxisE, modlE, amps, TSins):
        stddev_lam = config["other"]["PhysParams"]["widIRF"]["spect_FWHM_ele"] / 2.3548
        stddev_ang = config["other"]["PhysParams"]["widIRF"]["ang_FWHM_ele"] / 2.3548
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
        # apply 2d convolution
        print("modlE shape ", jnp.shape(modlE))
        ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
        print("ThryE shape after conv1 ", jnp.shape(ThryE))
        ThryE = jnp.array([jnp.convolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
        # renorm (not sure why this is needed)
        ThryE = jnp.array(
            [(jnp.amax(modlE[:, i]) / jnp.amax(ThryE[:, i])) * ThryE[:, i] for i in range(modlE.shape[1])]
        )
        ThryE = ThryE.transpose()

        print("ThryE shape after conv2 ", jnp.shape(ThryE))

        if config["other"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        print("ThryE shape after amps", jnp.shape(ThryE))
        lam_step = round(ThryE.shape[1] / data.shape[1])
        ang_step = round(ThryE.shape[0] / data.shape[0])

        ThryE = jnp.array([jnp.average(ThryE[:, i : i + lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
        print("ThryE shape after 1 resize", jnp.shape(ThryE))
        ThryE = jnp.array([jnp.average(ThryE[:, i : i + ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])
        print("ThryE shape after 2 resize", jnp.shape(ThryE))

        # ThryE = ThryE.transpose()
        if config["other"]["PhysParams"]["norm"] == 0:
            # lamAxisE = jnp.average(lamAxisE.reshape(data.shape[0], -1), axis=1)
            lamAxisE = jnp.array(
                [jnp.average(lamAxisE[i : i + lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)]
            )
            ThryE = amps[0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)
        print("ThryE shape after norm ", jnp.shape(ThryE))
        # ThryE = ThryE.transpose()

        return lamAxisE, ThryE

    @jit
    def postprocess_thry(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):
        if config["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = add_ion_IRF(lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan

        if config["other"]["extraoptions"]["load_ele_spec"] & (
            config["other"]["extraoptions"]["spectype"] == "angular_full"
        ):
            lamAxisE, ThryE = add_ATS_IRF(lamAxisE, modlE, amps, TSins)
        elif config["other"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = add_electron_IRF(lamAxisE, modlE, amps, TSins)
        else:
            lamAxisE = jnp.nan
            ThryE = jnp.nan

        return ThryE, ThryI, lamAxisE, lamAxisI

    if config["other"]["extraoptions"]["spectype"] == "angular_full":
        # ATS data can't be vmaped
        vmap_forward_pass = forward_pass
        vmap_postprocess_thry = postprocess_thry
    else:
        vmap_forward_pass = vmap(forward_pass)
        vmap_postprocess_thry = vmap(postprocess_thry)

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(dummy_batch["data"][:, 1, :])
        e_norm = np.amax(dummy_batch["data"][:, 0, :])
    else:
        i_norm = e_norm = 1.0

    if config["optimizer"]["x_norm"] and config["nn"]["use"]:
        i_input_norm = np.amax(dummy_batch["data"][:, 1, :])
        e_input_norm = np.amax(dummy_batch["data"][:, 0, :])
    else:
        i_input_norm = e_input_norm = 1.0

    class TSParameterGenerator(hk.Module):
        def __init__(self, cfg, num_spectra):
            super(TSParameterGenerator, self).__init__()
            self.cfg = cfg
            self.num_spectra = num_spectra
            self.nn = cfg["nn"]
            convs = [int(i) for i in cfg["nn"]["conv_filters"].split("|")]
            linears = [int(i) for i in cfg["nn"]["linear_widths"].split("|")]

            self.param_extractors = []
            for i in range(num_spectra):
                layers = []

                for cc in convs:
                    layers.append(hk.Conv1D(output_channels=cc, kernel_shape=3, stride=1))
                    layers.append(jax.nn.tanh)

                layers.append(hk.Conv1D(1, 3))
                layers.append(jax.nn.tanh)

                layers.append(hk.Flatten())
                for ll in linears:
                    layers.append(hk.Linear(ll))
                    layers.append(jax.nn.tanh)

                self.param_extractors.append(hk.Sequential(layers))

            num_outputs = 0
            for k, v in self.cfg["parameters"].items():
                if v["active"]:
                    num_outputs += 1

            self.combiner = hk.Linear(num_outputs)

        def __call__(self, spectra: jnp.ndarray):
            embeddings = jnp.concatenate(
                [self.param_extractors[i](spectra[:, i][..., None]) for i in range(self.num_spectra)], axis=-1
            )

            return jax.nn.sigmoid(self.combiner(embeddings))

    class TSSpectraGenerator(hk.Module):
        def __init__(self, cfg: Dict, num_spectra: int = 2):
            super(TSSpectraGenerator, self).__init__()
            self.cfg = cfg
            self.num_spectra = num_spectra
            self.batch_size = cfg["optimizer"]["batch_size"]
            if cfg["nn"]["use"]:
                self.ts_parameter_generator = TSParameterGenerator(cfg, num_spectra)

            self.crop_window = cfg["other"]["crop_window"]

        def initialize_params(self, batch):
            if self.cfg["nn"]["use"]:
                all_params = self.ts_parameter_generator(batch[:, :, self.crop_window : -self.crop_window])
                these_params = defaultdict(list)
                for i_slice in range(self.batch_size):
                    i = 0
                    for param_name, param_config in self.cfg["parameters"].items():
                        if param_config["active"]:
                            these_params[param_name].append(all_params[i_slice, i].reshape((1, 1)))
                            i = i + 1
                        else:
                            these_params[param_name].append(jnp.array(param_config["val"]).reshape((1, -1)))

                for param_name, param_config in self.cfg["parameters"].items():
                    these_params[param_name] = jnp.concatenate(these_params[param_name])
            else:
                these_params = defaultdict(list)
                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name] = hk.get_parameter(
                            param_name,
                            shape=[self.batch_size, 1],
                            init=hk.initializers.RandomUniform(minval=0, maxval=1),
                        )
                    else:
                        these_params[param_name] = jnp.concatenate(param_config["val"]).reshape((self.batch_size, -1))

            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = (
                        these_params[param_name] * self.cfg["units"]["norms"][param_name]
                        + self.cfg["units"]["shifts"][param_name]
                    )

            return these_params

        def __call__(self, batch):
            params = self.initialize_params(batch["data"])
            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_forward_pass(params)  # , sas["weights"])
            ThryE, ThryI, lamAxisE, lamAxisI = vmap_postprocess_thry(
                modlE, modlI, lamAxisE, lamAxisI, batch["amps"], live_TSinputs
            )

            ThryE = ThryE + batch["noise_e"]
            # ThryI = ThryI + batch["noise_i"]

            return ThryE, ThryI, lamAxisE, lamAxisI, params

    config_for_loss = copy.deepcopy(config)

    def array_loss_fn(batch):
        i_data = batch["data"][:, 1, :]
        e_data = batch["data"][:, 0, :]

        normed_i_data = i_data / i_input_norm
        normed_e_data = e_data / e_input_norm

        normed_batch = jnp.concatenate([normed_e_data[:, None, :], normed_i_data[:, None, :]], axis=1)

        i_error = 0.0
        e_error = 0.0
        spectrumator = TSSpectraGenerator(config_for_loss)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator(
            {"data": normed_batch, "amps": batch["amps"], "noise_e": batch["noise_e"]}
        )

        if config["other"]["extraoptions"]["fit_IAW"]:
            #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            i_error += jnp.square(i_data - ThryI) / jnp.square(i_norm)

        if config["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = jnp.square(e_data - ThryE) / jnp.square(e_norm)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"]),
                _error_,
                0.0,
            )

            e_error += _error_

        if config["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = jnp.square(e_data - ThryE) / jnp.square(e_norm)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"]),
                _error_,
                0.0,
            )
            e_error += _error_

        return e_error, [ThryE, normed_e_data, params]

    def loss_fn(batch):
        i_data = batch["data"][:, 1, :]
        e_data = batch["data"][:, 0, :]

        normed_i_data = i_data / i_input_norm
        normed_e_data = e_data / e_input_norm

        normed_batch = jnp.concatenate([normed_e_data[:, None, :], normed_i_data[:, None, :]], axis=1)

        i_error = 0.0
        e_error = 0.0
        spectrumator = TSSpectraGenerator(config_for_loss)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator(
            {"data": normed_batch, "amps": batch["amps"], "noise_e": batch["noise_e"]}
        )

        if config["other"]["extraoptions"]["fit_IAW"]:
            i_error += jnp.mean(jnp.square(i_data - ThryI) / jnp.square(i_norm))

        if config["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = jnp.square(e_data - ThryE) / jnp.square(e_norm)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"]),
                _error_,
                0.0,
            )

            e_error += jnp.mean(_error_)

        if config["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = jnp.square(e_data - ThryE) / jnp.square(e_norm)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"]),
                _error_,
                0.0,
            )
            e_error += jnp.mean(_error_)

        return i_error + e_error, [ThryE, normed_e_data, params]

    # def _sp_loss_fn_(batch):
    #
    #     i_data = batch["data"][:, 1, :] / i_norm
    #     e_data = batch["data"][:, 0, :] / e_norm
    #
    #     normed_batch = jnp.concatenate([e_data[:, None, :], i_data[:, None, :]], axis=1)
    #
    #     loss = 0.0
    #     spectrumator = TSSpectraGenerator(config_for_loss)
    #     ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator({"data": normed_batch, "amps": batch["amps"]})
    #
    #     if config["other"]["extraoptions"]["fit_IAW"]:
    #         #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
    #         loss = loss + jnp.square(i_data - ThryI)
    #
    #     if config["other"]["extraoptions"]["fit_EPWb"]:
    #         # vmin = config["other"]["extraoptions"]["fit_EPWb"]["min"]
    #         # vmax = config["other"]["extraoptions"]["fit_EPWb"]["max"]
    #         thry_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), ThryE, 0.0)
    #         data_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), e_data, 0.0)
    #
    #         loss = loss + jnp.square(data_slc - thry_slc)
    #
    #     if config["other"]["extraoptions"]["fit_EPWr"]:
    #         thry_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), ThryE, 0.0)
    #         data_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), e_data, 0.0)
    #
    #         loss = loss + jnp.square(data_slc - thry_slc)
    #
    #     return jnp.mean(loss)

    loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
    array_loss_fn = jit(hk.without_apply_rng(hk.transform(array_loss_fn)).apply)

    rng_key = jax.random.PRNGKey(42)
    init_weights = loss_fn.init(rng_key, dummy_batch)

    _vg_func_ = jit(value_and_grad(loss_fn.apply, has_aux=True))
    config_for_params = copy.deepcopy(config)

    def __get_params__(batch):
        i_data = batch["data"][:, 1, :]
        e_data = batch["data"][:, 0, :]

        normed_i_data = i_data / i_input_norm
        normed_e_data = e_data / e_input_norm

        normed_batch = jnp.concatenate([normed_e_data[:, None, :], normed_i_data[:, None, :]], axis=1)
        spectrumator = TSSpectraGenerator(config_for_params)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator({"data": normed_batch, "amps": batch["amps"]})

        return params, ThryE, e_data

    _get_params_ = hk.without_apply_rng(hk.transform(__get_params__)).apply

    if config["optimizer"]["method"] == "adam":
        vg_func = _vg_func_
        get_params = _get_params_
        loss_dict = dict(vg_func=vg_func, array_loss_fn=array_loss_fn, init_weights=init_weights, get_params=get_params)
    elif config["optimizer"]["method"] == "l-bfgs-b":
        flattened_weights, unravel_pytree = ravel_pytree(init_weights)

        def vg_func(weights: np.ndarray):
            """
            Full batch training so dummy batch actually contains the whole batch

            Args:
                weights:

            Returns:

            """

            pytree_weights = unravel_pytree(weights)
            (value, aux), grad = vg_func(pytree_weights, dummy_batch)
            temp_grad, _ = ravel_pytree(grad)
            flattened_grads = np.array(temp_grad).flatten()
            return value, flattened_grads

        def get_params(weights, batch=None):
            pytree_weights = unravel_pytree(weights)
            params = _get_params_(pytree_weights, dummy_batch)

            return params

        loss_dict = dict(
            vg_func=vg_func,
            array_loss_fn=array_loss_fn,
            pytree_weights=init_weights,
            init_weights=flattened_weights,
            get_params=get_params,
        )

    else:
        raise NotImplementedError

    return loss_dict
