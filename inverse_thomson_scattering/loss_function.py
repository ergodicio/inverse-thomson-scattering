import copy
from typing import Dict
from functools import partial
from collections import defaultdict

import jax
from jax import numpy as jnp

from jax import jit, value_and_grad
import haiku as hk
import numpy as np
from inverse_thomson_scattering.generate_spectra import get_forward_pass


def get_loss_function(config: Dict, xie, sas, dummy_batch: Dict, norms: Dict, shifts: Dict):
    """

    Args:
        config:
        xie:
        sas:
        dummy_batch:
        norms:
        shifts:

    Returns:

    """
    forward_pass = get_forward_pass(config, xie, sas, backend="haiku")
    lam = config["parameters"]["lam"]["val"]
    stddev = config["other"]["PhysParams"]["widIRF"]

    vmap = partial(hk.vmap, split_rng=False)

    def transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins):
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddev[1] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddev[1]) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if config["other"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["amp3"]["val"] * amps[1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def transform_electron_spec(lamAxisE, modlE, amps, TSins):
        # Conceptual_origin so the convolution doesn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddev[0] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddev[0]) ** 2.0))
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

    @jit
    def postprocess(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):

        if config["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan
            # raise NotImplementedError("Need to create an ion spectrum so we can compare it against data!")

        if config["other"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = transform_electron_spec(lamAxisE, modlE, amps, TSins)
        else:
            raise NotImplementedError("Need to create an electron spectrum so we can compare it against data!")

        return ThryE, ThryI, lamAxisE, lamAxisI

    vmap_forward_pass = vmap(forward_pass)
    vmap_postprocess = vmap(postprocess)

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(dummy_batch["data"][:, 1, :])
        e_norm = np.amax(dummy_batch["data"][:, 0, :])
    else:
        i_norm = e_norm = 1.0

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
                        these_params[param_name] = jnp.array(param_config["val"]).reshape((self.batch_size, -1))

            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = these_params[param_name] * norms[param_name] + shifts[param_name]

            return these_params

        def __call__(self, batch):

            params = self.initialize_params(batch["data"])
            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_forward_pass(params)
            ThryE, ThryI, lamAxisE, lamAxisI = vmap_postprocess(
                modlE, modlI, lamAxisE, lamAxisI, batch["amps"], live_TSinputs
            )

            ThryE = ThryE / e_norm
            ThryI = ThryI / i_norm

            return ThryE, ThryI, lamAxisE, lamAxisI, params

    config_for_loss = copy.deepcopy(config)

    def array_loss_fn(batch):

        i_data = batch["data"][:, 1, :] / i_norm
        e_data = batch["data"][:, 0, :] / e_norm

        normed_batch = jnp.concatenate([e_data[:, None, :], i_data[:, None, :]], axis=1)

        loss = 0.0
        spectrumator = TSSpectraGenerator(config_for_loss)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator({"data": normed_batch, "amps": batch["amps"]})

        if config["other"]["extraoptions"]["fit_IAW"]:
            #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            loss = loss + jnp.square(i_data - ThryI)

        if config["other"]["extraoptions"]["fit_EPWb"]:
            # vmin = config["other"]["extraoptions"]["fit_EPWb"]["min"]
            # vmax = config["other"]["extraoptions"]["fit_EPWb"]["max"]
            thry_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), e_data, 0.0)

            loss = loss + jnp.square(data_slc - thry_slc)

        if config["other"]["extraoptions"]["fit_EPWr"]:
            thry_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), e_data, 0.0)

            loss = loss + jnp.square(data_slc - thry_slc)

        return loss, [ThryE, e_data, params]

    def loss_fn(batch):

        i_data = batch["data"][:, 1, :] / i_norm
        e_data = batch["data"][:, 0, :] / e_norm

        normed_batch = jnp.concatenate([e_data[:, None, :], i_data[:, None, :]], axis=1)

        loss = 0.0
        spectrumator = TSSpectraGenerator(config_for_loss)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator({"data": normed_batch, "amps": batch["amps"]})

        if config["other"]["extraoptions"]["fit_IAW"]:
            #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            loss = loss + jnp.square(i_data - ThryI)

        if config["other"]["extraoptions"]["fit_EPWb"]:
            # vmin = config["other"]["extraoptions"]["fit_EPWb"]["min"]
            # vmax = config["other"]["extraoptions"]["fit_EPWb"]["max"]
            thry_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), e_data, 0.0)

            loss = loss + jnp.square(data_slc - thry_slc)

        if config["other"]["extraoptions"]["fit_EPWr"]:
            thry_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), e_data, 0.0)

            loss = loss + jnp.square(data_slc - thry_slc)

        return jnp.mean(loss), [ThryE, e_data, params]

    loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
    array_loss_fn = jit(hk.without_apply_rng(hk.transform(array_loss_fn)).apply)

    rng_key = jax.random.PRNGKey(42)
    init_params = loss_fn.init(rng_key, dummy_batch)

    vg_func = jit(value_and_grad(loss_fn.apply, has_aux=True))
    config_for_params = copy.deepcopy(config)

    def __get_params__(batch):
        i_data = batch["data"][:, 1, :] / i_norm
        e_data = batch["data"][:, 0, :] / e_norm

        normed_batch = jnp.concatenate([e_data[:, None, :], i_data[:, None, :]], axis=1)
        spectrumator = TSSpectraGenerator(config_for_params)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator({"data": normed_batch, "amps": batch["amps"]})

        return params, ThryE, e_data

    _get_params_ = hk.without_apply_rng(hk.transform(__get_params__)).apply

    return vg_func, array_loss_fn, init_params, _get_params_
