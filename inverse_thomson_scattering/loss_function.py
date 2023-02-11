import copy
from typing import Dict
from functools import partial
from collections import defaultdict
from jax import config

config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp


from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
import haiku as hk
import numpy as np
from inverse_thomson_scattering.generate_spectra import get_fit_model
from inverse_thomson_scattering.process import irf


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

    @jit
    def postprocess_thry(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):
        if config["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = irf.add_ion_IRF(config, lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan

        if config["other"]["extraoptions"]["load_ele_spec"] & (
            config["other"]["extraoptions"]["spectype"] == "angular_full"
        ):
            lamAxisE, ThryE = irf.add_ATS_IRF(config, sas, lamAxisE, modlE, amps, TSins, dummy_batch["data"], lam)
        elif config["other"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = irf.add_electron_IRF(config, lamAxisE, modlE, amps, TSins, lam)
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

    class TSSpectraGenerator(hk.Module):
        def __init__(self, cfg: Dict, num_spectra: int = 2):
            super(TSSpectraGenerator, self).__init__()
            self.cfg = cfg
            self.num_spectra = num_spectra
            self.batch_size = cfg["optimizer"]["batch_size"]
            if cfg["nn"]["use"]:
                self.ts_parameter_generator = TSParameterGenerator(cfg, num_spectra)

            self.crop_window = cfg["other"]["crop_window"]

        def _init_nn_params_(self, batch):
            all_params = self.ts_parameter_generator(batch[:, :, self.crop_window : -self.crop_window])
            these_params = defaultdict(list)
            for i_slice in range(self.batch_size):
                # unpack all params which is an array that came out of the NN and into a dictionary that contains
                # the parameter names
                i = 0
                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name].append(all_params[i_slice, i].reshape((1, 1)))
                        i = i + 1
                    else:
                        these_params[param_name].append(jnp.array(param_config["val"]).reshape((1, -1)))

            for param_name, param_config in self.cfg["parameters"].items():
                these_params[param_name] = jnp.concatenate(these_params[param_name])

            return these_params

        def _init_params_(self):
            these_params = dict()
            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = hk.get_parameter(
                        param_name,
                        shape=[self.batch_size, 1],
                        init=hk.initializers.RandomUniform(minval=0, maxval=1),
                    )
                else:
                    these_params[param_name] = jnp.concatenate(
                        [jnp.array(param_config["val"]).reshape(1, -1) for _ in range(self.batch_size)]
                    ).reshape(self.batch_size, -1)

            return these_params

        def initialize_params(self, batch):
            if self.cfg["nn"]["use"]:
                these_params = self._init_nn_params_(batch)
            else:
                these_params = self._init_params_()

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

    config_for_params = copy.deepcopy(config)

    def __get_params__(batch):
        i_data = batch["data"][:, 1, :]
        e_data = batch["data"][:, 0, :]

        normed_i_data = i_data / i_input_norm
        normed_e_data = e_data / e_input_norm

        normed_batch = jnp.concatenate([normed_e_data[:, None, :], normed_i_data[:, None, :]], axis=1)
        spectrumator = TSSpectraGenerator(config_for_params)
        ThryE, ThryI, lamAxisE, lamAxisI, params = spectrumator(
            {"data": normed_batch, "amps": batch["amps"], "noise_e": batch["noise_e"]}
        )

        return ThryE, ThryI, lamAxisE, lamAxisI, params, i_data, e_data, normed_e_data

    _get_params_ = hk.without_apply_rng(hk.transform(__get_params__))

    def array_loss_fn(weights, batch):
        ThryE, ThryI, lamAxisE, lamAxisI, params, i_data, e_data, normed_e_data = _get_params_.apply(weights, batch)

        i_error = 0.0
        e_error = 0.0

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

    def loss_fn(weights, batch):
        ThryE, ThryI, lamAxisE, lamAxisI, params, i_data, e_data, normed_e_data = _get_params_.apply(weights, batch)
        i_error = 0.0
        e_error = 0.0

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

    rng_key = jax.random.PRNGKey(42)
    init_weights = _get_params_.init(rng_key, dummy_batch)
    _v_func_ = jit(loss_fn)
    _vg_func_ = jit(value_and_grad(loss_fn, argnums=0, has_aux=True))

    if config["optimizer"]["method"] == "adam":
        vg_func = _vg_func_
        get_params = _get_params_
        loss_dict = dict(vg_func=vg_func, array_loss_fn=array_loss_fn, init_weights=init_weights, get_params=get_params)
    elif config["optimizer"]["method"] == "l-bfgs-b":
        if config["nn"]["use"]:
            init_weights = init_weights
            flattened_weights, unravel_pytree = ravel_pytree(init_weights)
            bounds = None
        else:
            lb, ub, init_weights = init_weights_and_bounds(
                config, init_weights, num_slices=len(config["data"]["lineouts"]["val"])
            )
            flattened_weights, unravel_pytree = ravel_pytree(init_weights)
            flattened_lb, _ = ravel_pytree(lb)
            flattened_ub, _ = ravel_pytree(ub)
            bounds = zip(flattened_lb, flattened_ub)

        def vg_func(weights: np.ndarray, batch):
            """
            Full batch training so dummy batch actually contains the whole batch

            Args:
                weights:
                batch:

            Returns:

            """

            pytree_weights = unravel_pytree(weights)
            (value, aux), grad = _vg_func_(pytree_weights, batch)
            temp_grad, _ = ravel_pytree(grad)
            flattened_grads = np.array(temp_grad)  # .flatten()
            return value, flattened_grads

        def v_func(weights: np.ndarray, batch):
            """
            Full batch training so dummy batch actually contains the whole batch

            Args:
                weights:
                batch:

            Returns:

            """

            pytree_weights = unravel_pytree(weights)
            value, _ = _v_func_(pytree_weights, batch)
            return value

        def get_params(weights, batch):
            pytree_weights = unravel_pytree(weights)
            _, _, _, _, params, _, _, _ = _get_params_.apply(pytree_weights, batch)

            return params

        loss_dict = dict(
            vg_func=vg_func,
            array_loss_fn=array_loss_fn,
            pytree_weights=init_weights,
            init_weights=flattened_weights,
            get_params=get_params,
            bounds=bounds,
            unravel_pytree=unravel_pytree,
            v_func=v_func,
        )

    else:
        raise NotImplementedError

    return loss_dict


def init_weights_and_bounds(config, init_weights, num_slices):
    """
    this dict form will be unpacked for scipy consumption, we assemble them all in the same way so that we can then
    use ravel pytree from JAX utilities to unpack it
    Args:
        config:
        init_weights:
        num_slices:

    Returns:

    """
    lb = {"ts_spectra_generator": {}}
    ub = {"ts_spectra_generator": {}}
    iw = {"ts_spectra_generator": {}}
    for k, v in init_weights["ts_spectra_generator"].items():
        lb["ts_spectra_generator"][k] = np.array([0 * config["units"]["lb"][k] for _ in range(num_slices)])
        ub["ts_spectra_generator"][k] = np.array([1.0 + 0 * config["units"]["ub"][k] for _ in range(num_slices)])
        iw["ts_spectra_generator"][k] = np.array([config["parameters"][k]["val"] for _ in range(num_slices)])[:, None]
        iw["ts_spectra_generator"][k] = (iw["ts_spectra_generator"][k] - config["units"]["shifts"][k]) / config[
            "units"
        ]["norms"][k]

    return lb, ub, iw
