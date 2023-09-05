import copy
from typing import Dict
from collections import defaultdict
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)
import jax
from jax import numpy as jnp


from jax import jit, value_and_grad
from jax.lax import scan
from jax.flatten_util import ravel_pytree
import haiku as hk
import numpy as np
from inverse_thomson_scattering import nn
from inverse_thomson_scattering.generate_spectra import get_fit_model
from inverse_thomson_scattering.process import irf


class LossFunction:
    def __init__(self, config, sas, dummy_batch):
        pass


def get_loss_function(config: Dict, sas, dummy_batch: Dict):
    """

    Args:
        config: Dictionary containing all parameter and static values
        sas: dictionary of angles and relative weights
        dummy_batch: Data to be compared against

    Returns:

    """
    forward_pass = get_fit_model(config, sas, backend="jax")
    lam = config["parameters"]["lam"]["val"]
    vmap = jax.vmap

    @jit
    def postprocess_thry(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):
        if config["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, ThryI = irf.add_ion_IRF(config, lamAxisI, modlI, amps["i_amps"], TSins)
        else:
            # lamAxisI = jnp.nan
            ThryI = modlI  # jnp.nan

        if config["other"]["extraoptions"]["load_ele_spec"] & (
            config["other"]["extraoptions"]["spectype"] == "angular_full"
        ):
            lamAxisE, ThryE = irf.add_ATS_IRF(config, sas, lamAxisE, modlE, amps["e_amps"], TSins, lam)
        elif config["other"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = irf.add_electron_IRF(config, lamAxisE, modlE, amps["e_amps"], TSins, lam)
        else:
            # lamAxisE = jnp.nan
            ThryE = modlE  # jnp.nan

        return ThryE, ThryI, lamAxisE, lamAxisI

    if (
        config["other"]["extraoptions"]["spectype"] == "angular_full"
        or max(dummy_batch["e_data"].shape[0], dummy_batch["i_data"].shape[0]) <= 1
    ):
        # ATS data can't be vmaped and single lineouts cant be vmapped
        vmap_forward_pass = forward_pass
        vmap_postprocess_thry = postprocess_thry
    else:
        vmap_forward_pass = vmap(forward_pass)
        vmap_postprocess_thry = vmap(postprocess_thry)

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(dummy_batch["i_data"])
        e_norm = np.amax(dummy_batch["e_data"])
    else:
        i_norm = e_norm = 1.0

    if config["optimizer"]["x_norm"] and config["nn"]["use"]:
        i_input_norm = np.amax(dummy_batch["i_data"])
        e_input_norm = np.amax(dummy_batch["e_data"])
    else:
        i_input_norm = e_input_norm = 1.0

    class TSParameterGenerator(hk.Module):
        def __init__(self, cfg: Dict, num_spectra: int = 2):
            super(TSParameterGenerator, self).__init__()
            self.cfg = cfg
            self.num_spectra = num_spectra
            self.batch_size = cfg["optimizer"]["batch_size"]
            if cfg["nn"]["use"]:
                self.nn_reparameterizer = nn.Reparameterizer(cfg, num_spectra)

            self.crop_window = cfg["other"]["crop_window"]
            self.smooth_window_len = round(round(cfg["velocity"].size / 10) * 1.5)

            if cfg["dist_fit"]["window"]["type"] == "hamming":
                self.w = jnp.hamming(self.smooth_window_len)
            elif cfg["dist_fit"]["window"]["type"] == "hann":
                self.w = jnp.hanning(self.smooth_window_len)
            elif cfg["dist_fit"]["window"]["type"] == "bartlett":
                self.w = jnp.bartlett(self.smooth_window_len)
            else:
                raise NotImplementedError

        def _init_nn_params_(self, batch):
            all_params = self.nn_reparameterizer(batch[:, :, self.crop_window : -self.crop_window])
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
                    if param_name == "fe":
                        these_params[param_name] = hk.get_parameter(
                            param_name,
                            shape=[self.batch_size, param_config["length"]],
                            init=hk.initializers.RandomUniform(minval=0, maxval=1),
                        )
                    else:
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

        def get_active_params(self, batch):
            # print("in get_active_params")
            if self.cfg["nn"]["use"]:
                all_params = self.nn_reparameterizer(batch["data"][:, :, self.crop_window : -self.crop_window])
                these_params = defaultdict(list)
                for i_slice in range(self.batch_size):
                    # unpack all params which is an array that came out of the NN and into a dictionary that contains
                    # the parameter names
                    i = 0
                    for param_name, param_config in self.cfg["parameters"].items():
                        if param_config["active"]:
                            these_params[param_name].append(all_params[i_slice, i].reshape((1, 1)))
                            i = i + 1

                for param_name, param_config in these_params.items():
                    these_params[param_name] = jnp.concatenate(these_params[param_name])

            else:
                these_params = dict()
                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name] = hk.get_parameter(
                            param_name,
                            shape=[self.batch_size, 1],
                            init=hk.initializers.RandomUniform(minval=0, maxval=1),
                        )
                        these_params[param_name] = (
                            these_params[param_name] * self.cfg["units"]["norms"][param_name]
                            + self.cfg["units"]["shifts"][param_name]
                        )

            return these_params

        def smooth(self, distribution):
            s = jnp.r_[
                distribution[self.smooth_window_len - 1 : 0 : -1],
                distribution,
                distribution[-2 : -self.smooth_window_len - 1 : -1],
            ]
            return jnp.convolve(self.w / self.w.sum(), s, mode="same")[
                self.smooth_window_len - 1 : -(self.smooth_window_len - 1)
            ]

        def __call__(self, batch):
            if self.cfg["nn"]["use"]:
                these_params = self._init_nn_params_(batch["data"])
            else:
                these_params = self._init_params_()

            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = (
                        these_params[param_name] * self.cfg["units"]["norms"][param_name]
                        + self.cfg["units"]["shifts"][param_name]
                    )
                    if param_name == "fe":
                        these_params["fe"] = jnp.log(self.smooth(jnp.exp(these_params["fe"][0]))[None, :])

            return these_params

    def initialize_rest_of_params(config):
        # print("in initialize_rest_of_params")
        these_params = dict()
        for param_name, param_config in config["parameters"].items():
            if param_config["active"]:
                pass
            else:
                these_params[param_name] = jnp.concatenate(
                    [jnp.array(param_config["val"]).reshape(1, -1) for _ in range(config["optimizer"]["batch_size"])]
                ).reshape(config["optimizer"]["batch_size"], -1)

        return these_params

    # def get_normed_e_and_i_data(batch):
    #    normed_i_data = batch["i_data"] / i_input_norm
    #    normed_e_data = batch["e_data"] / e_input_norm

    #    return normed_e_data, normed_i_data

    def get_normed_batch(batch):
        normed_batch = copy.deepcopy(batch)
        normed_batch["i_data"] = normed_batch["i_data"] / i_input_norm
        normed_batch["e_data"] = normed_batch["e_data"] / e_input_norm
        # normed_e_data, normed_i_data = get_normed_e_and_i_data(batch)
        # normed_batch = jnp.concatenate([normed_e_data[:, None, :], normed_i_data[:, None, :]], axis=1)
        return normed_batch

    def __get_params__(batch):
        normed_batch = get_normed_batch(batch)
        spectrumator = TSParameterGenerator(config_for_params)
        params = spectrumator(
            {
                "data": normed_batch,
                "amps": {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]},
                "noise_e": batch["noise_e"],
                "noise_i": batch["noise_i"],
            }
        )

        return params

    _get_params_ = hk.without_apply_rng(hk.transform(__get_params__))

    def __get_active_params__(batch):
        normed_batch = get_normed_batch(batch)
        spectrumator = TSParameterGenerator(config_for_params)
        active_params = spectrumator.get_active_params(
            {
                "data": normed_batch,
                "amps": {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]},
                "noise_e": batch["noise_e"],
                "noise_i": batch["noise_i"],
            }
        )

        return active_params

    _get_active_params_ = hk.without_apply_rng(hk.transform(__get_active_params__))

    def get_spectra_from_params(params, batch):
        modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_forward_pass(params)  # , sas["weights"])
        ThryE, ThryI, lamAxisE, lamAxisI = vmap_postprocess_thry(
            modlE, modlI, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, live_TSinputs
        )
        if config["other"]["extraoptions"]["spectype"] == "angular_full":
            ThryE, lamAxisE = reduce_ATS_to_resunit(ThryE, lamAxisE, live_TSinputs, batch)

        ThryE = ThryE + batch["noise_e"]
        ThryI = ThryI + batch["noise_i"]

        return ThryE, ThryI, lamAxisE, lamAxisI

    dv = config["velocity"][1] - config["velocity"][0]

    def calculate_spectra(weights, batch):
        params = _get_params_.apply(weights, batch)
        ThryE, ThryI, lamAxisE, lamAxisI = get_spectra_from_params(params, batch)
        return ThryE, ThryI, lamAxisE, lamAxisI, params

    config_for_params = copy.deepcopy(config)

    def reduce_ATS_to_resunit(ThryE, lamAxisE, TSins, batch):
        lam_step = round(ThryE.shape[1] / batch["e_data"].shape[1])
        ang_step = round(ThryE.shape[0] / config["other"]["CCDsize"][0])

        ThryE = jnp.array([jnp.average(ThryE[:, i : i + lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
        ThryE = jnp.array([jnp.average(ThryE[:, i : i + ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])

        lamAxisE = jnp.array(
            [jnp.average(lamAxisE[i : i + lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)]
        )
        ThryE = ThryE[config["data"]["lineouts"]["start"] : config["data"]["lineouts"]["end"], :]
        ThryE = batch["e_amps"] * ThryE / jnp.amax(ThryE, axis=1, keepdims=True)
        ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)
        return ThryE, lamAxisE

    def array_loss_fn(weights, batch):
        # Used for postprocessing
        ThryE, ThryI, lamAxisE, lamAxisI, params = calculate_spectra(weights, batch)

        used_points = 0
        loss = 0

        i_data = batch["i_data"]
        e_data = batch["e_data"]
        # normed_batch = get_normed_batch(batch)
        # normed_i_data = normed_batch["i_data"]
        # normed_e_data = normed_batch["e_data"]
        sqdev = {"ele": jnp.zeros(e_data.shape), "ion": jnp.zeros(i_data.shape)}

        if config["other"]["extraoptions"]["fit_IAW"]:
            #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            sqdev["ion"] = jnp.square(i_data - ThryI) / (jnp.abs(i_data) + 1e-1)
            sqdev["ion"] = jnp.where(
                (lamAxisI > config["data"]["fit_rng"]["iaw_min"]) & (lamAxisI < config["data"]["fit_rng"]["iaw_max"]),
                sqdev["ion"],
                0.0,
            )
            loss += jnp.sum(sqdev["ion"], axis=1)
            used_points += jnp.sum(
                (lamAxisI > config["data"]["fit_rng"]["iaw_min"]) & (lamAxisI < config["data"]["fit_rng"]["iaw_max"])
            )

        if config["other"]["extraoptions"]["fit_EPWb"]:
            sqdev_e_b = jnp.square(e_data - ThryE) / jnp.abs(e_data)  # jnp.square(e_norm)
            sqdev_e_b = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"]),
                sqdev_e_b,
                0.0,
            )
            # not sure whether this should be lamAxisE[0,:]  or lamAxisE
            used_points += jnp.sum(
                (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"])
            )

            loss += jnp.sum(sqdev_e_b, axis=1)
            sqdev["ele"] += sqdev_e_b

        if config["other"]["extraoptions"]["fit_EPWr"]:
            sqdev_e_r = jnp.square(e_data - ThryE) / jnp.abs(e_data)  # jnp.square(e_norm)
            sqdev_e_r = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"]),
                sqdev_e_r,
                0.0,
            )
            used_points += jnp.sum(
                (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"])
            )

            loss += jnp.sum(sqdev_e_r, axis=1)
            sqdev["ele"] += sqdev_e_r

        loss = loss

        return loss, sqdev, used_points, [ThryE, ThryI, params]

    def loss_for_hess_fn(params, batch):
        params = {**params, **initialize_rest_of_params(config)}
        ThryE, ThryI, lamAxisE, lamAxisI = get_spectra_from_params(params, batch)
        i_error = 0.0
        e_error = 0.0

        i_data = batch["i_data"]
        e_data = batch["e_data"]

        if config["other"]["extraoptions"]["fit_IAW"]:
            _error_ = jnp.square(i_data - ThryI) / (jnp.abs(i_data) + 1e-10)
            _error_ = jnp.where(
                (lamAxisI > config["data"]["fit_rng"]["iaw_min"]) & (lamAxisI < config["data"]["fit_rng"]["iaw_max"]),
                _error_,
                0.0,
            )
            i_error += jnp.sum(_error_)

        if config["other"]["extraoptions"]["fit_EPWb"]:
            _error_ = jnp.square(e_data - ThryE) / (jnp.abs(e_data) + 1e-10)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["blue_min"]) & (lamAxisE < config["data"]["fit_rng"]["blue_max"]),
                _error_,
                0.0,
            )

            e_error += jnp.sum(_error_)

        if config["other"]["extraoptions"]["fit_EPWr"]:
            _error_ = jnp.square(e_data - ThryE) / (jnp.abs(e_data) + 1e-10)
            _error_ = jnp.where(
                (lamAxisE > config["data"]["fit_rng"]["red_min"]) & (lamAxisE < config["data"]["fit_rng"]["red_max"]),
                _error_,
                0.0,
            )
            e_error += jnp.sum(_error_)

        return i_error + e_error

    def loss_fn(weights, batch):
        ThryE, ThryI, lamAxisE, lamAxisI, params = calculate_spectra(weights, batch)
        i_error = 0.0
        e_error = 0.0

        i_data = batch["i_data"]
        e_data = batch["e_data"]
        normed_batch = get_normed_batch(batch)
        normed_i_data = normed_batch["i_data"]
        normed_e_data = normed_batch["e_data"]

        if config["other"]["extraoptions"]["fit_IAW"]:
            # i_error += jnp.mean(jnp.square(i_data - ThryI) / jnp.square(i_norm))
            _error_ = jnp.square(i_data - ThryI) / jnp.square(i_norm)
            _error_ = jnp.where(
                (lamAxisI > config["data"]["fit_rng"]["iaw_min"]) & (lamAxisI < config["data"]["fit_rng"]["iaw_max"]),
                _error_,
                0.0,
            )
            i_error += jnp.mean(_error_)

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

        dv = config["velocity"][1] - config["velocity"][0]
        if config["parameters"]["fe"]["symmetric"]:
            density_loss = jnp.mean(jnp.square(1.0 - 2.0 * jnp.sum(jnp.exp(params["fe"]) * dv, axis=1)))
            temperature_loss = jnp.mean(
                jnp.square(1.0 - 2.0 * jnp.sum(jnp.exp(params["fe"]) * config["velocity"] ** 2.0 * dv, axis=1))
            )
        else:
            density_loss = jnp.mean(jnp.square(1.0 - jnp.sum(jnp.exp(params["fe"]) * dv, axis=1)))
            temperature_loss = jnp.mean(
                jnp.square(1.0 - jnp.sum(jnp.exp(params["fe"]) * config["velocity"] ** 2.0 * dv, axis=1))
            )

        if config["parameters"]["fe"]["fe_decrease_strict"]:
            gradfe = jnp.sign(config["velocity"][1:]) * jnp.diff(params["fe"].squeeze())
            vals = jnp.where(gradfe > 0, gradfe, 0).sum()
            fe_penalty = jnp.tan(jnp.amin(jnp.array([vals, jnp.pi / 2])))
        else:
            fe_penalty = 0

        return i_error + e_error + density_loss + temperature_loss, [ThryE, normed_e_data, params]

    rng_key = jax.random.PRNGKey(42)
    init_weights = _get_params_.init(rng_key, dummy_batch)
    _v_func_ = jit(loss_fn)
    _vg_func_ = jit(value_and_grad(loss_fn, argnums=0, has_aux=True))
    _h_func_ = jit(jax.hessian(loss_for_hess_fn, argnums=0))

    if config["optimizer"]["method"] == "adam":
        vg_func = _vg_func_
        get_params = _get_params_
        h_func = _h_func_
        loss_dict = dict(
            vg_func=vg_func,
            array_loss_fn=array_loss_fn,
            init_weights=init_weights,
            get_params=get_params,
            h_func=h_func,
        )
    else:
        if config["nn"]["use"]:
            init_weights = init_weights
            flattened_weights, unravel_pytree = ravel_pytree(init_weights)
            bounds = None
        else:
            lb, ub, init_weights = init_weights_and_bounds(
                config, init_weights, num_slices=config["optimizer"]["batch_size"]
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
            grad["ts_parameter_generator"]["fe"] = 0.1 * grad["ts_parameter_generator"]["fe"]
            temp_grad, _ = ravel_pytree(grad)
            flattened_grads = np.array(temp_grad)
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

        h_func = _h_func_

        def get_params(weights, batch):
            pytree_weights = unravel_pytree(weights)
            return _get_params_.apply(pytree_weights, batch)

        loss_dict = dict(
            vg_func=vg_func,
            array_loss_fn=array_loss_fn,
            pytree_weights=init_weights,
            init_weights=flattened_weights,
            get_params=get_params,
            get_active_params=_get_active_params_.apply,
            bounds=bounds,
            unravel_pytree=unravel_pytree,
            v_func=v_func,
            h_func=h_func,
            calculate_spectra=calculate_spectra,
        )

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
    lb = {"ts_parameter_generator": {}}
    ub = {"ts_parameter_generator": {}}
    iw = {"ts_parameter_generator": {}}

    for k, v in init_weights["ts_parameter_generator"].items():
        lb["ts_parameter_generator"][k] = np.array([0 * config["units"]["lb"][k] for _ in range(num_slices)])
        ub["ts_parameter_generator"][k] = np.array([1.0 + 0 * config["units"]["ub"][k] for _ in range(num_slices)])
        if k != "fe":
            iw["ts_parameter_generator"][k] = np.array([config["parameters"][k]["val"] for _ in range(num_slices)])[
                :, None
            ]
        else:
            iw["ts_parameter_generator"][k] = np.concatenate(
                [config["parameters"][k]["val"] for _ in range(num_slices)]
            )
        iw["ts_parameter_generator"][k] = (iw["ts_parameter_generator"][k] - config["units"]["shifts"][k]) / config[
            "units"
        ]["norms"][k]

    return lb, ub, iw
