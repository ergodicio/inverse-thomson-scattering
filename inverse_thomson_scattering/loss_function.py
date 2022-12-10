from typing import Dict

import jax
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
import haiku as hk
import numpy as np
from inverse_thomson_scattering.fitmodl import get_fit_model


def get_loss_function(
    config: Dict, xie, sas, data: np.ndarray, norms: np.ndarray, shifts: np.ndarray, starting_params: np.ndarray
):
    fit_model = get_fit_model(config, xie, sas)
    lam = config["parameters"]["lam"]["val"]
    stddev = config["D"]["PhysParams"]["widIRF"]

    def transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins):
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddev[1] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddev[1]) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if config["D"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = TSins["amp3"]["val"] * amps[1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def transform_electron_spec(lamAxisE, modlE, amps, TSins):
        # Conceptual_origin so the convolution donsn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddev[0] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddev[0]) ** 2.0))
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

    @jit
    def get_spectra(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):

        if config["D"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan
            # raise NotImplementedError("Need to create an ion spectrum so we can compare it against data!")

        if config["D"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = transform_electron_spec(lamAxisE, modlE, amps, TSins)
        else:
            raise NotImplementedError("Need to create an electron spectrum so we can compare it against data!")

        return ThryE, ThryI, lamAxisE, lamAxisI

    vmap_fit_model = jit(vmap(fit_model))
    vmap_get_spectra = jit(vmap(get_spectra))

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(data[:, 1, :])
        e_norm = np.amax(data[:, 0, :])
    else:
        i_norm = e_norm = 1.0

    class compare_TS_spectra(hk.Module):
        def __init__(self, cfg):
            super(compare_TS_spectra, self).__init__()
            self.cfg = cfg

        def map_params(self):
            these_params = {}
            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = hk.get_parameter(
                        param_name,
                        shape=[],
                        init=hk.initializers.RandomUniform(minval=param_config["lb"], maxval=param_config["ub"]),
                    )
                else:
                    these_params[param_name] = jnp.array(param_config["val"])
            return these_params

        def __call__(self, *args, **kwargs):
            params = self.map_params()
            for k, v in params.items():
                params[k] = v[None, None]
            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_fit_model(params)
            ThryE, ThryI, lamAxisE, lamAxisI = vmap_get_spectra(
                modlE, modlI, lamAxisE, lamAxisI, jnp.concatenate(self.cfg["D"]["PhysParams"]["amps"]), live_TSinputs
            )

            ThryE = ThryE / e_norm
            ThryI = ThryI / i_norm

            loss = 0
            i_data = data[:, 1, :] / i_norm
            e_data = data[:, 0, :] / e_norm
            if self.cfg["D"]["extraoptions"]["fit_IAW"]:
                #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
                loss = loss + jnp.sum(jnp.square(i_data - ThryI))

            if self.cfg["D"]["extraoptions"]["fit_EPWb"]:
                thry_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), ThryE, 0.0)
                data_slc = jnp.where((lamAxisE > 450) & (lamAxisE < 510), e_data, 0.0)

                loss = loss + jnp.sum(jnp.square(data_slc - thry_slc))

            if self.cfg["D"]["extraoptions"]["fit_EPWr"]:
                thry_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), ThryE, 0.0)
                data_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 625), e_data, 0.0)

                loss = loss + jnp.sum(jnp.square(data_slc - thry_slc))

            return loss

    def loss_fn(x):
        calc_loss = compare_TS_spectra(config)
        return calc_loss(x)

    print(starting_params.shape)
    loss_fn = hk.without_apply_rng(hk.transform(loss_fn))
    init_params = loss_fn.init(jax.random.PRNGKey(42), starting_params)
    print(init_params)
    vg_func = jit(value_and_grad(loss_fn.apply))
    loss_func = jit(loss_fn.apply)
    hess_func = jit(jax.hessian(loss_fn.apply))

    def val_and_grad_loss(x: np.ndarray):
        x = x * norms + shifts
        reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
        value, grad = vg_func(reshaped_x)

        return value, np.array(grad).flatten()

    def value(x: np.ndarray):
        x = x * norms + shifts
        reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
        val = loss_func(reshaped_x)

        return val

    return value, val_and_grad_loss, hess_func
