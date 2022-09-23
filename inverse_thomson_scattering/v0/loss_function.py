from typing import Dict
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
import numpy as np
from inverse_thomson_scattering.v0.fitmodl import get_fit_model


def get_loss_function(TSinputs: Dict, xie, sas, data: np.ndarray):
    fit_model = get_fit_model(TSinputs, xie, sas)
    lam = TSinputs["lam"]["val"]
    amp1 = TSinputs["amp1"]["val"]
    amp2 = TSinputs["amp2"]["val"]
    amp3 = TSinputs["amp3"]["val"]

    stddev = TSinputs["D"]["PhysParams"]["widIRF"]

    def load_ion_spec(lamAxisI, modlI, lamAxisE, amps):
        originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
        inst_funcI = jnp.squeeze(
            (1.0 / (stddev[1] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddev[1]) ** 2.0))
        )  # Gaussian
        ThryI = jnp.convolve(modlI, inst_funcI, "same")
        ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
        ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

        if TSinputs["D"]["PhysParams"]["norm"] == 0:
            lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
            ThryI = amp3 * amps[1] * ThryI / jnp.amax(ThryI)
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        return lamAxisI, lamAxisE, ThryI

    def load_electron_spec(lamAxisE, modlE, amps):
        # Conceptual_origin so the convolution donsn't shift the signal
        originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
        inst_funcE = jnp.squeeze(
            (1.0 / (stddev[0] * jnp.sqrt(2.0 * jnp.pi)))
            * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddev[0]) ** 2.0))
        )  # Gaussian
        ThryE = jnp.convolve(modlE, inst_funcE, "same")
        ThryE = (jnp.amax(modlE) / jnp.amax(ThryE)) * ThryE

        if TSinputs["D"]["PhysParams"]["norm"] > 0:
            ThryE = jnp.where(
                lamAxisE < lam,
                amp1 * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
                amp2 * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
            )

        ThryE = jnp.average(ThryE.reshape(1024, -1), axis=1)
        if TSinputs["D"]["PhysParams"]["norm"] == 0:
            lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)
            ThryE = amps[0] * ThryE / jnp.amax(ThryE)
            ThryE = jnp.where(lamAxisE < lam, amp1 * ThryE, amp2 * ThryE)

        return lamAxisE, ThryE

    @jit
    def get_spectra(modlE, modlI, lamAxisE, lamAxisI, amps):

        if TSinputs["D"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = load_ion_spec(lamAxisI, modlI, lamAxisE, amps)
        else:
            raise NotImplementedError("Need to create an ion spectrum so we can compare it against data!")

        if TSinputs["D"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = load_electron_spec(lamAxisE, modlE, amps)
        else:
            raise NotImplementedError("Need to create an electron spectrum so we can compare it against data!")

        return ThryE, ThryI, lamAxisE, lamAxisI

    vmap_fit_model = vmap(fit_model)
    vmap_get_spectra = jit(vmap(get_spectra))

    def loss_fn(x: jnp.ndarray):
        # modlE, modlI, lamAxisE, lamAxisI = fit_model(x)
        modlE, modlI, lamAxisE, lamAxisI = vmap_fit_model(x)
        # ThryE, ThryI, lamAxisE, lamAxisI = get_spectra(
        #     modlE, modlI, lamAxisE, lamAxisI, TSinputs["D"]["PhysParams"]["amps"]
        # )
        ThryE, ThryI, lamAxisE, lamAxisI = vmap_get_spectra(
            modlE, modlI, lamAxisE, lamAxisI, jnp.concatenate(TSinputs["D"]["PhysParams"]["amps"])
        )

        loss = 0
        i_data = data[:, 1, :]
        e_data = data[:, 0, :]
        if TSinputs["D"]["extraoptions"]["fit_IAW"]:
            #    loss=loss+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            loss = loss + jnp.sum(jnp.square(i_data - ThryI))

        if TSinputs["D"]["extraoptions"]["fit_EPWb"]:
            thry_slc = jnp.where((lamAxisE > 410) & (lamAxisE < 510), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 410) & (lamAxisE < 510), e_data, 0.0)

            # data_slc = data[:, 0, :]
            # thry_slc = ThryE  # [eslc[:, 0, :]]
            loss = loss + jnp.sum((data_slc - thry_slc) ** 2)

        if TSinputs["D"]["extraoptions"]["fit_EPWr"]:
            thry_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 680), ThryE, 0.0)
            data_slc = jnp.where((lamAxisE > 540) & (lamAxisE < 680), e_data, 0.0)

            # data_slc = data[:, 0, :]
            # thry_slc = ThryE  # [eslc[:, 0, :]]
            loss = loss + jnp.sum(jnp.square(data_slc - thry_slc))

        return loss

    vg_func = jit(value_and_grad(loss_fn))

    def val_and_grad_loss(x: np.ndarray):
        reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
        value, grad = vg_func(reshaped_x)

        return value, np.array(grad).flatten()

    return loss_fn, val_and_grad_loss
