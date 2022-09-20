from jax import numpy as jnp
from jax import jit
import jax
from inverse_thomson_scattering.v0.fitmodl import get_fitModel2


def get_chisq2(TSinputs, xie, sas, data):

    fitModel2 = get_fitModel2(TSinputs, xie, sas)

    @jit
    def rest_of_chisq2(modlE, modlI, lamAxisE, lamAxisI):
        lam = TSinputs["lam"]["val"]
        amp1 = TSinputs["amp1"]["val"]
        amp2 = TSinputs["amp2"]["val"]
        amp3 = TSinputs["amp3"]["val"]

        stddev = TSinputs["D"]["PhysParams"]["widIRF"]

        if TSinputs["D"]["extraoptions"]["load_ion_spec"]:
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
                ThryI = amp3 * TSinputs["D"]["PhysParams"]["amps"][1] * ThryI / jnp.amax(ThryI)
                lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

        if TSinputs["D"]["extraoptions"]["load_ele_spec"]:
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
                ThryE = TSinputs["D"]["PhysParams"]["amps"][0] * ThryE / jnp.amax(ThryE)
                ThryE = jnp.where(lamAxisE < lam, amp1 * ThryE, amp2 * ThryE)

        return ThryE, ThryI, lamAxisE

    def chiSq2(x):

        modlE, modlI, lamAxisE, lamAxisI = fitModel2(x)
        ThryE, ThryI, lamAxisE = rest_of_chisq2(modlE, modlI, lamAxisE, lamAxisI)
        chisq = 0
        if TSinputs["D"]["extraoptions"]["fit_IAW"]:
            #    chisq=chisq+sum((10*data(2,:)-10*ThryI).^2); %multiplier of 100 is to set IAW and EPW data on the same scale 7-5-20 %changed to 10 9-1-21
            chisq = chisq + jnp.sum((data[1, :] - ThryI) ** 2)

        if TSinputs["D"]["extraoptions"]["fit_EPWb"]:
            chisq = chisq + jnp.sum(
                (data[0, (lamAxisE > 410) & (lamAxisE < 510)] - ThryE[(lamAxisE > 410) & (lamAxisE < 510)]) ** 2
            )

        if TSinputs["D"]["extraoptions"]["fit_EPWr"]:
            chisq = chisq + jnp.sum(
                (data[0, (lamAxisE > 540) & (lamAxisE < 680)] - ThryE[(lamAxisE > 540) & (lamAxisE < 680)]) ** 2
            )

        return chisq

    vg_func = jax.value_and_grad(chiSq2)

    def val_and_grad_chisq2(x):
        x = jnp.array(x)
        value, grad = vg_func(x)

        return value, jnp.array(grad)

    return chiSq2, val_and_grad_chisq2


def initFe(TSinputs, xie):
    # generate fe from inputs or keep numerical fe
    if TSinputs["fe"]["type"] == "DLM":
        TSinputs["fe"]["val"] = jnp.log(
            NumDistFunc([TSinputs["fe"]["type"], TSinputs["m"]["val"]], xie, TSinputs["fe"]["type"])
        )

    elif TSinputs["fe"]["type"] == "Fourkal":
        TSinputs["fe"]["val"] = jnp.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["Z"]["val"]], xie, TSinputs["fe"]["type"]
            )
        )

    elif TSinputs["fe"]["type"] == "SpitzerDLM":
        TSinputs["fe"]["val"] = jnp.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["fe"]["theta"], TSinputs["fe"]["delT"]],
                xie,
                TSinputs["fe"]["type"],
            )
        )

    elif TSinputs["fe"]["type"] == "MYDLM":  # This will eventually need another parameter for density gradient
        TSinputs["fe"]["val"] = jnp.log(
            NumDistFunc(
                [TSinputs["fe"]["type"], TSinputs["m"]["val"], TSinputs["fe"]["theta"], TSinputs["fe"]["delT"]],
                xie,
                TSinputs["fe"]["type"],
            )
        )

    else:
        raise NameError("Unrecognized distribtuion function type")

    TSinputs["fe"]["val"][TSinputs["fe"]["val"] <= -100] = -99
