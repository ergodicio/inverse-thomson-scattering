from jax.lax import scan
from jax import vmap
from jax import numpy as jnp
from inverse_thomson_scattering.process.convolve import fftconvolve


def add_ATS_IRF(config, sas, lamAxisE, modlE, amps, TSins, lam):
    # t0 = time.time()
    # print("Staritng add_ATS_IRF ", round(time.time() - t0, 2))
    stddev_lam = config["other"]["PhysParams"]["widIRF"]["spect_FWHM_ele"] / 2.3548
    stddev_ang = config["other"]["PhysParams"]["widIRF"]["ang_FWHM_ele"] / 2.3548
    # Conceptual_origin so the convolution donsn't shift the signal
    origin_lam = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    origin_ang = (jnp.amax(sas["angAxis"]) + jnp.amin(sas["angAxis"])) / 2.0
    # print("Origins defined ", round(time.time() - t0, 2))
    inst_func_lam = jnp.squeeze(
        (1.0 / (stddev_lam * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((lamAxisE - origin_lam) ** 2.0) / (2.0 * (stddev_lam) ** 2.0))
    )  # Gaussian
    inst_func_ang = jnp.squeeze(
        (1.0 / (stddev_ang * jnp.sqrt(2.0 * jnp.pi)))
        * jnp.exp(-((sas["angAxis"] - origin_ang) ** 2.0) / (2.0 * (stddev_ang) ** 2.0))
    )  # Gaussian
    # print("inst ang shape ", jnp.shape(inst_func_ang))
    # print("inst lam shape ", jnp.shape(inst_func_lam))
    # apply 2d convolution
    # print("modlE shape ", jnp.shape(modlE))
    # print("inst_funcs defined ", round(time.time() - t0, 2))
    ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    #ThryE = jnp.array([fftconvolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
    # print("ThryE shape after conv1 ", jnp.shape(ThryE))
    # print("convolve1 done ", round(time.time() - t0, 2))
    ThryE = jnp.array([jnp.convolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
    #ThryE = jnp.array([fftconvolve(ThryE[:, i], inst_func_lam, "same") for i in range(ThryE.shape[1])])
    # renorm (not sure why this is needed)
    # ThryE = jnp.array([(jnp.amax(modlE[:, i]) / jnp.amax(ThryE[:, i])) * ThryE[:, i] for i in range(modlE.shape[1])])
    #print(jnp.shape(ThryE))
    # print(jnp.shape(modlE))
    # print(jnp.shape(jnp.amax(modlE, axis=1, keepdims=True)))
    ThryE = jnp.amax(modlE, axis=1, keepdims=True) / jnp.amax(ThryE, axis=1, keepdims=True) * ThryE
    # ThryE = ThryE.transpose()
    # jax.debug.print("in IRF")
    # print("inst_funcs defined ", round(time.time() - t0, 2))
    # inst_func_2D = jnp.outer(inst_func_ang, inst_func_lam)
    # print(jnp.shape(inst_func_2D))
    # print(jnp.shape(modlE))
    # print("outer produce done ", round(time.time() - t0, 2))
    # jax.debug.print("done outer")

    # print(type(modlE))
    # print(type(inst_func_2D))
    # print(type(modlE[1][1]))
    # print(type(inst_func_2D[1][1]))
    # return
    # ThryE = convolve(modlE, inst_func_2D, "same", "direct")
    # print("convolve done ", round(time.time() - t0, 2))
    # jax.debug.print("done convolve")

    # print("ThryE shape after conv2 ", jnp.shape(ThryE))

    if config["other"]["PhysParams"]["norm"] > 0:
        ThryE = jnp.where(
            lamAxisE < lam,
            TSins["amp1"] * (ThryE / jnp.amax(ThryE[lamAxisE < lam])),
            TSins["amp2"] * (ThryE / jnp.amax(ThryE[lamAxisE > lam])),
        )
    # print("renorm done ", round(time.time() - t0, 2))
    return lamAxisE, ThryE


def add_ion_IRF(config, lamAxisI, modlI, lamAxisE, amps, TSins):
    stddevI = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ion"]
    originI = (jnp.amax(lamAxisI) + jnp.amin(lamAxisI)) / 2.0
    inst_funcI = jnp.squeeze(
        (1.0 / (stddevI * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisI - originI) ** 2.0) / (2.0 * (stddevI) ** 2.0))
    )  # Gaussian
    ThryI = jnp.convolve(modlI, inst_funcI, "same")
    ThryI = (jnp.amax(modlI) / jnp.amax(ThryI)) * ThryI
    ThryI = jnp.average(ThryI.reshape(1024, -1), axis=1)

    if config["other"]["PhysParams"]["norm"] == 0:
        lamAxisI = jnp.average(lamAxisI.reshape(1024, -1), axis=1)
        ThryI = TSins["amp3"]["val"] * amps * ThryI / jnp.amax(ThryI)
        lamAxisE = jnp.average(lamAxisE.reshape(1024, -1), axis=1)

    return lamAxisI, lamAxisE, ThryI


def add_electron_IRF(config, lamAxisE, modlE, amps, TSins, lam):
    stddevE = config["other"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
    # Conceptual_origin so the convolution doesn't shift the signal
    originE = (jnp.amax(lamAxisE) + jnp.amin(lamAxisE)) / 2.0
    inst_funcE = jnp.squeeze(
        (1.0 / (stddevE * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(-((lamAxisE - originE) ** 2.0) / (2.0 * (stddevE) ** 2.0))
    )  # Gaussian
    ThryE = jnp.convolve(modlE, inst_funcE, "same")
    ThryE = jnp.array([jnp.convolve(modlE[:, i], inst_func_ang, "same") for i in range(modlE.shape[1])])
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
        ThryE = amps * ThryE / jnp.amax(ThryE)
        ThryE = jnp.where(lamAxisE < lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)

    return lamAxisE, ThryE
