from typing import Dict

import jax
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
import numpy as np
from inverse_thomson_scattering.fitmodl import get_fit_model
#from jax.config import config

#config.update('jax_disable_jit', True)


def get_loss_function(config: Dict, xie, sas, data: np.ndarray, norms: np.ndarray, shifts: np.ndarray):
    fit_model = get_fit_model(config, xie, sas)
    lam = config["parameters"]["lam"]["val"]

    def transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins):
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

    def transform_electron_spec(lamAxisE, modlE, amps, TSins):
        stddevE = config["D"]["PhysParams"]["widIRF"]["spect_stddev_ele"]
        # Conceptual_origin so the convolution donsn't shift the signal
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
    
    def transform_angular_spec(lamAxisE, modlE, amps, TSins):
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
    def get_spectra(modlE, modlI, lamAxisE, lamAxisI, amps, TSins):

        if config["D"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, lamAxisE, ThryI = transform_ion_spec(lamAxisI, modlI, lamAxisE, amps, TSins)
        else:
            lamAxisI = jnp.nan
            ThryI = jnp.nan

        if config["D"]["extraoptions"]["load_ele_spec"] & (config["D"]["extraoptions"]["spectype"] == "angular_full"):
            lamAxisE, ThryE = transform_angular_spec(lamAxisE, modlE, amps, TSins)
        elif config["D"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = transform_electron_spec(lamAxisE, modlE, amps, TSins)
        else:
            lamAxisE = jnp.nan
            ThryE = jnp.nan

        return ThryE, ThryI, lamAxisE, lamAxisI

    vmap_fit_model = jit(vmap(fit_model))
    jit_fit_model = jit(fit_model) #ATS data can't be vmaped
    vmap_get_spectra = jit(vmap(get_spectra))
    jit_get_spectra = jit(get_spectra) #ATS data can't be vmaped

    if config["optimizer"]["y_norm"]:
        i_norm = np.amax(data[:, 1, :])
        e_norm = np.amax(data[:, 0, :])
    else:
        i_norm = e_norm = 1.0

    def loss_fn(x: jnp.ndarray):
        if config["D"]["extraoptions"]["spectype"] == "angular_full":
            reshaped_x = jnp.array(x)
            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = jit_fit_model(reshaped_x, sas["weights"])
            #print("modleE shape ", jnp.shape(modlE))
            ThryE, ThryI, lamAxisE, lamAxisI = jit_get_spectra(
                modlE, modlI, lamAxisE, lamAxisI, config["D"]["PhysParams"]["amps"], live_TSinputs
            )
        else:
            reshaped_x = jnp.array(x.reshape((data.shape[0], -1)))
            #print("x shape ",jnp.shape(reshaped_x))
            #print("weights shape ", jnp.shape(sas["weights"]))
            modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = vmap_fit_model(reshaped_x, sas["weights"])
            #print("modleE shape ", jnp.shape(modlE))
            ThryE, ThryI, lamAxisE, lamAxisI = vmap_get_spectra(
                modlE, modlI, lamAxisE, lamAxisI, jnp.concatenate(config["D"]["PhysParams"]["amps"]), live_TSinputs
            )
        #print("ThryE shape ", jnp.shape(ThryE))
        ThryE = ThryE + jnp.array(config["D"]["PhysParams"]["noiseE"])
        ThryI = ThryI + jnp.array(config["D"]["PhysParams"]["noiseI"])
        
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
            sqdev = jnp.square(e_data - ThryE) /ThryE
            sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["blue_min"]) 
                              & (lamAxisE < config["D"]["fit_rng"]["blue_max"]),
                              sqdev, 0.0)
            
            loss = loss + jnp.sum(sqdev)

        if config["D"]["extraoptions"]["fit_EPWr"]:
            sqdev = jnp.square(e_data - ThryE) /ThryE
            sqdev = jnp.where((lamAxisE > config["D"]["fit_rng"]["red_min"]) 
                              & (lamAxisE < config["D"]["fit_rng"]["red_max"]),
                              sqdev, 0.0)
            
            loss = loss + jnp.sum(sqdev)

        return loss

    vg_func = jit(value_and_grad(loss_fn))
    loss_func = jit(loss_fn)
    hess_func = jit(jax.hessian(loss_fn))

    def val_and_grad_loss(x: np.ndarray):
        x = x * norms + shifts
        value, grad = vg_func(x)

        return value, np.array(grad).flatten()
    
    def value(x: np.ndarray):
        x = x * norms + shifts
        val = loss_func(x)

        return val

    return value, val_and_grad_loss, hess_func
