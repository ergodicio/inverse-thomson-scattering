import copy

from inverse_thomson_scattering.form_factor import get_form_factor_fn
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func

from jax import numpy as jnp
from jax import jit
#from jax.config import config

#config.update('jax_disable_jit', True)

def get_fit_model(config, xie, sa):
    nonMaxwThomsonE_jax = get_form_factor_fn(config["D"]["lamrangE"], config["D"]["npts"], backend="jax")
    nonMaxwThomsonI_jax = get_form_factor_fn(config["D"]["lamrangI"], config["D"]["npts"], backend="jax")
    num_dist_func = get_num_dist_func(config["parameters"]["fe"]["type"], xie)

    def fit_model(fitted_params, sa_weights):

        parameters = config["parameters"]
        for key in parameters.keys():
            if parameters[key]["active"]:
                parameters[key]["val"] = jnp.squeeze(fitted_params[key])

        # if parameters["fe"]["active"]:
        #     parameters["fe"]["val"] = fitted_params[-parameters["fe"]["length"] : :]
        if parameters["m"]["active"]:
            parameters["fe"]["val"] = jnp.log(num_dist_func(parameters["m"]["val"]))

        # Add gradients to electron temperature and density just being applied to EPW
        cur_Te = jnp.linspace(
            (1 - parameters["Te"]["gradient"] / 200) * parameters["Te"]["val"],
            (1 + parameters["Te"]["gradient"] / 200) * parameters["Te"]["val"],
            10,
        )
        cur_ne = jnp.linspace(
            (1 - parameters["ne"]["gradient"] / 200) * parameters["ne"]["val"],
            (1 + parameters["ne"]["gradient"] / 200) * parameters["ne"]["val"],
            10,
        )

        fecur = jnp.exp(parameters["fe"]["val"])
        lam = parameters["lam"]["val"]

        if config["D"]["extraoptions"]["load_ion_spec"]:
            ThryI, lamAxisI = jit(nonMaxwThomsonI_jax)(
                parameters["Te"]["val"],
                parameters["Ti"]["val"],
                parameters["Z"]["val"],
                parameters["A"]["val"],
                parameters["fract"]["val"],
                parameters["ne"]["val"] * jnp.array([1e20]),
                parameters["Va"]["val"],
                parameters["ud"]["val"],
                sa["sa"],
                (fecur, xie),
                526.5,
            )

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7

            ThryI = jnp.real(ThryI)
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * sa["weights"], axis=1)
        else:
            modlI = 0
            lamAxisI = []

        if config["D"]["extraoptions"]["load_ele_spec"]:
            ThryE, lamAxisE = jit(nonMaxwThomsonE_jax)(
                cur_Te,
                parameters["Ti"]["val"],
                parameters["Z"]["val"],
                parameters["A"]["val"],
                parameters["fract"]["val"],
                cur_ne * 1e20,
                parameters["Va"]["val"],
                parameters["ud"]["val"],
                sa["sa"],
                (fecur, xie),
                lam,
            )

            # if parameters.fe['Type']=='MYDLM':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,parameters.fe['thetaphi'])
            # elif parameters.fe['Type']=='Numeric':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,[2*np.pi/3,0])
            # else:
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,expion=D['expandedions'])
            # nonMaxwThomson,_ =get_form_factor_fn(D['lamrangE'],lam)
            # [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,sa['sa'], [fecur,xie])

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7

            ThryE = jnp.real(ThryE)
            ThryE = jnp.mean(ThryE, axis=0)
            if config["D"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(sa_weights, ThryE.transpose())
            else: 
                modlE = jnp.sum(ThryE * sa_weights, axis=1)

            if config["D"]["iawoff"] and (config["D"]["lamrangE"][0] < lam < config["D"]["lamrangE"][1]):
                # set the ion feature to 0 #should be switched to a range about lam
                lamloc = jnp.argmin(jnp.abs(lamAxisE - lam))
                modlE = jnp.concatenate([modlE[: lamloc - 2000], jnp.zeros(4000), modlE[lamloc + 2000 :]])

            if config["D"]["iawfilter"][0]:
                filterb = config["D"]["iawfilter"][3] - config["D"]["iawfilter"][2] / 2
                filterr = config["D"]["iawfilter"][3] + config["D"]["iawfilter"][2] / 2
                if config["D"]["lamrangE"][0] < filterr and config["D"]["lamrangE"][1] > filterb:
                    if config["D"]["lamrangE"][0] < filterb:
                        lamleft = jnp.argmin(jnp.abs(lamAxisE - filterb))
                    else:
                        lamleft = 0

                    if config["D"]["lamrangE"][1] > filterr:
                        lamright = jnp.argmin(jnp.abs(lamAxisE - filterr))
                    else:
                        lamright = lamAxisE.size

                    indices = (filterb < lamAxisE) & (filterr > lamAxisE)
                    modlE = jnp.where(indices, modlE * 10 ** (-config["D"]["iawfilter"][1]), modlE)

        else:
            modlE = 0
            lamAxisE = []

        return modlE, modlI, lamAxisE, lamAxisI, parameters

    return fit_model
