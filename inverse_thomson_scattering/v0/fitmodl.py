import copy

from inverse_thomson_scattering.jax.form_factor import get_form_factor_fn
from inverse_thomson_scattering.v0.numDistFunc import get_num_dist_func
from jax import numpy as jnp
from jax import jit
import jax


def get_fit_model(TSins, xie, sa):
    nonMaxwThomsonE_jax, _ = get_form_factor_fn(TSins["D"]["lamrangE"])
    nonMaxwThomsonI_jax, _ = get_form_factor_fn(TSins["D"]["lamrangI"])
    num_dist_func = get_num_dist_func(TSins["fe"]["type"], xie)

    def fit_model(x):
        param_dict = copy.deepcopy(TSins)

        i = 0
        for key in TSins.keys():
            if TSins[key]["active"]:
                param_dict[key]["val"] = x[i]
                i = i + 1
        if TSins["fe"]["active"]:
            param_dict["fe"]["val"] = x[-TSins["fe"]["length"] : :]
        elif TSins["m"]["active"]:
            param_dict["fe"]["val"] = jnp.log(num_dist_func(TSins["m"]["val"]))

        fecur = jnp.exp(param_dict["fe"]["val"])
        lam = param_dict["lam"]["val"]

        if TSins["D"]["extraoptions"]["load_ion_spec"]:
            ThryI, lamAxisI = jit(nonMaxwThomsonI_jax)(
                param_dict["Te"]["val"],
                param_dict["Ti"]["val"],
                param_dict["Z"]["val"],
                param_dict["A"]["val"],
                param_dict["fract"]["val"],
                param_dict["ne"]["val"] * 1e20,
                param_dict["Va"]["val"],
                param_dict["ud"]["val"],
                sa["sa"],
                (fecur, xie),
                526.5,
                # ,
                # expion=D["expandedions"],
            )

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7

            ThryI = jnp.real(ThryI)
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * sa["weights"], axis=1)
        else:
            modlI = []
            lamAxisI = []

        if TSins["D"]["extraoptions"]["load_ele_spec"]:
            ThryE, lamAxisE = jit(nonMaxwThomsonE_jax)(
                param_dict["Te"]["val"],
                param_dict["Ti"]["val"],
                param_dict["Z"]["val"],
                param_dict["A"]["val"],
                param_dict["fract"]["val"],
                param_dict["ne"]["val"] * 1e20,
                param_dict["Va"]["val"],
                param_dict["ud"]["val"],
                sa["sa"],
                (fecur, xie),
                lam,
                # ,
                # expion=D["expandedions"],
            )

            # if TSins.fe['Type']=='MYDLM':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,TSins.fe['thetaphi'])
            # elif TSins.fe['Type']=='Numeric':
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,[2*np.pi/3,0])
            # else:
            #    [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,D['lamrangE'],lam,sa['sa'], fecur,xie,expion=D['expandedions'])
            # nonMaxwThomson,_ =get_form_factor_fn(D['lamrangE'],lam)
            # [Thry,lamAxisE]=nonMaxwThomson(Te,Te,1,1,1,ne*1e20,0,0,sa['sa'], [fecur,xie])

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7

            ThryE = jnp.real(ThryE)
            ThryE = jnp.mean(ThryE, axis=0)
            modlE = jnp.sum(ThryE * sa["weights"], axis=1)

            # [modl,lamAx]=S2Signal(Thry,lamAxis,D);

            if TSins["D"]["iawoff"] and (TSins["D"]["lamrangE"][0] < lam and TSins["D"]["lamrangE"][1] > lam):
                # set the ion feature to 0 #should be switched to a range about lam
                lamloc = jnp.argmin(jnp.abs(lamAxisE - lam))
                modlE = jnp.concatenate([modlE[: lamloc - 2000], jnp.zeros(4000), modlE[lamloc + 2000 :]])

            if TSins["D"]["iawfilter"][0]:
                filterb = TSins["D"]["iawfilter"][3] - TSins["D"]["iawfilter"][2] / 2
                filterr = TSins["D"]["iawfilter"][3] + TSins["D"]["iawfilter"][2] / 2
                if TSins["D"]["lamrangE"][0] < filterr and TSins["D"]["lamrangE"][1] > filterb:
                    if TSins["D"]["lamrangE"][0] < filterb:
                        lamleft = jnp.argmin(jnp.abs(lamAxisE - filterb))
                    else:
                        lamleft = 0

                    if TSins["D"]["lamrangE"][1] > filterr:
                        lamright = jnp.argmin(jnp.abs(lamAxisE - filterr))
                    else:
                        lamright = lamAxisE.size

                    modlE = jnp.concatenate(
                        [
                            modlE[:lamleft],
                            modlE[lamleft:lamright] * 10 ** (-TSins["D"]["iawfilter"][1]),
                            modlE[lamright:],
                        ]
                    )
        else:
            modlE = []
            lamAxisE = []

        return modlE, modlI, lamAxisE, lamAxisI

    return fit_model
