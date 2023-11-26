from inverse_thomson_scattering.model.physics.form_factor import FormFactor
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func

from jax import numpy as jnp


class FitModel:
    def __init__(self, config, sa):
        self.config = config
        self.sa = sa
        self.electron_form_factor = FormFactor(config["other"]["lamrangE"], npts=config["other"]["npts"])
        self.ion_form_factor = FormFactor(config["other"]["lamrangI"], npts=config["other"]["npts"])
        self.num_dist_func = get_num_dist_func(config["parameters"]["fe"]["type"], config["velocity"])

    def __call__(self, all_params):
        for key in self.config["parameters"].keys():
            if key != "fe":
                all_params[key] = jnp.squeeze(all_params[key])

        if self.config["parameters"]["m"]["active"]:
            all_params["fe"] = jnp.log(self.num_dist_func(all_params["m"]))
            if self.config["parameters"]["m"]["active"] and self.config["parameters"]["fe"]["active"]:
                raise ValueError("m and fe cannot be actively fit at the same time")

        # Add gradients to electron temperature and density just being applied to EPW
        cur_Te = jnp.linspace(
            (1 - all_params["Te_gradient"] / 200) * all_params["Te"],
            (1 + all_params["Te_gradient"] / 200) * all_params["Te"],
            self.config["parameters"]["Te_gradient"]["num_grad_points"],
        )
        cur_ne = jnp.linspace(
            (1 - all_params["ne_gradient"] / 200) * all_params["ne"],
            (1 + all_params["ne_gradient"] / 200) * all_params["ne"],
            self.config["parameters"]["ne_gradient"]["num_grad_points"],
        )

        fecur = jnp.exp(all_params["fe"])
        vcur = self.config["velocity"]
        if self.config["parameters"]["fe"]["symmetric"]:
            fecur = jnp.concatenate((jnp.flip(fecur[1:]), fecur))
            vcur = jnp.concatenate((-jnp.flip(vcur[1:]), vcur))

        lam = all_params["lam"]

        if self.config["other"]["extraoptions"]["load_ion_spec"]:
            ThryI, lamAxisI = self.ion_form_factor(all_params, cur_ne * 1e20, cur_Te, self.sa["sa"], (fecur, vcur), lam)

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded

            ThryI = jnp.real(ThryI)
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * self.sa["weights"][0], axis=1)
        else:
            modlI = 0
            lamAxisI = []

        if self.config["other"]["extraoptions"]["load_ele_spec"]:
            ThryE, lamAxisE = self.electron_form_factor(
                all_params,
                cur_ne * jnp.array([1e20]),
                cur_Te,
                self.sa["sa"],
                (fecur, vcur),
                lam + self.config["data"]["ele_lam_shift"],
            )

            # remove extra dimensions and rescale to nm
            lamAxisE = jnp.squeeze(lamAxisE) * 1e7  # TODO hardcoded

            ThryE = jnp.real(ThryE)
            ThryE = jnp.mean(ThryE, axis=0)
            if self.config["other"]["extraoptions"]["spectype"] == "angular_full":
                modlE = jnp.matmul(self.sa["weights"], ThryE.transpose())
            else:
                modlE = jnp.sum(ThryE * self.sa["weights"][0], axis=1)

            if self.config["other"]["iawoff"] and (
                self.config["other"]["lamrangE"][0] < lam < self.config["other"]["lamrangE"][1]
            ):
                # set the ion feature to 0 #should be switched to a range about lam
                lamloc = jnp.argmin(jnp.abs(lamAxisE - lam))
                modlE = jnp.concatenate(
                    [modlE[: lamloc - 2000], jnp.zeros(4000), modlE[lamloc + 2000 :]]
                )  # TODO hardcoded

            if self.config["other"]["iawfilter"][0]:
                filterb = self.config["other"]["iawfilter"][3] - self.config["other"]["iawfilter"][2] / 2
                filterr = self.config["other"]["iawfilter"][3] + self.config["other"]["iawfilter"][2] / 2

                if self.config["other"]["lamrangE"][0] < filterr and self.config["other"]["lamrangE"][1] > filterb:
                    indices = (filterb < lamAxisE) & (filterr > lamAxisE)
                    modlE = jnp.where(indices, modlE * 10 ** (-self.config["other"]["iawfilter"][1]), modlE)
        else:
            modlE = 0
            lamAxisE = []

        return modlE, modlI, lamAxisE, lamAxisI, all_params
