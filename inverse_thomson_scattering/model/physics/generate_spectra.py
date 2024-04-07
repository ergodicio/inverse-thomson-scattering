from typing import Dict

from inverse_thomson_scattering.model.physics.form_factor import FormFactor

# from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func
from inverse_thomson_scattering.misc.gen_num_dist_func import DistFunc

from jax import numpy as jnp


class FitModel:
    """
    The FitModel Class wraps the FormFactor class adding finite aperture effects and finite volume effects. This class
    also handles the options for calculating the form factor.

    Args:
        config: Dict- configuration dictionary built from input deck
        sa: Dict- has fields containing the scattering angles the spectrum will be calculated at and the relative
        weights of each of the scattering angles in the final spectrum
    """

    def __init__(self, config: Dict, sa):
        self.config = config
        self.sa = sa
        # this will need to be fixed for multi electron
        self.num_ions = 0
        self.num_electrons = 0
        for species in config["parameters"].keys():
            if "electron" in config["parameters"][species]["type"].keys():
                self.num_dist_func = DistFunc(config["parameters"][species])
                self.e_species = species
                self.num_electrons += 1
            elif "ion" in config["parameters"][species]["type"].keys():
                self.num_ions += 1

        self.electron_form_factor = FormFactor(
            config["other"]["lamrangE"],
            npts=config["other"]["npts"],
            fe_dim=self.num_dist_func.dim,
            vax=config["parameters"][self.e_species]["fe"]["velocity"],
        )
        self.ion_form_factor = FormFactor(
            config["other"]["lamrangI"],
            npts=config["other"]["npts"],
            fe_dim=self.num_dist_func.dim,
            vax=config["parameters"][self.e_species]["fe"]["velocity"],
        )

    def __call__(self, all_params: Dict):
        """
        TODO


        Args:
            all_params:

        Returns:

        """

        # not sure why this is required
        # for key in self.config["parameters"].keys():
        #     if key != "fe":
        #         all_params[key] = jnp.squeeze(all_params[key])

        if self.config["parameters"][self.e_species]["m"]["active"]:
            (
                self.config["parameters"][self.e_species]["fe"]["velocity"],
                all_params[self.e_species]["fe"],
            ) = self.num_dist_func(all_params[self.e_species]["m"])
            # self.config["velocity"], all_params["fe"] = self.num_dist_func(self.config["parameters"]["m"]["val"])
            all_params[self.e_species]["fe"] = jnp.log(all_params[self.e_species]["fe"])
            # all_params["fe"] = jnp.log(self.num_dist_func(self.config["parameters"]["m"]))
            if (
                self.config["parameters"][self.e_species]["m"]["active"]
                and self.config["parameters"][self.e_species]["fe"]["active"]
            ):
                raise ValueError("m and fe cannot be actively fit at the same time")

        # Add gradients to electron temperature and density just being applied to EPW
        cur_Te = jnp.zeros((self.config["parameters"]["general"]["Te_gradient"]["num_grad_points"], self.num_electrons))
        cur_ne = jnp.zeros((self.config["parameters"]["general"]["Te_gradient"]["num_grad_points"], self.num_electrons))
        A = jnp.zeros(self.num_ions)
        Z = jnp.zeros(self.num_ions)
        Ti = jnp.zeros(self.num_ions)
        fract = jnp.zeros(self.num_ions)

        ion_c = 0
        ele_c = 0
        for species in self.config["parameters"].keys():
            if "electron" in self.config["parameters"][species]["type"].keys():
                cur_Te = cur_Te.at[:, ele_c].set(
                    jnp.linspace(
                        (1 - all_params["general"]["Te_gradient"] / 200) * all_params[species]["Te"],
                        (1 + all_params["general"]["Te_gradient"] / 200) * all_params[species]["Te"],
                        self.config["parameters"]["general"]["Te_gradient"]["num_grad_points"],
                    ).squeeze(-1)
                )

                cur_ne = cur_ne.at[:, ele_c].set(
                    (
                        jnp.linspace(
                            (1 - all_params["general"]["ne_gradient"] / 200) * all_params[species]["ne"],
                            (1 + all_params["general"]["ne_gradient"] / 200) * all_params[species]["ne"],
                            self.config["parameters"]["general"]["ne_gradient"]["num_grad_points"],
                        )
                        * 1e20
                    ).squeeze(-1)
                )
                ele_c += 1

            elif "ion" in self.config["parameters"][species]["type"].keys():
                A = A.at[ion_c].set(all_params[species]["A"].squeeze(-1))
                Z = Z.at[ion_c].set(all_params[species]["Z"].squeeze(-1))
                if self.config["parameters"][species]["Ti"]["same"]:
                    Ti = Ti.at[ion_c].set(Ti[ion_c - 1])
                else:
                    Ti = Ti.at[ion_c].set(all_params[species]["Ti"].squeeze(-1))
                fract = fract.at[ion_c].set(all_params[species]["fract"].squeeze(-1))
                ion_c += 1

        # cur_ne = jnp.array(cur_ne).squeeze()
        # cur_Te = jnp.array(cur_Te).squeeze()
        # Ti = jnp.array(Ti).squeeze()

        fecur = jnp.exp(all_params[self.e_species]["fe"])
        vcur = self.config["parameters"][self.e_species]["fe"]["velocity"]
        if self.config["parameters"][self.e_species]["fe"]["symmetric"]:
            fecur = jnp.concatenate((jnp.flip(fecur[1:]), fecur))
            vcur = jnp.concatenate((-jnp.flip(vcur[1:]), vcur))

        lam = all_params["general"]["lam"]

        if self.config["other"]["extraoptions"]["load_ion_spec"]:
            if self.num_dist_func.dim == 1:
                ThryI, lamAxisI = self.ion_form_factor(
                    all_params, cur_ne, cur_Te, A, Z, Ti, fract, self.sa["sa"], (fecur, vcur), lam
                )
            else:
                ThryI, lamAxisI = self.ion_form_factor.calc_in_2D(
                    all_params,
                    self.config["parameters"]["general"]["ud"]["angle"],
                    self.config["parameters"]["general"]["ud"]["angle"],
                    cur_ne * jnp.array([1e20]),
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
                    self.sa["sa"],
                    (fecur, vcur),
                    lam,
                )

            # remove extra dimensions and rescale to nm
            lamAxisI = jnp.squeeze(lamAxisI) * 1e7  # TODO hardcoded

            ThryI = jnp.real(ThryI)
            ThryI = jnp.mean(ThryI, axis=0)
            modlI = jnp.sum(ThryI * self.sa["weights"][0], axis=1)
        else:
            modlI = 0
            lamAxisI = []

        if self.config["other"]["extraoptions"]["load_ele_spec"]:
            if self.num_dist_func.dim == 1:
                ThryE, lamAxisE = self.electron_form_factor(
                    all_params,
                    cur_ne,
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
                    self.sa["sa"],
                    (fecur, vcur),
                    lam + self.config["data"]["ele_lam_shift"],
                )
            else:
                ThryE, lamAxisE = self.electron_form_factor.calc_in_2D(
                    all_params,
                    self.config["parameters"]["general"]["ud"]["angle"],
                    self.config["parameters"]["general"]["ud"]["angle"],
                    cur_ne,
                    cur_Te,
                    A,
                    Z,
                    Ti,
                    fract,
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
                lamlocb = jnp.argmin(jnp.abs(lamAxisE - lam - 3.0))
                lamlocr = jnp.argmin(jnp.abs(lamAxisE - lam + 3.0))
                modlE = jnp.concatenate(
                    [modlE[:lamlocb], jnp.zeros(lamlocr - lamlocb), modlE[lamlocr:]]
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
