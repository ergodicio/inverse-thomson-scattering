from jax import numpy as jnp
from jax import vmap

from inverse_thomson_scattering.model.physics.generate_spectra import get_fit_model
from inverse_thomson_scattering.process import irf


class SpectrumCalculator:
    def __init__(self, cfg, sas, dummy_batch):
        super().__init__()
        self.cfg = cfg
        self.sas = sas

        self.forward_pass = get_fit_model(cfg, sas, backend="jax")
        self.lam = cfg["parameters"]["lam"]["val"]

        if (
            cfg["other"]["extraoptions"]["spectype"] == "angular_full"
            or max(dummy_batch["e_data"].shape[0], dummy_batch["i_data"].shape[0]) <= 1
        ):
            # ATS data can't be vmaped and single lineouts cant be vmapped
            self.vmap_forward_pass = self.forward_pass
            self.vmap_postprocess_thry = self.postprocess_thry
        else:
            self.vmap_forward_pass = vmap(self.forward_pass)
            self.vmap_postprocess_thry = vmap(self.postprocess_thry)

    def postprocess_thry(self, modlE, modlI, lamAxisE, lamAxisI, amps, TSins):
        if self.cfg["other"]["extraoptions"]["load_ion_spec"]:
            lamAxisI, ThryI = irf.add_ion_IRF(self.cfg, lamAxisI, modlI, amps["i_amps"], TSins)
        else:
            # lamAxisI = jnp.nan
            ThryI = modlI  # jnp.nan

        if self.cfg["other"]["extraoptions"]["load_ele_spec"] & (
            self.cfg["other"]["extraoptions"]["spectype"] == "angular_full"
        ):
            lamAxisE, ThryE = irf.add_ATS_IRF(self.cfg, self.sas, lamAxisE, modlE, amps["e_amps"], TSins, self.lam)
        elif self.cfg["other"]["extraoptions"]["load_ele_spec"]:
            lamAxisE, ThryE = irf.add_electron_IRF(self.cfg, lamAxisE, modlE, amps["e_amps"], TSins, self.lam)
        else:
            # lamAxisE = jnp.nan
            ThryE = modlE  # jnp.nan

        return ThryE, ThryI, lamAxisE, lamAxisI

    def reduce_ATS_to_resunit(self, ThryE, lamAxisE, TSins, batch):
        lam_step = round(ThryE.shape[1] / batch["e_data"].shape[1])
        ang_step = round(ThryE.shape[0] / self.cfg["other"]["CCDsize"][0])

        ThryE = jnp.array([jnp.average(ThryE[:, i : i + lam_step], axis=1) for i in range(0, ThryE.shape[1], lam_step)])
        ThryE = jnp.array([jnp.average(ThryE[:, i : i + ang_step], axis=1) for i in range(0, ThryE.shape[1], ang_step)])

        lamAxisE = jnp.array(
            [jnp.average(lamAxisE[i : i + lam_step], axis=0) for i in range(0, lamAxisE.shape[0], lam_step)]
        )
        ThryE = ThryE[self.cfg["data"]["lineouts"]["start"] : self.cfg["data"]["lineouts"]["end"], :]
        ThryE = batch["e_amps"] * ThryE / jnp.amax(ThryE, axis=1, keepdims=True)
        ThryE = jnp.where(lamAxisE < self.lam, TSins["amp1"]["val"] * ThryE, TSins["amp2"]["val"] * ThryE)
        return ThryE, lamAxisE

    def __call__(self, params, batch):
        modlE, modlI, lamAxisE, lamAxisI, live_TSinputs = self.vmap_forward_pass(params)  # , sas["weights"])
        ThryE, ThryI, lamAxisE, lamAxisI = self.vmap_postprocess_thry(
            modlE, modlI, lamAxisE, lamAxisI, {"e_amps": batch["e_amps"], "i_amps": batch["i_amps"]}, live_TSinputs
        )
        if self.cfg["other"]["extraoptions"]["spectype"] == "angular_full":
            ThryE, lamAxisE = self.reduce_ATS_to_resunit(ThryE, lamAxisE, live_TSinputs, batch)

        ThryE = ThryE + batch["noise_e"]
        ThryI = ThryI + batch["noise_i"]

        return ThryE, ThryI, lamAxisE, lamAxisI
