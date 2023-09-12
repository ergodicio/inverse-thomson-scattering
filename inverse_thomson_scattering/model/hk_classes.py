from typing import Dict
from collections import defaultdict
import copy

import haiku as hk
import jax
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

        dv = cfg["velocity"][1] - cfg["velocity"][0]
        self.cfg_for_params = copy.deepcopy(cfg)
        self.cfg_for_params_2 = copy.deepcopy(cfg)

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

    def initialize_rest_of_params(self, config):
        # print("in initialize_rest_of_params")
        these_params = dict()
        for param_name, param_config in config["parameters"].items():
            if param_config["active"]:
                pass
            else:
                these_params[param_name] = jnp.concatenate(
                    [jnp.array(param_config["val"]).reshape(1, -1) for _ in range(config["optimizer"]["batch_size"])]
                ).reshape(config["optimizer"]["batch_size"], -1)

        return these_params

    # def get_normed_e_and_i_data(batch):
    #    normed_i_data = batch["i_data"] / i_input_norm
    #    normed_e_data = batch["e_data"] / e_input_norm

    #    return normed_e_data, normed_i_data

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


class TSParameterGenerator(hk.Module):
    def __init__(self, cfg: Dict, num_spectra: int = 2):
        super(TSParameterGenerator, self).__init__()
        self.cfg = cfg
        self.num_spectra = num_spectra
        self.batch_size = cfg["optimizer"]["batch_size"]
        if cfg["nn"]["use"]:
            self.nn_reparameterizer = nn.Reparameterizer(cfg, num_spectra)

        self.crop_window = cfg["other"]["crop_window"]

        self.smooth_window_len = round(cfg["velocity"].size * cfg["dist_fit"]["window"]["len"])
        self.smooth_window_len = self.smooth_window_len if self.smooth_window_len > 1 else 2

        if cfg["dist_fit"]["window"]["type"] == "hamming":
            self.w = jnp.hamming(self.smooth_window_len)
        elif cfg["dist_fit"]["window"]["type"] == "hann":
            self.w = jnp.hanning(self.smooth_window_len)
        elif cfg["dist_fit"]["window"]["type"] == "bartlett":
            self.w = jnp.bartlett(self.smooth_window_len)
        else:
            raise NotImplementedError

    def _init_nn_params_(self, batch):
        all_params = self.nn_reparameterizer(batch[:, :, self.crop_window : -self.crop_window])
        these_params = defaultdict(list)
        for i_slice in range(self.batch_size):
            # unpack all params which is an array that came out of the NN and into a dictionary that contains
            # the parameter names
            i = 0
            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name].append(all_params[i_slice, i].reshape((1, 1)))
                    i = i + 1
                else:
                    these_params[param_name].append(jnp.array(param_config["val"]).reshape((1, -1)))

        for param_name, param_config in self.cfg["parameters"].items():
            these_params[param_name] = jnp.concatenate(these_params[param_name])

        return these_params

    def _init_params_(self):
        these_params = dict()
        for param_name, param_config in self.cfg["parameters"].items():
            if param_config["active"]:
                if param_name == "fe":
                    these_params[param_name] = hk.get_parameter(
                        param_name,
                        shape=[self.batch_size, param_config["length"]],
                        init=hk.initializers.RandomUniform(minval=0, maxval=1),
                    )
                else:
                    these_params[param_name] = hk.get_parameter(
                        param_name,
                        shape=[self.batch_size, 1],
                        init=hk.initializers.RandomUniform(minval=0, maxval=1),
                    )
            else:
                these_params[param_name] = jnp.concatenate(
                    [jnp.array(param_config["val"]).reshape(1, -1) for _ in range(self.batch_size)]
                ).reshape(self.batch_size, -1)

        return these_params

    def get_active_params(self, batch):
        # print("in get_active_params")
        if self.cfg["nn"]["use"]:
            all_params = self.nn_reparameterizer(batch["data"][:, :, self.crop_window : -self.crop_window])
            these_params = defaultdict(list)
            for i_slice in range(self.batch_size):
                # unpack all params which is an array that came out of the NN and into a dictionary that contains
                # the parameter names
                i = 0
                for param_name, param_config in self.cfg["parameters"].items():
                    if param_config["active"]:
                        these_params[param_name].append(all_params[i_slice, i].reshape((1, 1)))
                        i = i + 1

            for param_name, param_config in these_params.items():
                these_params[param_name] = jnp.concatenate(these_params[param_name])

        else:
            these_params = dict()
            for param_name, param_config in self.cfg["parameters"].items():
                if param_config["active"]:
                    these_params[param_name] = hk.get_parameter(
                        param_name,
                        shape=[self.batch_size, 1],
                        init=hk.initializers.RandomUniform(minval=0, maxval=1),
                    )
                    these_params[param_name] = (
                        these_params[param_name] * self.cfg["units"]["norms"][param_name]
                        + self.cfg["units"]["shifts"][param_name]
                    )

        return these_params

    def smooth(self, distribution):
        s = jnp.r_[
            distribution[self.smooth_window_len - 1 : 0 : -1],
            distribution,
            distribution[-2 : -self.smooth_window_len - 1 : -1],
        ]
        return jnp.convolve(self.w / self.w.sum(), s, mode="same")[
            self.smooth_window_len - 1 : -(self.smooth_window_len - 1)
        ]

    def __call__(self, batch):
        if self.cfg["nn"]["use"]:
            these_params = self._init_nn_params_(batch["data"])
        else:
            these_params = self._init_params_()

        for param_name, param_config in self.cfg["parameters"].items():
            if param_config["active"]:
                these_params[param_name] = (
                    these_params[param_name] * self.cfg["units"]["norms"][param_name]
                    + self.cfg["units"]["shifts"][param_name]
                )
                if param_name == "fe":
                    these_params["fe"] = jnp.log(self.smooth(jnp.exp(these_params["fe"][0]))[None, :])

        return these_params


class Reparameterizer(hk.Module):
    def __init__(self, cfg, num_spectra):
        super(Reparameterizer, self).__init__()
        self.cfg = cfg
        self.num_spectra = num_spectra
        self.nn = cfg["nn"]
        convs = [int(i) for i in cfg["nn"]["conv_filters"].split("|")]
        widths = (
            [cfg["nn"]["linear_widths"]]
            if isinstance(cfg["nn"]["linear_widths"], int)
            else cfg["nn"]["linear_widths"].split("|")
        )
        linears = [int(i) for i in widths]

        num_outputs = 0
        for k, v in self.cfg["parameters"].items():
            if v["active"]:
                num_outputs += 1
        if self.nn == "resnet":
            self.embedding_generators = defaultdict(list)
            for i in range(num_spectra):
                res_blocks = []
                down_convs = []
                first_convs = []
                for cc in convs:
                    first_convs.append(hk.Sequential([hk.Conv1D(cc, 3, padding="same"), jax.nn.tanh]))
                    res_blocks.append(
                        hk.Sequential(
                            [hk.Conv1D(output_channels=cc, kernel_shape=3, stride=1, padding="same"), jax.nn.tanh]
                        )
                    )
                    down_convs.append(
                        hk.Sequential([hk.Conv1D(output_channels=cc, kernel_shape=3, stride=1), jax.nn.tanh])
                    )
                self.embedding_generators["first_convs"].append(first_convs)
                self.embedding_generators["res_blocks"].append(res_blocks)
                self.embedding_generators["down_convs"].append(down_convs)
                self.embedding_generators["final"].append(hk.Sequential([hk.Conv1D(1, 3), jax.nn.tanh]))
            self.combiner = hk.nets.MLP(linears + [num_outputs], activation=jax.nn.tanh)

        else:
            self.param_extractors = []
            for i in range(num_spectra):
                layers = []

                for cc in convs:
                    layers.append(hk.Conv1D(output_channels=cc, kernel_shape=3, stride=1))
                    layers.append(jax.nn.tanh)

                # layers.append(hk.Conv1D(1, 3))
                # layers.append(jax.nn.tanh)

                layers.append(hk.Flatten())
                for ll in linears:
                    layers.append(hk.Linear(ll))
                    layers.append(jax.nn.tanh)

                self.param_extractors.append(hk.Sequential(layers))
            self.combiner = hk.Linear(num_outputs)

    def __call__(self, spectra: jnp.ndarray):
        if self.nn == "resnet":
            embedding = []
            for i_spec, (res_blocks, down_convs, first_conv) in enumerate(
                zip(
                    self.embedding_generators["res_blocks"],
                    self.embedding_generators["down_convs"],
                    self.embedding_generators["first_convs"],
                )
            ):
                out = spectra[:, i_spec, :, None]
                for res_block, down_conv in zip(res_blocks, down_convs):
                    out = first_conv(out)
                    temp_out = res_block(out)
                    out = down_conv(temp_out + out)
                embedding.append(out)
            return jax.nn.sigmoid(self.combiner(hk.Flatten()(embedding)))
        else:
            embeddings = jnp.concatenate(
                [self.param_extractors[i](spectra[:, i][..., None]) for i in range(self.num_spectra)], axis=-1
            )
            return jax.nn.sigmoid(self.combiner(embeddings))
