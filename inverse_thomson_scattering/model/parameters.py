from typing import Dict
from collections import defaultdict

import haiku as hk
import jax
from jax import numpy as jnp


class TSParameterGenerator(hk.Module):
    def __init__(self, cfg: Dict, num_spectra: int = 2):
        super(TSParameterGenerator, self).__init__()
        self.cfg = cfg
        self.num_spectra = num_spectra
        self.batch_size = cfg["optimizer"]["batch_size"]
        if cfg["nn"]["use"]:
            self.nn_reparameterizer = Reparameterizer(cfg, num_spectra)

        self.crop_window = cfg["other"]["crop_window"]

        if "dist_fit" in cfg:
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
        else:
            print("\n !!! Distribution function not fitted !!! Make sure this is what you thought you were running \n")

    def _init_nn_params_(self, batch):
        nn_batch = jnp.concatenate([batch["e_data"][:, None, :], batch["e_data"][:, None, :]], axis=1)

        all_params = self.nn_reparameterizer(nn_batch[:, :, self.crop_window : -self.crop_window])
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

    def _init_params_(self, get_active=False):
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
                if ~get_active:
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
