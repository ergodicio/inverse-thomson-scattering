from collections import defaultdict

import jax
from jax import numpy as jnp


import haiku as hk


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
