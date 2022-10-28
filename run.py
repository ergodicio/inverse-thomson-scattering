import tempfile
import time
import multiprocessing as mp
import yaml
import numpy as np
import mlflow
import os
from flatten_dict import flatten, unflatten
import flatdict

from inverse_thomson_scattering.v0 import datafitter


def log_params(cfg):
    flattened_dict = dict(flatdict.FlatDict(cfg, delimiter="."))
    num_entries = len(flattened_dict.keys())

    if num_entries > 100:
        num_batches = num_entries % 100
        fl_list = list(flattened_dict.items())
        for i in range(num_batches):
            end_ind = min((i + 1) * 100, num_entries)
            trunc_dict = {k: v for k, v in fl_list[i * 100 : end_ind]}
            mlflow.log_params(trunc_dict)
    else:
        mlflow.log_params(flattened_dict)


def update(base_dict, new_dict):
    combined_dict = {}
    for k, v in new_dict.items():
        combined_dict[k] = base_dict[k]
        if isinstance(v, dict):
            combined_dict[k] = update(base_dict[k], v)
        else:
            combined_dict[k] = new_dict[k]

    return combined_dict


if __name__ == "__main__":

    for num_slices in [5]:#[1, 2, 4, 8, 16, 32][::-1]:
        slices = [int(i) for i in np.linspace(-800, 800, num_slices)]
        with open("./defaults.yaml", "r") as fi:
            defaults = yaml.safe_load(fi)

        with open("./inputs.yaml", "r") as fi:
            inputs = yaml.safe_load(fi)

        defaults = flatten(defaults)
        defaults.update(flatten(inputs))
        config = unflatten(defaults)
        bgshot = {"type": "Fit", "val": 102584}
        lnout = {"type": "um", "val": slices}
        bglnout = {"type": "pixel", "val": 900}
        extraoptions = {"spectype": 2}

        mlflow.set_experiment(config["mlflow"]["experiment"])

        with mlflow.start_run() as run:
            log_params(config)
            with tempfile.TemporaryDirectory() as td:
                with open(os.path.join(td, "config.yaml"), "w") as fi:
                    yaml.safe_dump(config, fi)

            config["bgshot"] = bgshot
            config["lineoutloc"] = lnout
            config["bgloc"] = bglnout
            config["extraoptions"] = extraoptions
            config["num_cores"] = int(mp.cpu_count())

            config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

            mlflow.log_params({"num_slices": len(slices)})
            t0 = time.time()
            fit_results = datafitter.fit(config=config)
            metrics_dict = {"datafitter_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
            mlflow.log_metrics(metrics=metrics_dict)
            mlflow.log_params(fit_results)
            mlflow.set_tag("status", "completed")
