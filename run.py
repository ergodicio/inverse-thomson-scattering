import time
import multiprocessing as mp
import yaml
import numpy as np
import mlflow
from flatten_dict import flatten, unflatten

from inverse_thomson_scattering.v0 import datafitter


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

        config["bgshot"] = bgshot
        config["lineoutloc"] = lnout
        config["bgloc"] = bglnout
        config["extraoptions"] = extraoptions
        config["num_cores"] = int(mp.cpu_count())

        config = {**config, **dict(shotnum=102583, bgscale=1, dpixel=2)}

        mlflow.set_experiment(config["mlflow"]["experiment"])

        with mlflow.start_run() as run:
            mlflow.log_params({"num_slices": len(slices)})
            t0 = time.time()
            fit_results = datafitter.fit(config=config)
            metrics_dict = {"datafitter_time": time.time() - t0, "num_cores": int(mp.cpu_count())}  # , "loss": fit_results["loss"]}
            mlflow.log_metrics(metrics=metrics_dict)
            mlflow.log_params(fit_results)
            mlflow.set_tag("status", "completed")
