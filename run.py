import time

import yaml
import mlflow

from inverse_thomson_scattering.v0 import datafitter


if __name__ == "__main__":
    with open("./defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("./inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    bgshot = {"type": [], "val": []}
    lnout = {"type": "ps", "val": [2000]}
    bglnout = {"type": "pixel", "val": 900}
    extraoptions = {"spectype": 2}

    config = defaults.update(inputs)
    config["bgshot"] = bgshot
    config["lnout"] = lnout
    config["bglnout"] = bglnout
    config["extraoptions"] = extraoptions
    config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        t0 = time.time()
        fit_results = datafitter.fit(config=config)
        metrics_dict = {"datafitter_time": time.time() - t0, "loss": fit_results["loss"]}

        mlflow.log_params(config)
        mlflow.log_metrics(metrics=metrics_dict)
