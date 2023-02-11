import time
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils


def one_run(config):
    with mlflow.start_run(run_name=config["mlflow"]["run"]) as run:
        utils.log_params(config)
        t0 = time.time()
        fit_results = fitter.fit(config=config)
        metrics_dict = {"fit_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")


if __name__ == "__main__":

    for tend in np.arange(502, 510, 2):
        for dd in ["FD", "AD"]:
            with open("tests/configs/defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("tests/configs/inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)
            mlflow.set_experiment(config["mlflow"]["experiment"])

            config["parameters"]["Te"]["val"] = 0.5
            config["parameters"]["ne"]["val"] = 0.2  # 0.25
            config["parameters"]["m"]["val"] = 3.0  # 2.2
            config["data"]["lineouts"]["end"] = int(tend)
            config["optimizer"]["batch_size"] = int(tend - 500)
            config["optimizer"]["grad_method"] = str(dd)
            config["mlflow"]["run"] = f"cpu-{dd}-tend={tend}"
            one_run(config)

        # raise ValueError

