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
    
    for batch_size in range(2, 8):
        for i in range(1,4):

            with open("defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)
            mlflow.set_experiment(config["mlflow"]["experiment"])

            config["data"]["lineouts"]["start"] = int(320)
            config["data"]["lineouts"]["skip"] = int(1)
            config["optimizer"]["batch_size"] = batch_size
            config["data"]["lineouts"]["end"] = int(320+np.floor(360/batch_size)*batch_size)
            
            # config["optimizer"]["method"] = "adam"
            config["nn"]["use"] = False
            # config["optimizer"]["grad_method"] = str(dd)
            config["mlflow"]["run"] = f"batch_size={batch_size}-run={i}"
            one_run(config)

    # raise ValueError
