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
        metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")


if __name__ == "__main__":
    
    for batch_size in range(2, 8):
        for i in range(1,11):

            with open("defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)
            mlflow.set_experiment(config["mlflow"]["experiment"])
            
            config["parameters"]["Te"]["val"] = 1.1*np.random.random_sample()+0.3
            config["parameters"]["ne"]["val"] = 0.4*np.random.random_sample()+0.1
            config["parameters"]["m"]["val"] = 3.0*np.random.random_sample()+2.0

            config["data"]["lineouts"]["start"] = int(320)
            config["data"]["lineouts"]["skip"] = int(1)
            config["optimizer"]["batch_size"] = batch_size
            config["data"]["lineouts"]["end"] = int(320+np.floor(360/batch_size)*batch_size)
            
            # config["optimizer"]["method"] = "adam"
            config["nn"]["use"] = False
            # config["optimizer"]["grad_method"] = str(dd)
            config["mlflow"]["run"] = f"random_start_convergence test_batch_size={batch_size}-run={i}"
            one_run(config)

    # raise ValueError
