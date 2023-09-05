import time
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
from matplotlib import pyplot as plt
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
    for filt in ["hamming", "hann", "bartlett"]:
        for i in [3, 5, 9, 15, 25]:
            for ii in [1, 2]:

                with open("defaults.yaml", "r") as fi:
                    defaults = yaml.safe_load(fi)

                with open("inputs.yaml", "r") as fi:
                    inputs = yaml.safe_load(fi)

                defaults = flatten(defaults)
                defaults.update(flatten(inputs))
                config = unflatten(defaults)
                mlflow.set_experiment(config["mlflow"]["experiment"])
                
                config["dist_fit"]["window"]["len"] = i
                config["dist_fit"]["window"]["type"] = filt
                
                config["mlflow"]["run"] = f"arts_filter_scaling_{filt}_{i}-run={ii}"
                one_run(config)
                
                plt.close('all')

