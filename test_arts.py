import time
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
import numpy as np
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_disable_jit", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils


def one_run(config):
    with mlflow.start_run(run_name=config["mlflow"]["run"]) as run:
        utils.log_params(config)
        t0 = time.time()
        fit_results = fitter.fit(config=config)
        #config["parameters"]["fe"]["val"] = fit_results["fe"]
        #config["parameters"]["fe"]["length"] = 2*config["parameters"]["fe"]["length"]
        #refined_v = np.linspace(-7, 7, config["parameters"]["fe"]["length"])
        #refined_fe = np.interp(config["velocity"], refined_v, config["parameters"]["fe"]["val"])    
        #config["parameters"]["fe"]["val"] = refined_fe
        #config["velocity"] = refined_v
        metrics_dict = {"fit_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")


if __name__ == "__main__":
    # for dd in ["FD", "AD"]:
    # for tend in np.arange(510, 560, 4):
    # for nn_reparam in [False, True]:
    nn_reparam = True
    with open("defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)
    mlflow.set_experiment(config["mlflow"]["experiment"])
    config["data"]["shotnum"] = 94475
    config["other"]["crop_window"] = 1

    config["parameters"]["lam"]["val"] = 524.5
    config["parameters"]["Te"]["val"] = 1.0
    config["parameters"]["ne"]["val"] = 0.5  # 0.25
    config["parameters"]["m"]["val"] = 3.5  # 2.2

    config["parameters"]["m"]["active"] = False
    config["parameters"]["fe"]["active"] = True
    config["parameters"]["fe"]["ub"] = -0.5
    config["parameters"]["fe"]["lb"] = -50
    config["parameters"]["fe"]["length"] = 80

    config["data"]["lineouts"]["type"] = "range"
    config["data"]["lineouts"]["start"] = 90
    config["data"]["lineouts"]["end"] = 1015

    config["data"]["background"]["type"] = "Fit"
    config["data"]["background"]["slice"] = 94477

    # config["data"]["lineouts"]["start"] = int(320)
    # config["data"]["lineouts"]["skip"] = int(18)
    # config["data"]["lineouts"]["end"] = int(680)
    config["optimizer"]["batch_size"] = int(925)  # this should be automatic
    config["optimizer"]["method"] = "adam"
    config["nn"]["use"] = False
    # config["optimizer"]["grad_method"] = str(dd)
    config["mlflow"]["run"] = f"arts_160pts_normed_2fit_TNC-LBFGSB_refine"
    one_run(config)

    # raise ValueError
