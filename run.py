import time
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_check_tracer_leaks", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils


def test_data():
    with open("tests/configs/defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    bgshot = {"type": [], "val": []}
    lnout = {"type": "pixel", "val": [i for i in range(500, 550, 16)]}
    bglnout = {"type": "pixel", "val": 900}
    extraoptions = {"spectype": 2}

    config["parameters"]["Te"]["val"] = 0.5
    config["parameters"]["ne"]["val"] = 0.2  # 0.25
    config["parameters"]["m"]["val"] = 3.0  # 2.2

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        config["bgshot"] = bgshot
        config["lineoutloc"] = lnout
        config["bgloc"] = bglnout
        config["extraoptions"] = extraoptions
        config["num_cores"] = int(mp.cpu_count())

        config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

        utils.log_params(config)

        t0 = time.time()
        # mlflow.log_params(flatten(config))
        fit_results = fitter.fit(config=config)
        metrics_dict = {"fit_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")

        # assert_allclose(fit_results["amp1"], 0.9257, rtol=1e-1)
        # assert_allclose(fit_results["amp2"], 0.6727, rtol=1e-1)  # 0.98734!
        # assert_allclose(fit_results["lam"], 524.2455, rtol=5e-3)
        # assert_allclose(fit_results["Te"], 0.67585, rtol=1e-1)  # 0.57567
        # assert_allclose(fit_results["ne"], 0.21792, rtol=5e-2)
        # assert_allclose(fit_results["m"], 3.3673, rtol=15e-2)


if __name__ == "__main__":
    test_data()
