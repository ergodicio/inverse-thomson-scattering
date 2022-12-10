import yaml, time
from jax import config

config.update("jax_enable_x64", True)
import multiprocessing as mp
from flatten_dict import flatten, unflatten
import mlflow
from inverse_thomson_scattering import datafitter, utils


def test_data():
    """
    Test #3: Data test, compare fit to a preknown fit result
    currently just runs one line of shot 101675 for the electron, should be expanded in the future

    Returns:

    """
    with open("./defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("./inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    bgshot = {"type": [], "val": []}
    # bgshot = {"type": "Fit", "val": 102584}
    lnout = {"type": "pixel", "val": [500]}
    # lnout = {"type": "um", "val": slices}
    bglnout = {"type": "pixel", "val": 900}
    extraoptions = {"spectype": 2}

    config["parameters"]["Te"]["val"] = 0.5
    config["parameters"]["ne"]["val"] = 0.2  # 0.25
    config["parameters"]["m"]["val"] = 3.0  # 2.2

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        utils.log_params(config)

        config["bgshot"] = bgshot
        config["lineoutloc"] = lnout
        config["bgloc"] = bglnout
        config["extraoptions"] = extraoptions
        config["num_cores"] = int(mp.cpu_count())

        config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

        mlflow.log_params({"num_slices": 1})
        t0 = time.time()
        # mlflow.log_params(flatten(config))
        fit_results = datafitter.fit(config=config)
        metrics_dict = {"datafitter_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")

        # print((fit_results["amp1"]["val"]-0.9257)/0.9257)
        # print((fit_results["amp2"]["val"]-0.6727)/0.6727)
        # print((fit_results["lam"]["val"]-524.2455)/524.2455)
        # print((fit_results["Te"]["val"]-0.67585)/0.67585)
        # print((fit_results["ne"]["val"]-0.21792)/0.21792)
        # print((fit_results["m"]["val"]-3.3673)/3.3673)
        if (
            (fit_results["amp1"]["val"] - 0.9257) / 0.9257 < 0.05
            and (fit_results["amp2"]["val"] - 0.6727) / 0.6727 < 0.05
            and (fit_results["lam"]["val"] - 524.2455) / 524.2455 < 0.05
            and (fit_results["Te"]["val"] - 0.67585) / 0.67585 < 0.05
            and (fit_results["ne"]["val"] - 0.21792) / 0.21792 < 0.05
            and (fit_results["m"]["val"] - 3.3673) / 3.3673 < 0.1
        ):
            test3 = True
            print("Fit values are within 5-10% of known values")
        else:
            test3 = False
            print("Fit values do NOT agree with known values")
