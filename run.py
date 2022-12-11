import time
import multiprocessing as mp
import yaml
import numpy as np
import mlflow
from flatten_dict import flatten, unflatten
from jax import config

config.update("jax_enable_x64", True)

from inverse_thomson_scattering import datafitter, utils

if __name__ == "__main__":

    numtimes = 10
    starttimes = np.linspace(1600, 3700, numtimes + 1)
    ms = [3.0, 3.0, 3.0, 3.5, 3.0, 3.0, 2.5, 2.5, 2.5, 2.5]
    for ii in range(numtimes):
        tstart = starttimes[ii]
        tend = starttimes[ii + 1]

        for num_slices in [8]:  # [1, 2, 4, 8, 16, 32][::-1]:
            slices = [int(i) for i in np.linspace(tstart, tend, num_slices)]
            slices = slices[:-1]

            with open("./defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("./inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)

            bgshot = {"type": [], "val": []}
            # bgshot = {"type": "Fit", "val": 102584}
            lnout = {"type": "ps", "val": slices}
            # lnout = {"type": "um", "val": slices}
            bglnout = {"type": "pixel", "val": 900}
            extraoptions = {"spectype": 2}

            config["parameters"]["Te"]["val"] = list(
                np.interp(
                    slices,
                    np.linspace(1600, 3700, 19),
                    [
                        0.2,
                        0.4,
                        0.5,
                        0.55,
                        0.6,
                        0.6,
                        0.65,
                        0.65,
                        0.65,
                        0.65,
                        0.65,
                        0.5,
                        0.4,
                        0.4,
                        0.3,
                        0.3,
                        0.25,
                        0.2,
                        0.2,
                    ],
                )
            )
            config["parameters"]["ne"]["val"] = list(
                np.interp(
                    slices,
                    np.linspace(1600, 3700, 19),
                    [
                        0.15,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.2,
                        0.15,
                        0.15,
                        0.15,
                        0.15,
                    ],
                )
            )
            # config["parameters"]["m"]["val"]= np.array(np.interp(slices, np.linspace(1600,3700,19), [2.2,3.,3.,3.,3.,3.,3.5,3.5,3.5,3.,3.,3.,3.,2.5,2.5,2.5,2.5,2.5,2.5]))
            config["parameters"]["m"]["val"] = ms[ii]

            mlflow.set_experiment(config["mlflow"]["experiment"])

            with mlflow.start_run() as run:
                utils.log_params(config)
                # with tempfile.TemporaryDirectory() as td:
                #    with open(os.path.join(td, "config.yaml"), "w") as fi:
                #        yaml.safe_dump(config, fi)

                config["bgshot"] = bgshot
                config["lineoutloc"] = lnout
                config["bgloc"] = bglnout
                config["extraoptions"] = extraoptions
                config["num_cores"] = int(mp.cpu_count())

                config = {**config, **dict(shotnum=101675, bgscale=1, dpixel=2)}

                mlflow.log_params({"num_slices": len(slices)})
                t0 = time.time()
                # mlflow.log_params(flatten(config))
                fit_results = datafitter.fit(config=config)
                metrics_dict = {"datafitter_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
                mlflow.log_metrics(metrics=metrics_dict)
                mlflow.set_tag("status", "completed")
