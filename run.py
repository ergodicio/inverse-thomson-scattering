import time
import multiprocessing as mp
import yaml
import numpy as np
import mlflow
from flatten_dict import flatten, unflatten
from jax import config

config.update("jax_enable_x64", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils

if __name__ == "__main__":

    numtimes = 1
    starttimes = np.linspace(1600, 3700, numtimes + 1)
    ms = [3.0, 3.0, 3.0, 3.5, 3.0, 3.0, 2.5, 2.5, 2.5, 2.5]
    for ii in range(numtimes):
        tstart = starttimes[ii]
        tend = starttimes[ii + 1]

        for num_slices in [2]:#[8]:  # [1, 2, 4, 8, 16, 32][::-1]:
            slices = [int(i) for i in np.linspace(tstart, tend, num_slices)]
            slices = slices[:-1]

            with open("./defaults.yaml", "r") as fi:
                defaults = yaml.safe_load(fi)

            with open("./inputs.yaml", "r") as fi:
                inputs = yaml.safe_load(fi)

            defaults = flatten(defaults)
            defaults.update(flatten(inputs))
            config = unflatten(defaults)

            #bgshot = {"type": [], "val": []}
            #bgshot = {"type": "Fit", "val": []}
            bgshot = {"type": "Fit", "val": 94477}
            #lnout = {"type": "ps", "val": slices}
            #lnout = {"type": "um", "val": slices}
            #lnout = {"type": "pixel", "val": [300, 500, 700]}
            lnout = {"type": "range", "val": [90, 1015]} #new unique option for ARTS defines the range over which to perform the full matching
            bglnout = {"type": "pixel", "val": 900, "show": False}

            #config["parameters"]["Te"]["val"]= list(np.interp(slices, np.linspace(1600,3700,19), [.2,.4,.5,.55,.6,.6,.65,.65,.65,.65,.65,.5,.4,.4,.3,.3,.25,.2,.2]))
            #config["parameters"]["ne"]["val"]= list(np.interp(slices, np.linspace(1600,3700,19), [.15,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.15,.15,.15,.15]))
            #config["parameters"]["m"]["val"]= np.array(np.interp(slices, np.linspace(1600,3700,19), [2.2,3.,3.,3.,3.,3.,3.5,3.5,3.5,3.,3.,3.,3.,2.5,2.5,2.5,2.5,2.5,2.5]))
            #config["parameters"]["m"]["val"]=ms[ii]

            mlflow.set_experiment(config["mlflow"]["experiment"])

            with mlflow.start_run() as run:
                utils.log_params(config)
                # with tempfile.TemporaryDirectory() as td:
                #    with open(os.path.join(td, "config.yaml"), "w") as fi:
                #        yaml.safe_dump(config, fi)

                config["bgshot"] = bgshot
                config["lineoutloc"] = lnout
                config["bgloc"] = bglnout
                config["num_cores"] = int(mp.cpu_count())

                config = {**config, **dict(shotnum=94475, dpixel=2)}
                #config = {**config, **dict(shotnum=101675, dpixel=2)}

                mlflow.log_params({"num_slices": len(slices)})
                t0 = time.time()
                # mlflow.log_params(flatten(config))
                fit_results = fitter.fit(config=config)
                metrics_dict = {"fit_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
                mlflow.log_metrics(metrics=metrics_dict)
                mlflow.set_tag("status", "completed")
