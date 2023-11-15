import tempfile, time, os, yaml
import multiprocessing as mp
import mlflow
from flatten_dict import flatten, unflatten
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_disable_jit", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils


def one_run(config):
    utils.log_params(config)
    t0 = time.time()
    fit_results, loss = fitter.fit(config=config)
    metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
    mlflow.log_metrics(metrics=metrics_dict)
    mlflow.set_tag("status", "completed")

    return loss


if __name__ == "__main__":
    all_configs = {}
    basedir = os.path.join(os.getcwd(), "configs", "arts")
    for k in ["defaults", "inputs"]:
        with open(f"{os.path.join(basedir, k)}.yaml", "r") as fi:
            all_configs[k] = yaml.safe_load(fi)

    if "mlflow" in all_configs["inputs"].keys():
        experiment = all_configs["inputs"]["mlflow"]["experiment"]
        run_name = all_configs["inputs"]["mlflow"]["run"]

    else:
        experiment = all_configs["defaults"]["mlflow"]["experiment"]
        run_name = all_configs["defaults"]["mlflow"]["run"]

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

        defaults = flatten(all_configs["defaults"])
        defaults.update(flatten(all_configs["inputs"]))
        config = unflatten(defaults)

        one_run(config)

    if "MLFLOW_EXPORT" in os.environ:
        from mlflow_export_import.run.export_run import RunExporter

        t0 = time.time()
        run_exp = RunExporter(mlflow_client=mlflow.MlflowClient())
        with tempfile.TemporaryDirectory() as td2:
            run_exp.export_run(run.info.run_id, td2)
            print(f"Export took {round(time.time() - t0, 2)} s")
            t0 = time.time()
            utils.upload_dir_to_s3(td2, "remote-mlflow-staging", f"artifacts/{run.info.run_id}", run.info.run_id)
        print(f"Uploading took {round(time.time() - t0, 2)} s")
