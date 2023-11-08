import tempfile, time, os, yaml
import multiprocessing as mp
import mlflow
from flatten_dict import flatten, unflatten
from jax import config
from mlflow_export_import.run.export_run import RunExporter
from tqdm import tqdm
import boto3

config.update("jax_enable_x64", True)
#config.update("jax_disable_jit", True)

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

def upload_dir_to_s3(local_directory: str, bucket: str, destination: str, run_id: str):
    client = boto3.client("s3")

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        for filename in tqdm(files):
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            client.upload_file(local_path, bucket, s3_path)

    with open(os.path.join(local_directory, f"ingest-{run_id}.txt"), "w") as fi:
        fi.write("ready")

    client.upload_file(os.path.join(local_directory, f"ingest-{run_id}.txt"), bucket, f"ingest-{run_id}.txt")
    

if __name__ == "__main__":
    all_configs = {}
    basedir = os.path.join(os.getcwd(), "configs", "1d")
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

    t0 = time.time()
    run_exp = RunExporter(mlflow_client=mlflow.MlflowClient())
    with tempfile.TemporaryDirectory() as td2:
        run_exp.export_run(run.info.run_id, td2)
        print(f"Export took {round(time.time() - t0, 2)} s")
        t0 = time.time()
        upload_dir_to_s3(td2, "remote-mlflow-staging", f"artifacts/{run.info.run_id}", run.info.run_id)
    print(f"Uploading took {round(time.time() - t0, 2)} s")