import tempfile, os, yaml, argparse
import mlflow
from flatten_dict import flatten, unflatten
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)

from inverse_thomson_scattering.misc import utils
from inverse_thomson_scattering.runner import run


def _run_(cfg_path, mode="fit"):
    all_configs = {}
    basedir = os.path.join(os.getcwd(), f"{cfg_path}")
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

    with mlflow.start_run(run_name=run_name) as mlflow_run:
        with tempfile.TemporaryDirectory() as td:
            for k in ["defaults", "inputs"]:
                with open(os.path.join(td, f"{k}.yaml"), "w") as fi:
                    yaml.dump(all_configs[k], fi)

            mlflow.log_artifacts(td)

        defaults = flatten(all_configs["defaults"])
        defaults.update(flatten(all_configs["inputs"]))
        config = unflatten(defaults)

        run(config, mode=mode)

    if "MLFLOW_EXPORT" in os.environ:
        utils.export_run(mlflow_run.info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSADAR - Thomson Scattering with Automatic Differentiation")
    parser.add_argument("--cfg", help="enter path to cfg")
    parser.add_argument("--mode", help="enter forward or fit")
    args = parser.parse_args()

    _run_(args.cfg, args.mode)
