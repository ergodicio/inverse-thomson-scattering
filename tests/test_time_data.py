import time, pytest
import multiprocessing as mp
import yaml
import mlflow
from flatten_dict import flatten, unflatten
from numpy.testing import assert_allclose
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_check_tracer_leaks", True)

from inverse_thomson_scattering import fitter
from inverse_thomson_scattering.misc import utils


@pytest.mark.parametrize("nn", [False])
def test_data(nn):
    # Test #3: Data test, compare fit to a preknown fit result
    # currently just runs one line of shot 101675 for the electron, should be expanded in the future

    with open("tests/configs/time_test_defaults.yaml", "r") as fi:
        defaults = yaml.safe_load(fi)

    with open("tests/configs/time_test_inputs.yaml", "r") as fi:
        inputs = yaml.safe_load(fi)

    defaults = flatten(defaults)
    defaults.update(flatten(inputs))
    config = unflatten(defaults)

    config["nn"]["use"] = nn
    # config["parameters"]["Te"]["val"] = 0.5
    # config["parameters"]["ne"]["val"] = 0.2  # 0.25
    # config["parameters"]["m"]["val"] = 3.0  # 2.2

    mlflow.set_experiment(config["mlflow"]["experiment"])

    with mlflow.start_run() as run:
        utils.log_params(config)
        config["num_cores"] = int(mp.cpu_count())

        t0 = time.time()
        fit_results, loss = fitter.fit(config=config)
        metrics_dict = {"total_time": time.time() - t0, "num_cores": int(mp.cpu_count())}
        mlflow.log_metrics(metrics=metrics_dict)
        mlflow.set_tag("status", "completed")

        assert_allclose(fit_results["amp1"][0], 0.9257, rtol=1e-1)
        assert_allclose(fit_results["amp2"][0], 0.6727, rtol=1e-1)  # 0.98734!
        assert_allclose(fit_results["lam"][0], 524.2455, rtol=5e-3)
        assert_allclose(fit_results["Te"][0], 0.67585, rtol=1e-1)  # 0.57567
        assert_allclose(fit_results["ne"][0], 0.21792, rtol=5e-2)
        assert_allclose(fit_results["m"][0], 3.3673, rtol=15e-2)


if __name__ == "__main__":
    test_data(False)
