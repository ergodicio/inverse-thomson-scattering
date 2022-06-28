# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import matplotlib.pyplot as plt
from inverse_thomson_scattering.v0 import form_factor as np_ff
from inverse_thomson_scattering.jax import form_factor as jnp_ff
import numpy as np


def make_plots():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(formf[0, :, 0], label="np/sp")
    ax[0].plot(formf_jax[0, :, 0], label="jax")
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(formf[0, :, 9], label="np/sp")
    ax[1].plot(formf_jax[0, :, 9], label="jax")
    ax[1].grid()
    ax[1].legend()
    print(f"L2 norm between jax and np/sp is {np.sqrt(np.sum((formf - formf_jax) ** 2.0))}")
    # plt.plot(lams[0,:,0],formf[0,:,1])
    plt.show()


if __name__ == "__main__":
    x = np.array(np.arange(-8, 8, 0.1))
    distf = 1 / (2 * np.pi) ** (1 / 2) * np.exp(-(x**2) / 2)
    sa = np.linspace(55, 65, 10)
    backend = "jax"

    # if backend == "numpy":
    t0 = time.time()
    formf, lams = np_ff.nonMaxwThomson(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, [400, 700], 526.5, sa, distf, x)
    t1 = time.time()
    print(f"numpy/scipy form factor calculation {np.round(t1 - t0, 4)} s")

    # elif backend == "jax":
    # get the functions
    ff_fn, vg_ff_fn = jnp_ff.get_form_factor_fn([400, 700], 526.5)

    # run them once so they're compiled
    _ = ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    _ = vg_ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))

    # then run them again to benchmark them
    # TODO: find a better way to measure this
    t0 = time.time()
    formf_jax, lams_jax = ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    t1 = time.time()
    print(f"jax form factor calculation took {np.round(t1 - t0, 4)} s")

    t0 = time.time()
    val, grad = vg_ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    t1 = time.time()
    print(f"value and gradient took {np.round(t1 - t0, 4)} s")
    print(f"gradient was {grad}")
    make_plots()
    print("end")
