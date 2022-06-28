# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import time
import matplotlib.pyplot as plt
from inverse_thomson_scattering.v0 import form_factor as np_ff
from inverse_thomson_scattering.jax import form_factor as jnp_ff
import numpy as np


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
    formf, lams = ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    val, grad = vg_ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))

    # then run them again to benchmark them
    # TODO: find a better way to measure this
    t0 = time.time()
    formf, lams = ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    t1 = time.time()
    print(f"jax form factor calculation took {np.round(t1 - t0, 4)} s")

    t0 = time.time()
    val, grad = vg_ff_fn(1.0, 1.0, 1.0, 1.0, 1.0, 0.3e20, 0.0, 0.0, sa, (distf, x))
    t1 = time.time()
    print(f"value and gradient took {np.round(t1 - t0, 4)} s")

    print(f"gradient was {grad}")

    plt.plot(formf[0, :, 0])
    plt.plot(formf[0, :, 9])
    # plt.plot(lams[0,:,0],formf[0,:,1])
    plt.show()
    print("end")
