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

    if backend == "numpy":
        [formf, lams] = np_ff.nonMaxwThomson(1, 1, 1, 1, 1, 0.3e20, 0, 0, [400, 700], 526.5, sa, distf, x)
    elif backend == "jax":
        ff_fn = jnp_ff.get_form_factor_fn([400, 700], 526.5)
        [formf, lams] = ff_fn(1, 1, 1, 1, 1, 0.3e20, 0, 0, sa, (distf, x))

    t0 = time.time()
    [formf, lams] = np_ff.nonMaxwThomson(1, 1, 1, 1, 1, 0.3e20, 0, 0, [400, 700], 526.5, sa, distf, x)
    # [formf,lams]=nonMaxwThomson(1,1,[1,2],[1,4],[.5, .5],.3e20,0,0,[400,700],526.5,sa,distf,x)
    t1 = time.time()
    # print(formf[0,:,1])
    print(t1 - t0)
    plt.plot(formf[0, :, 0])
    plt.plot(formf[0, :, 9])
    # plt.plot(lams[0,:,0],formf[0,:,1])
    plt.show()
    print("end")
