from matplotlib import pyplot as plt
import numpy as np


def make_plots(formf, formf_jax):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
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
