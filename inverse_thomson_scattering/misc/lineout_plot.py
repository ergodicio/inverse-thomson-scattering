import matplotlib.pyplot as plt
import numpy as np
import os


def lineout_plot(sorted_data, sorted_fits, sorted_sqdev, yaxis, s_ind, e_ind, titlestr, filename, td, tag):
    if len(sorted_data) == 2:
        num_col = 2
    else:
        num_col = 1

    fig, ax = plt.subplots(2, num_col, figsize=(12, 8), squeeze=False, tight_layout=True, sharex=False)
    for col in range(num_col):
        ax[0][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]], np.squeeze(sorted_data[col][s_ind[col] : e_ind[col]]), label="Data"
        )
        ax[0][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]], np.squeeze(sorted_fits[col][s_ind[col] : e_ind[col]]), label="Fit"
        )

        ax[0][col].set_title(titlestr, fontsize=14)
        ax[0][col].set_ylabel("Amp (arb. units)")
        ax[0][col].legend(fontsize=14)
        ax[0][col].grid()
        ax[0][col].set_ylim((0,1000))

        ax[1][col].plot(
            yaxis[col][s_ind[col] : e_ind[col]],
            np.squeeze(sorted_sqdev[col][s_ind[col] : e_ind[col]]),
            label="Residual",
        )
        ax[1][col].set_xlabel("Wavelength (nm)")
        ax[1][col].set_ylabel(r"$\chi_i^2$")

    fig.savefig(os.path.join(td, tag, filename), bbox_inches="tight")
    plt.close(fig)
