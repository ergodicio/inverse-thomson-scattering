# import numpy as np
#
#
# def lamParse(lamrang, lam, *args):
#     """
#     This function handles the calculation of wavelength grids and their associated frequency grids. A few predefined ranges
#     are provided as string input options or a numeric range can be selected. A boolean switch is included to allow a finer
#     griding in the ion spectrum allowing for simultaneous resolution of the ion and electron frequencies.
#     """
#
#     c = 2.99792458e10
#     if isinstance(lamrang, str):
#         if lamrang == "EPWb":
#             rng = 59.2
#             off = 67.2
#             npts = 48000
#         elif lamrang == "EPWr":
#             rng = 150
#             off = -173.5
#             npts = 37910
#         elif lamrang == "Full":
#             rng = 450
#             off = 0
#             npts = 102500
#         elif lamrang == "EPWbNIF":
#             rng = 35
#             off = 45.6
#             npts = 20273
#
#         min_lam = lam - off - rng  # Minimum wavelength to run code over
#         max_lam = lam - off + rng  # Max wavelength
#
#     else:
#         min_lam = lamrang[0]
#         max_lam = lamrang[1]
#         npts = 102500
#
#     fineion = False
#     if len(args) != 0:
#         npts = args[0]
#         if len(args) > 1:
#             fineion = args[1]
#
#     if fineion and (min_lam < lam and max_lam > lam):
#         lamAxis = np.linspace(min_lam, max_lam, npts)
#         L = next(i for i in range(len(lamAxis)) if lamAxis[i] >= lam - 2)
#         # print(L)
#         # L = find(lamAxis >= lam - 2, 1, 'first');
#         rlamAxis = lamAxis[::-1]
#         R = np.abs(next(i for i in range(len(lamAxis)) if rlamAxis[i] <= lam + 2) - len(lamAxis))
#         # R = find(lamAxis <= lam + 2, 1, 'last');
#         V = np.linspace(lamAxis[L], lamAxis[R], npts)
#         # V = linspace(lamAxis(L), lamAxis(R), npts);
#         lamAxis = np.concatenate((lamAxis[0:L], V, lamAxis[R + 1 : -1]))
#
#     else:
#         lamAxis = np.linspace(min_lam, max_lam, npts)
#
#     omgs = 2e7 * np.pi * c / lamAxis  # Scattered frequency axis(1 / sec)
#     omgL = 2 * np.pi * 1e7 * c / lam  # laser frequency Rad / s
#
#     return omgL, omgs, lamAxis, npts


from jax import numpy as jnp


def lamParse(lamrang, lam, npts, fineion=True):
    """
    This function handles the calculation of wavelength grids and their associated frequency grids. A few predefined ranges
    are provided as string input options or a numeric range can be selected. A boolean switch is included to allow a finer
    griding in the ion spectrum allowing for simultaneous resolution of the ion and electron frequencies.
    """

    c = 2.99792458e10
    # if isinstance(lamrang, str):
    #     if lamrang == "EPWb":
    #         rng = 59.2
    #         off = 67.2
    #         npts = 48000
    #     elif lamrang == "EPWr":
    #         rng = 150
    #         off = -173.5
    #         npts = 37910
    #     elif lamrang == "Full":
    #         rng = 450
    #         off = 0
    #         npts = 102500
    #     elif lamrang == "EPWbNIF":
    #         rng = 35
    #         off = 45.6
    #         npts = 20273
    #
    #     min_lam = lam - off - rng  # Minimum wavelength to run code over
    #     max_lam = lam - off + rng  # Max wavelength
    #
    # else:
    min_lam = lamrang[0]
    max_lam = lamrang[1]
    npts = 1024

    # if fineion and (min_lam < lam and max_lam > lam):
    #     lamAxis = jnp.linspace(min_lam, max_lam, npts)
    #     L = next(i for i in range(len(lamAxis)) if lamAxis[i] >= lam - 2)
    #     # print(L)
    #     # L = find(lamAxis >= lam - 2, 1, 'first');
    #     rlamAxis = lamAxis[::-1]
    #     R = jnp.abs(next(i for i in range(len(lamAxis)) if rlamAxis[i] <= lam + 2) - len(lamAxis))
    #     # R = find(lamAxis <= lam + 2, 1, 'last');
    #     V = jnp.linspace(lamAxis[L], lamAxis[R], npts)
    #     # V = linspace(lamAxis(L), lamAxis(R), npts);
    #     lamAxis = jnp.concatenate((lamAxis[0:L], V, lamAxis[R + 1 : -1]))
    #
    # else:
    lamAxis = jnp.linspace(min_lam, max_lam, npts)

    omgs = 2e7 * jnp.pi * c / lamAxis  # Scattered frequency axis(1 / sec)
    omgL = 2 * jnp.pi * 1e7 * c / lam  # laser frequency Rad / s

    return omgL, omgs, lamAxis, npts
