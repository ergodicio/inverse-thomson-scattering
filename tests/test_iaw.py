import numpy as np
from jax import config
from jax import jit
from jax import numpy as jnp
from copy import deepcopy

config.update("jax_enable_x64", True)
from numpy.testing import assert_allclose
from scipy.signal import find_peaks
from inverse_thomson_scattering.form_factor import get_form_factor_fn
from inverse_thomson_scattering.misc.num_dist_func import get_num_dist_func


def test_iaw():
    """
    Test #2: IAW test, calculate a spectrum and compare the resonance to the IAW dispersion relation

    Returns:

    """

    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    Mp = Me * 1836.1  # proton mass keV/C^2
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm

    nonMaxwThomsonI_jax = get_form_factor_fn([525, 528], npts=8192, backend="jax")
    xie = np.linspace(-7, 7, 1024)
    sa = np.array([60])
    num_dist_func = get_num_dist_func({"DLM": []}, xie)
    fecur = num_dist_func(2.0)
    lam = 526.5
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)

    inps = dict(
        Te=0.5, Ti=0.2, Z=1, A=1, fract=1, ne=np.array([0.2 * 1e20]), Va=0, ud=0, sa=sa, fe=(fecur, xie), lam=lam
    )

    ThryI, lamAxisI = jit(nonMaxwThomsonI_jax)(**inps)

    ThryI = jnp.real(ThryI)
    ThryI = jnp.mean(ThryI, axis=0)

    ThryI = np.squeeze(ThryI)
    test = deepcopy(np.asarray(ThryI))
    peaks, peak_props = find_peaks(test, height=0.1, prominence=0.2)
    highest_peak_index = peaks[np.argmax(peak_props["peak_heights"])]
    second_highest_peak_index = peaks[np.argpartition(peak_props["peak_heights"], -2)[-2]]

    lams = lamAxisI[0, [highest_peak_index, second_highest_peak_index], 0]
    omgpe = constants * jnp.sqrt(0.2 * 1e20)
    omgL = 2 * np.pi * 1e7 * C / lam  # laser frequency Rad / s
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C

    omgs = 2 * jnp.pi * C / lams  # peak frequencies
    omg = 2 * kL * jnp.sqrt((0.5 + 3 * 0.2) / Mp)
    omgs2 = [omgL + omg, omgL - omg]

    assert_allclose(omgs2, omgs, rtol=1e-2)


if __name__ == "__main__":
    test_iaw()
