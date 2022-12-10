import numpy as np
from jax import config
from jax import jit
from jax import numpy as jnp

config.update("jax_enable_x64", True)

from scipy.signal import find_peaks
from inverse_thomson_scattering.form_factor import get_form_factor_fn
from inverse_thomson_scattering.numDistFunc import get_num_dist_func


def test_epw():
    """
    Test #1: Bohm-Gross test, calculate a spectrum and compare the resonance to the Bohm gross dispersion relation

    Returns:

    """

    nonMaxwThomsonE_jax, _ = get_form_factor_fn([400, 700])
    xie = np.linspace(-7, 7, 3999)
    sa = np.array([60])
    num_dist_func = get_num_dist_func({"DLM": []}, xie)
    fecur = num_dist_func(2.0)
    lam = 526.5

    ThryE, lamAxisE = jit(nonMaxwThomsonE_jax)(0.5, 0.2, 1, 1, 1, np.array([0.2 * 1e20]), 0, 0, sa, (fecur, xie), lam)

    # ThryE = np.squeeze(ThryE)
    # test = deepcopy(np.asarray(ThryE))
    peaks, peak_props = find_peaks(np.array(np.squeeze(ThryE)), height=(0.1, 1.1), prominence=0.2)
    highest_peak_index = peaks[np.argmax(peak_props["peak_heights"])]
    second_highest_peak_index = peaks[np.argpartition(peak_props["peak_heights"], -2)[-2]]

    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)
    lams = lamAxisE[0, [highest_peak_index, second_highest_peak_index], 0]
    omgs = 2 * jnp.pi * C / lams  # peak frequencies
    omgpe = constants * jnp.sqrt(0.2 * 1e20)
    omgL = 2 * np.pi * 1e7 * C / lam  # laser frequency Rad / s
    ks = jnp.sqrt(omgs**2 - omgpe**2) / C
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C
    k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sa * jnp.pi / 180))
    vTe = jnp.sqrt(0.5 / Me)
    omg = jnp.sqrt(omgpe**2 + 3 * k**2 * vTe**2)
    omgs2 = [omgL + omg[0], omgL - omg[1]]

    np.testing.assert_allclose(omgs2, omgs, rtol=1e-2)


if __name__ == "__main__":
    test_epw()
