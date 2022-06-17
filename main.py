# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import scipy as sp
from jax import numpy as jnp

from lamParse import lamParse
from zprimeMaxw import zprimeMaxw
from ratintn import ratintn


def nonMaxwThomson(Te, Ti, Z, A, fract, ne, Va, ud, lamrang, lam, sa, *fe):
    """
    NONMAXWTHOMSON calculates the Thomson spectral density function S(k,omg) and is capable of handeling multiple plasma
     conditions and scattering angles. The spectral density function is calculated with and without the ion contribution
     which can be set to an independent grid from the electron contribution. Distribution functions can be one or two
     dimensional and the appropriate susceptibility is calculated with the rational integration.
    :param Te: electron temperature in keV [1 by number of plasma conditions]
    :param Ti: ion temperature in keV [1 by number of ion species]
    :param Z: ionization state [1 by number of ion species]
    :param A: atomic mass [1 by number of ion species]
    :param fract: relative ion composition [1 by number of ion species]
    :param ne: electron density in 1e20 cm^-3 [1 by number of plasma conditions]
    :param Va: flow velocity
    :param ud: drift velocity
    :param lamrang: wavelength range in nm [1 by 2]
    :param lam: probe wavelength in nm
    :param sa: scattering angle in degrees [1 by n]
    :param fe: Distribution function (DF) and normalized velocity (x) for 1D distributions and
    Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in radians
    :return:
    """
    interpAlg = "linear"

    # basic quantities
    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    Mp = Me * 1836.1  # proton mass keV/C^2
    Mi = A * Mp  # ion mass
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm
    constants = jnp.sqrt(4 * jnp.pi * Esq / Me)
    sarad = sa * jnp.pi / 180  # scattering angle in radians
    sarad = jnp.reshape(sarad, [1, 1, -1])
    Va = Va * 1e6  # flow velocity in 1e6 cm/s
    ud = ud * 1e6  # drift velocity in 1e6 cm/s
    npts = 20460

    [omgL, omgs, lamAxis, _] = lamParse(lamrang, lam, npts, False)

    # calculate k and omega vectors
    omgpe = constants * jnp.sqrt(jnp.transpose(ne))  # plasma frequency Rad/cm
    omg = omgs - omgL
    ks = jnp.sqrt(omgs**2 - omgpe**2) / C
    kL = jnp.sqrt(omgL**2 - omgpe**2) / C
    # k = jnp.sqrt()
    k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sarad))

    kdotv = k * Va
    omgdop = omg - kdotv

    # plasma parameters

    # electrons
    vTe = jnp.sqrt(Te / Me)  # electron thermal velocity
    klde = (vTe / omgpe) * k

    # ions
    Z = jnp.reshape(Z, [1, 1, 1, -1])
    A = jnp.reshape(A, [1, 1, 1, -1])
    Mi = jnp.reshape(Mi, [1, 1, 1, -1])
    fract = jnp.reshape(fract, [1, 1, 1, -1])
    Zbar = jnp.sum(Z * fract)
    ni = fract * ne / Zbar
    omgpi = constants * Z * jnp.sqrt(ni * Me / Mi)
    vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity
    kldi = jnp.swapaxes(vTi / omgpi, 1, 0) * k

    # ion susceptibilities
    # finding derivative of plasma dispersion function along xii array
    # proper handeling of multiple ion temperatures is not implemented
    xii = 1.0 / jnp.swapaxes((jnp.sqrt(2.0) * vTi), 1, 0) * (omgdop / k)
    num_species = len(fract)
    num_ion_pts = jnp.shape(xii)
    chiI = jnp.zeros(num_ion_pts)

    h = 0.01
    minmax = 8.2
    h1 = 1000
    xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
    xi2 = jnp.array(jnp.arange(-minmax, minmax, h))

    Zpi = zprimeMaxw(xi2)
    ZpiR = sp.interpolate.interpn(xi2, Zpi[0, :], xii, "spline", 0)
    ZpiI = sp.interpolate.interpn(xi2, Zpi[1, :], xii, "spline", 0)
    # ZpiR = interp1(xi2, Zpi(1,:), xii, 'spline', 0);
    # ZpiI = interp1(xi2, Zpi(2,:), xii, 'spline', 0);
    chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + jnp.sqrt(-1) * ZpiI), 4)

    # electron susceptibility
    # calculating normilized phase velcoity(xi's) for electrons
    xie = omgdop / (k * vTe) - ud / vTe

    # capable of handling isotropic or anisotropic distribution functions
    # fe is separated into components distribution function, v / vth axis, angles between f1 and kL
    if len(fe) == 2:
        [DF, x] = fe
        fe_vphi = jnp.exp(jnp.interp(x, jnp.log(DF), xie))# , interpAlg, -jnp.inf))
        fe_vphi[jnp.isnan(fe_vphi)] = 0

    elif len(fe) == 3:
        [DF, x, thetaphi] = fe
        # the angle each k makes with the anisotropy is calculated
        thetak = jnp.pi - jnp.arcsin((ks / k) * jnp.sin(sarad))
        thetak[:, omg < 0, :] = -jnp.arcsin((ks[omg < 0] / k[:, omg < 0, :]) * jnp.sin(sarad))
        # arcsin can only return values from -pi / 2 to pi / 2 this attempts to find the real value
        theta90 = jnp.arcsin((ks / k) * jnp.sin(sarad))
        ambcase = ((ks > kL / jnp.cos(sarad)) and (sarad < jnp.pi / 2)) # NBNBNBNB
        thetak[ambcase] = theta90[ambcase]

        beta = jnp.arccos(
            jnp.sin(thetaphi[1]) * jnp.sin(thetaphi[0]) * jnp.sin(thetak) + jnp.cos(thetaphi[0]) * jnp.cos(thetak)
        )

        # here the abs(xie) handles the double counting of the direction of k changing and delta omega being negative
        fe_vphi = jnp.exp(
            jnp.interp(
                jnp.arange(0, 2 * jnp.pi, 10**-1.2018), x, jnp.log(DF), beta, jnp.abs(xie) #, interpAlg, -jnp.inf
            )
        )

        fe_vphi[jnp.isnan(fe_vphi)] = 0

    else:
        raise NotImplementedError

    df = jnp.diff(fe_vphi, 1, 1) / jnp.diff(xie, 1, 1)
    df = df + jnp.zeros(jnp.shape(df + [0, 1, 0]))

    chiEI = jnp.pi / (klde**2) * jnp.sqrt(-1) * df

    # interpAlg, -jnp.inf
    ratdf = jnp.gradient(
        jnp.exp(jnp.transpose(jnp.interp(x, jnp.log(jnp.transpose(DF)), xi1))), xi1[1] - xi1[0]
    )
    ratdf[jnp.isnan(ratdf)] = 0
    if jnp.shape(ratdf, 1) == 1:
        ratdf = jnp.transpose(ratdf)

    chiERratprim = jnp.zeros(jnp.shape(ratdf, 0), len(xi2))
    for iw in range(len(xi2)):
        chiERratprim[:, iw] = jnp.real(ratintn(ratdf, xi1 - xi2(iw), xi1))

    if len(fe) == 2:
        chiERrat = jnp.interpn(xi2, chiERratprim, xie, "spline")
    else:
        chiERrat = jnp.interpn(jnp.arange(0, 2 * jnp.pi, 10**-1.2018), xi2, chiERratprim, beta, xie, "spline")
    chiERrat = -1.0 / (klde**2) * chiERrat

    chiE = chiERrat + chiEI
    epsilon = 1 + (chiE) + (chiI)

    # This line needs to be changed if ion distribution is changed!!!
    # ion_comp = Z. * sqrt(Te / Ti. * A * 1836) * (abs(chiE)). ^ 2. * exp(-(xii. ^ 2)) / sqrt(2 * pi);
    ion_comp = (
        jnp.swapaxes(fract * Z**2 / Zbar / vTi, 1, 0)
        * (jnp.abs(chiE)) ** 2.0
        * jnp.exp(-(xii**2))
        / jnp.sqrt(2 * jnp.pi)
    )
    ele_comp = (jnp.abs(1 + chiI)) ** 2.0 * jnp.float64(fe_vphi) / vTe
    ele_compE = jnp.float64(fe_vphi) / vTe

    SKW_ion_omg = 2 * jnp.pi * 1.0 / klde * (ion_comp) / ((jnp.abs(epsilon)) ** 2) * 1.0 / omgpe
    SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
    SKW_ele_omg = 2 * jnp.pi * 1.0 / klde * (ele_comp) / ((jnp.abs(epsilon)) ** 2) * vTe / omgpe
    SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe

    PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne)
    PsOmgE = (SKW_ele_omgE) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne)
    lams = 2 * jnp.pi * C / omgs
    PsLam = PsOmg * 2 * jnp.pi * C / lams**2
    PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2

    formfactor = PsLam
    formfactorE = PsLamE


x = jnp.array(jnp.arange(0, 8, 0.1))
distf = 1 / (2 * jnp.pi) ** (1 / 2) * jnp.exp(-(x**2) / 2)

nonMaxwThomson(1, 1, 1, 1, 1, 0.3, 0, 0, [400, 700], 526.5, 60, [distf, x])
