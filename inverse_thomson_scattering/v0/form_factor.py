import numpy as np
import scipy.interpolate as sp
from inverse_thomson_scattering.v0 import lamParse, ratintn

import matplotlib.pyplot as plt

def zprimeMaxw(xi):

    # This function calculates the derivitive of the Z - function given an array of normilzed phase velocities(xi) as
    # defined in Chapter 5. For values of xi between - 10 and 10 a table is used. Outside of this range the assumtotic
    # approximation(see Eqn. 5.2.10) is used.
    # xi is expected to be ascending

    rdWT = np.vstack(np.loadtxt("files/rdWT.txt"))
    idWT = np.vstack(np.loadtxt("files/idWT.txt"))

    ai = xi < -10
    bi = xi > 10
    ci = ~(ai + bi)

    rinterp = sp.interp1d(rdWT[:, 0], rdWT[:, 1], "linear")
    rZp = np.concatenate((xi[ai] ** -2, rinterp(xi), xi[bi] ** -2))
    iinterp = sp.interp1d(idWT[:, 0], idWT[:, 1], "linear")
    iZp = np.concatenate((0 * xi[ai], iinterp(xi), 0 * xi[bi]))

    Zp = np.vstack((rZp, iZp))
    # print(np.shape(Zp))
    return Zp


def nonMaxwThomson(Te, Ti, Z, A, fract, ne, Va, ud, lamrang, lam, sa, *fe, expion=False):
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
    Mi = np.array(A) * Mp  # ion mass
    re = 2.8179e-13  # classical electron radius cm
    Esq = Me * C**2 * re  # sq of the electron charge keV cm
    constants = np.sqrt(4 * np.pi * Esq / Me)

    print(sa)
    print(type(sa))
    sarad = sa * np.pi / 180  # scattering angle in radians
    sarad = np.reshape(sarad, [1, 1, -1])

    Va = Va * 1e6  # flow velocity in 1e6 cm/s
    ud = ud * 1e6  # drift velocity in 1e6 cm/s
    npts = 20480 #20460

    [omgL, omgs, lamAxis, _] = lamParse.lamParse(lamrang, lam, npts, expion)

    # calculate k and omega vectors
    omgpe = constants * np.sqrt(np.transpose(ne))  # plasma frequency Rad/cm
    omg = omgs - omgL
    omg = omg[np.newaxis, ..., np.newaxis]

    ks = np.sqrt(omgs**2 - omgpe**2) / C
    ks = ks[np.newaxis, ..., np.newaxis]
    kL = np.sqrt(omgL**2 - omgpe**2) / C
    k = np.sqrt(ks**2 + kL**2 - 2 * ks * kL * np.cos(sarad))

    kdotv = k * Va
    omgdop = omg - kdotv

    # plasma parameters

    # electrons
    vTe = np.sqrt(Te / Me)  # electron thermal velocity
    klde = (vTe / omgpe) * k

    # ions
    Z = np.reshape(Z, [1, 1, 1, -1])
    A = np.reshape(A, [1, 1, 1, -1])
    Mi = np.reshape(Mi, [1, 1, 1, -1])
    fract = np.reshape(fract, [1, 1, 1, -1])
    Zbar = np.sum(Z * fract)
    ni = fract * ne / Zbar
    omgpi = constants * Z * np.sqrt(ni * Me / Mi)

    vTi = np.sqrt(Ti / Mi)  # ion thermal velocity
    # kldi = np.transpose(vTi / omgpi, [1, 0, 2, 3]) * k
    kldi = (vTi / omgpi) * (k[..., np.newaxis])

    # ion susceptibilities
    # finding derivative of plasma dispersion function along xii array
    # proper handeling of multiple ion temperatures is not implemented
    xii = 1.0 / np.transpose((np.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k)[..., np.newaxis])
    num_species = len(fract)
    num_ion_pts = np.shape(xii)
    chiI = np.zeros(num_ion_pts)

    h = 0.01
    minmax = 8.2
    h1 = 1000
    xi1 = np.linspace(-minmax - np.sqrt(2.0) / h1, minmax + np.sqrt(2.0) / h1, h1)
    xi2 = np.array(np.arange(-minmax, minmax, h))

    Zpi = zprimeMaxw(xi2)
    ZpiR = sp.interp1d(xi2, Zpi[0, :], "cubic", bounds_error=False, fill_value=0)
    ZpiR = ZpiR(xii)
    ZpiI = sp.interp1d(xi2, Zpi[1, :], "cubic", bounds_error=False, fill_value=0)
    ZpiI = ZpiI(xii)
    chiI = np.sum(-0.5 / (kldi**2) * (ZpiR + np.sqrt(-1 + 0j) * ZpiI), 3)
    
    fig, ax0 = plt.subplots(2, 2)
    ax0[0, 0].plot(ZpiR[0,:,0])
    ax0[0, 1].plot(ZpiR[0,10000:10200,0])
    ax0[1, 0].plot(ZpiI[0,:,0])
    ax0[1, 1].plot(ZpiI[0,10000:10200,0])

    # electron susceptibility
    # calculating normilized phase velcoity(xi's) for electrons
    xie = omgdop / (k * vTe) - ud / vTe

    # capable of handling isotropic or anisotropic distribution functions
    # fe is separated into components distribution function, v / vth axis, angles between f1 and kL
    if len(fe) == 2:
        [DF, x] = fe
        fe_vphi = sp.interp1d(x, np.log(DF), interpAlg, bounds_error=False, fill_value=-np.inf)
        fe_vphi = np.exp(fe_vphi(xie))
        #    np.exp(sp.interp1d(x, np.log(DF), xie, interpAlg, -np.inf))
        fe_vphi[np.isnan(fe_vphi)] = 0

    elif len(fe) == 3:
        [DF, x, thetaphi] = fe
        # the angle each k makes with the anisotropy is calculated
        thetak = np.pi - np.arcsin((ks / k) * np.sin(sarad))
        thetak[:, omg < 0, :] = -np.arcsin((ks[omg < 0] / k[:, omg < 0, :]) * np.sin(sarad))
        # arcsin can only return values from -pi / 2 to pi / 2 this attempts to find the real value
        theta90 = np.arcsin((ks / k) * np.sin(sarad))
        ambcase = bool((ks > kL / np.cos(sarad)) * (sarad < np.pi / 2))
        thetak[ambcase] = theta90[ambcase]

        beta = np.arccos(
            np.sin(thetaphi[1]) * np.sin(thetaphi[0]) * np.sin(thetak) + np.cos(thetaphi[0]) * np.cos(thetak)
        )

        # here the abs(xie) handles the double counting of the direction of k changing and delta omega being negative
        fe_vphi = np.exp(
            np.interpn(np.arange(0, 2 * np.pi, 10**-1.2018), x, np.log(DF), beta, np.abs(xie), interpAlg, -np.inf)
        )

        fe_vphi[np.isnan(fe_vphi)] = 0

    df = np.diff(fe_vphi, 1, 1) / np.diff(xie, 1, 1)
    df = np.append(df, np.zeros((1, 1, len(sa))), 1)

    chiEI = np.pi / (klde**2) * np.sqrt(-1 + 0j) * df

    ratmod = sp.interp1d(x, np.log(DF), interpAlg, bounds_error=False, fill_value=-np.inf)
    ratmod = np.array(np.exp(ratmod(xi1)), dtype=float)
    ratdf = np.gradient(ratmod, xi1[1] - xi1[0])
    ratdf[np.isnan(ratdf)] = 0

    # ratdf= ratdf[np.newaxis,...]

    # if np.shape(ratdf) == 1:
    #    ratdf = np.transpose(ratdf)

    chiERratprim = np.zeros((1, len(xi2)))

    for iw in range(len(xi2)):
        chiERratprim[:, iw] = np.real(ratintn.ratintn(ratdf, xi1 - xi2[iw], xi1))

    if len(fe) == 2:
        # not sure about the extrapolation here
        chiERrat = sp.interp1d(xi2, chiERratprim, "cubic", bounds_error=False, fill_value=0)
        chiERrat = np.squeeze(chiERrat(xie), 0)
    else:
        chiERrat = np.interpn(np.arange(0, 2 * np.pi, 10**-1.2018), xi2, chiERratprim, beta, xie, "spline")
    chiERrat = -1.0 / (klde**2) * chiERrat
    # plt.plot(chiERrat[0, :, 1])

    chiE = chiERrat + chiEI
    # print(chiEI)
    # plt.plot(np.imag(chiEI[0, :, 1]))
    epsilon = 1 + (chiE) + (chiI)
    # plt.plot(np.real(epsilon[0,:,1]))

    # This line needs to be changed if ion distribution is changed!!!
    # ion_comp = Z. * sqrt(Te / Ti. * A * 1836) * (abs(chiE)). ^ 2. * exp(-(xii. ^ 2)) / sqrt(2 * pi);
    ion_comp_fact = np.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
    ion_comp = ion_comp_fact * ((np.abs(chiE[..., np.newaxis])) ** 2.0 * np.exp(-(xii**2)) / np.sqrt(2 * np.pi))
    ele_comp = (np.abs(1 + chiI)) ** 2.0 * fe_vphi / vTe
    ele_compE = fe_vphi / vTe

    SKW_ion_omg = (
        2 * np.pi * 1.0 / klde[..., np.newaxis] * (ion_comp) / ((np.abs(epsilon[..., np.newaxis])) ** 2) * 1.0 / omgpe
    )
    SKW_ion_omg = np.sum(SKW_ion_omg, 3)
    SKW_ele_omg = 2 * np.pi * 1.0 / klde * (ele_comp) / ((np.abs(epsilon)) ** 2) * vTe / omgpe
    SKW_ele_omgE = 2 * np.pi * 1.0 / klde * (ele_compE) / ((np.abs(1 + (chiE))) ** 2) * vTe / omgpe

    PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * np.transpose(ne)
    PsOmgE = (SKW_ele_omgE) * (1 + 2 * omgdop / omgL) * re**2.0 * np.transpose(ne)
    lams = 2 * np.pi * C / omgs
    lams = lams[np.newaxis, ..., np.newaxis]
    PsLam = PsOmg * 2 * np.pi * C / lams**2
    PsLamE = PsOmgE * 2 * np.pi * C / lams**2

    formfactor = PsLam
    formfactorE = PsLamE

    return formfactor, lams
