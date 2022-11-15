from jax import config

config.update("jax_enable_x64", True)
#config.update('jax_disable_jit',True)

from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
from inverse_thomson_scattering import ratintn
from inverse_thomson_scattering import lamParse
import numpy as np
import scipy.interpolate as sp

def zprimeMaxw(xi):
    """
    This function calculates the derivitive of the Z - function given an array of normilzed phase velocities(xi) as
    defined in Chapter 5. For values of xi between - 10 and 10 a table is used. Outside of this range the assumtotic
    approximation(see Eqn. 5.2.10) is used.
    xi is expected to be ascending
    Args:
        xi:
    Returns:
    """

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


def get_form_factor_fn(lamrang):
    npts = 20460

    # basic quantities
    C = 2.99792458e10
    Me = 510.9896 / C**2  # electron mass keV/C^2
    Mp = Me * 1836.1  # proton mass keV/C^2

    h = 0.01
    minmax = 8.2
    h1 = 1000
    xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
    xi2 = jnp.array(jnp.arange(-minmax, minmax, h))
    Zpi = jnp.array(zprimeMaxw(xi2))

    def nonMaxwThomson(Te, Ti, Z, A, fract, ne, Va, ud, sa, fe, lam):
        """
        NONMAXWTHOMSON calculates the Thomson spectral density function S(k,omg) and is capable of handeling multiple plasma conditions and scattering angles. The spectral density function is calculated with and without the ion contribution which can be set to an independent grid from the electron contribution. Distribution functions can be one or two dimensional and the appropriate susceptibility is calculated with the rational integration.
         
         
         
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

        Mi = jnp.array(A) * Mp  # ion mass
        re = 2.8179e-13  # classical electron radius cm
        Esq = Me * C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / Me)
        sarad = sa * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1])

        Va = Va * 1e6  # flow velocity in 1e6 cm/s
        ud = ud * 1e6  # drift velocity in 1e6 cm/s

        omgL, omgs, lamAxis, _ = lamParse.lamParse(lamrang, lam, npts)  # , True)

        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne[..., jnp.newaxis, jnp.newaxis])  # plasma frequency Rad/cm
        omgs = omgs[jnp.newaxis, ..., jnp.newaxis]
        omg = omgs - omgL

        ks = jnp.sqrt(omgs**2 - omgpe**2) / C
        kL = jnp.sqrt(omgL**2 - omgpe**2) / C
        #kL = kL[..., jnp.newaxis]
        k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sarad))

        kdotv = k * Va
        omgdop = omg - kdotv

        # plasma parameters

        # electrons
        vTe = jnp.sqrt(Te[..., jnp.newaxis, jnp.newaxis] / Me)  # electron thermal velocity
        klde = (vTe / omgpe) * k

        # ions
        Z = jnp.reshape(Z, [1, 1, 1, -1])
        # A = jnp.reshape(A, [1, 1, 1, -1])
        Mi = jnp.reshape(Mi, [1, 1, 1, -1])
        fract = jnp.reshape(fract, [1, 1, 1, -1])
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne[..., jnp.newaxis, jnp.newaxis, jnp.newaxis] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * Me / Mi)
        

        vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity
        # kldi = jnp.transpose(vTi / omgpi, [1, 0, 2, 3]) * k
        kldi = (vTi / omgpi) * (k[..., jnp.newaxis])

        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        # proper handeling of multiple ion temperatures is not implemented
        xii = 1.0 / jnp.transpose((jnp.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k)[..., jnp.newaxis])
        num_species = len(fract)
        num_ion_pts = jnp.shape(xii)
        chiI = jnp.zeros(num_ion_pts)

        ZpiR = jnp.interp(xii, xi2, Zpi[0, :], left=0, right=0)  # , "cubic", bounds_error=False, fill_value=0)
        ZpiI = jnp.interp(xii, xi2, Zpi[1, :], left=0, right=0)  # , "cubic", bounds_error=False, fill_value=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + jnp.sqrt(-1 + 0j) * ZpiI), 3)

        # electron susceptibility
        # calculating normilized phase velcoity(xi's) for electrons
        xie = omgdop / (k * vTe) - ud / vTe

        # capable of handling isotropic or anisotropic distribution functions
        # fe is separated into components distribution function, v / vth axis, angles between f1 and kL
        # if len(fe) == 2:
        DF, x = fe
        fe_vphi = jnp.exp(jnp.interp(xie, x, jnp.log(DF)))  # , interpAlg, bounds_error=False, fill_value=-jnp.inf)

        # elif len(fe) == 3:
        #     [DF, x, thetaphi] = fe
        #     # the angle each k makes with the anisotropy is calculated
        #     thetak = jnp.pi - jnp.arcsin((ks / k) * jnp.sin(sarad))
        #     thetak[:, omg < 0, :] = -jnp.arcsin((ks[omg < 0] / k[:, omg < 0, :]) * jnp.sin(sarad))
        #     # arcsin can only return values from -pi / 2 to pi / 2 this attempts to find the real value
        #     theta90 = jnp.arcsin((ks / k) * jnp.sin(sarad))
        #     ambcase = bool((ks > kL / jnp.cos(sarad)) * (sarad < jnp.pi / 2))
        #     thetak[ambcase] = theta90[ambcase]
        #
        #     beta = jnp.arccos(
        #         jnp.sin(thetaphi[1]) * jnp.sin(thetaphi[0]) * jnp.sin(thetak) + jnp.cos(thetaphi[0]) * jnp.cos(thetak)
        #     )
        #
        #     # here the abs(xie) handles the double counting of the direction of k changing and delta omega being negative
        #     fe_vphi = jnp.exp(
        #         jnp.interpn(
        #             jnp.arange(0, 2 * jnp.pi, 10**-1.2018), x, jnp.log(DF), beta, jnp.abs(xie), interpAlg, -jnp.inf
        #         )
        #     )
        #
        #     fe_vphi[jnp.isnan(fe_vphi)] = 0

        df = jnp.diff(fe_vphi, 1, 1) / jnp.diff(xie, 1, 1)
        df = jnp.append(df, jnp.zeros((len(ne), 1, len(sa))), 1)

        chiEI = jnp.pi / (klde**2) * jnp.sqrt(-1 + 0j) * df

        ratmod = jnp.exp(jnp.interp(xi1, x, jnp.log(DF)))  # , interpAlg, bounds_error=False, fill_value=-jnp.inf)
        ratdf = jnp.gradient(ratmod, xi1[1] - xi1[0])

        def this_ratintn(this_dx):
            return jnp.real(ratintn.ratintn(ratdf, this_dx, xi1))

        chiERratprim = vmap(this_ratintn)(xi1[None, :] - xi2[:, None])
        # if len(fe) == 2:
        # not sure about the extrapolation here
        chiERrat = jnp.reshape(jnp.interp(xie.flatten(), xi2, chiERratprim[:, 0]), xie.shape)
        # else:
        #     chiERrat = jnp.interpn(jnp.arange(0, 2 * jnp.pi, 10**-1.2018), xi2, chiERratprim, beta, xie, "spline")
        chiERrat = -1.0 / (klde**2) * chiERrat
        chiE = chiERrat + chiEI
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        # ion_comp = Z. * sqrt(Te / Ti. * A * 1836) * (abs(chiE)). ^ 2. * exp(-(xii. ^ 2)) / sqrt(2 * pi);
        ion_comp_fact = jnp.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE[..., jnp.newaxis])) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )
        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe
        # ele_compE = fe_vphi / vTe # commented because unused

        SKW_ion_omg = (
            2.0
            * jnp.pi
            * 1.0
            / klde[..., jnp.newaxis]
            * ion_comp
            / ((jnp.abs(epsilon[..., jnp.newaxis])) ** 2)
            * 1.0
            / omgpe[..., jnp.newaxis]
        )
        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ele_omg = 2 * jnp.pi * 1.0 / klde * (ele_comp) / ((jnp.abs(epsilon)) ** 2) * vTe / omgpe
        # SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe # commented because unused

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne)
        # PsOmgE = (SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne) # commented because unused
        lams = 2 * jnp.pi * C / omgs
        PsLam = PsOmg * 2 * jnp.pi * C / lams**2
        # PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2 # commented because unused
        formfactor = PsLam
        # formfactorE = PsLamE # commented because unused
        return formfactor, lams

    def cost_fn(Te, Ti, Z, A, fract, ne, Va, ud, sa, fe):
        formf, _ = nonMaxwThomson(Te, Ti, Z, A, fract, ne, Va, ud, sa, fe)
        return jnp.sum(formf)

    return jit(nonMaxwThomson), jit(value_and_grad(cost_fn))