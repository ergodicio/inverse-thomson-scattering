from jax import numpy as jnp
from jax import vmap

import scipy.interpolate as sp
import scipy.ndimage as snd
import numpy as np
from interpax import Interpolator2D, interp2d

from inverse_thomson_scattering.model.physics import ratintn
from inverse_thomson_scattering.misc import lam_parse
from inverse_thomson_scattering.misc.vector_tools import vadd, vsub, vdot, vdiv, v_add_dim


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

    rinterp = sp.interp1d(rdWT[:, 0], rdWT[:, 1], "linear")
    rZp = np.concatenate((xi[ai] ** -2, rinterp(xi), xi[bi] ** -2))
    iinterp = sp.interp1d(idWT[:, 0], idWT[:, 1], "linear")
    iZp = np.concatenate((0 * xi[ai], iinterp(xi), 0 * xi[bi]))

    Zp = np.vstack((rZp, iZp))
    return Zp


class FormFactor:
    def __init__(self, lamrang, npts):
        # basic quantities
        self.C = 2.99792458e10
        self.Me = 510.9896 / self.C**2  # electron mass keV/C^2
        self.Mp = self.Me * 1836.1  # proton mass keV/C^2
        self.lamrang = lamrang
        self.npts = npts
        self.h = 0.1  # 0.01
        minmax = 8.2
        h1 = 124  # 1024
        self.xi1 = jnp.linspace(-minmax - jnp.sqrt(2.0) / h1, minmax + jnp.sqrt(2.0) / h1, h1)
        self.xi2 = jnp.array(jnp.arange(-minmax, minmax, self.h))
        self.Zpi = jnp.array(zprimeMaxw(self.xi2))

    def __call__(self, params, cur_ne, cur_Te, sa, f_and_v, lam):
        """
        Calculates the Thomson spectral density function S(k,omg) and is capable of handeling multiple plasma conditions
        and scattering angles. The spectral density function is calculated with and without the ion contribution
        which can be set to an independent grid from the electron contribution. Distribution functions can be one or
        two dimensional and the appropriate susceptibility is calculated with the rational integration.

        In angular, `fe` is a Tuple, Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in radians



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
        :param fe: Distribution function (DF) and normalized velocity (x) for 1D distributions

        :return:
        """

        Te, Ti, Z, A, fract, ne, Va, ud, fe = (
            cur_Te,
            params["Ti"],
            params["Z"],
            params["A"],
            params["fract"],
            cur_ne,
            params["Va"],
            params["ud"],
            f_and_v,  # this is now a DistFunc object
        )

        Mi = jnp.array(A) * self.Mp  # ion mass
        re = 2.8179e-13  # classical electron radius cm
        Esq = self.Me * self.C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)
        sarad = sa * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1])

        Va = Va * 1e6  # flow velocity in 1e6 cm/s
        ud = ud * 1e6  # drift velocity in 1e6 cm/s

        omgL, omgs, lamAxis, _ = lam_parse.lamParse(self.lamrang, lam, npts=self.npts)  # , True)

        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne[..., jnp.newaxis, jnp.newaxis])  # plasma frequency Rad/cm
        omgs = omgs[jnp.newaxis, ..., jnp.newaxis]
        omg = omgs - omgL

        ks = jnp.sqrt(omgs**2 - omgpe**2) / self.C
        kL = jnp.sqrt(omgL**2 - omgpe**2) / self.C
        k = jnp.sqrt(ks**2 + kL**2 - 2 * ks * kL * jnp.cos(sarad))

        kdotv = k * Va
        omgdop = omg - kdotv

        # plasma parameters

        # electrons
        vTe = jnp.sqrt(Te[..., jnp.newaxis, jnp.newaxis] / self.Me)  # electron thermal velocity
        klde = (vTe / omgpe) * k

        # ions
        Z = jnp.reshape(Z, [1, 1, 1, -1])
        Mi = jnp.reshape(Mi, [1, 1, 1, -1])
        fract = jnp.reshape(fract, [1, 1, 1, -1])
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne[..., jnp.newaxis, jnp.newaxis, jnp.newaxis] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)

        vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity
        kldi = (vTi / omgpi) * (k[..., jnp.newaxis])
        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        # proper handeling of multiple ion temperatures is not implemented
        xii = 1.0 / jnp.transpose((jnp.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k)[..., jnp.newaxis])
        num_species = len(fract)
        num_ion_pts = jnp.shape(xii)
        chiI = jnp.zeros(num_ion_pts)
        ZpiR = jnp.interp(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = jnp.interp(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + jnp.sqrt(-1 + 0j) * ZpiI), 3)

        # electron susceptibility
        # calculating normilized phase velcoity(xi's) for electrons
        xie = omgdop / (k * vTe) - ud / vTe

        # capable of handling isotropic or anisotropic distribution functions
        # fe is separated into components distribution function, v / vth axis, angles between f1 and kL
        # if len(fe) == 2:
        DF, x = fe
        fe_vphi = jnp.exp(jnp.interp(xie, x, jnp.log(jnp.squeeze(DF))))

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

        ratmod = jnp.exp(jnp.interp(self.xi1, x, jnp.log(jnp.squeeze(DF))))
        ratdf = jnp.gradient(ratmod, self.xi1[1] - self.xi1[0])

        def this_ratintn(this_dx):
            return jnp.real(ratintn.ratintn(ratdf, this_dx, self.xi1))

        chiERratprim = vmap(this_ratintn)(self.xi1[None, :] - self.xi2[:, None])
        # if len(fe) == 2:
        chiERrat = jnp.reshape(jnp.interp(xie.flatten(), self.xi2, chiERratprim[:, 0]), xie.shape)
        # else:
        #     chiERrat = jnp.interpn(jnp.arange(0, 2 * jnp.pi, 10**-1.2018), xi2, chiERratprim, beta, xie, "spline")
        chiERrat = -1.0 / (klde**2) * chiERrat

        chiE = chiERrat + chiEI
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        ion_comp_fact = jnp.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE[..., jnp.newaxis])) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )

        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe
        # ele_compE = fe_vphi / vTe # commented because unused

        SKW_ion_omg = 1.0 / k[..., jnp.newaxis] * ion_comp / ((jnp.abs(epsilon[..., jnp.newaxis])) ** 2)

        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ele_omg = 1.0 / k * (ele_comp) / ((jnp.abs(epsilon)) ** 2)
        # SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe # commented because unused

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * ne[:, None, None]
        # PsOmgE = (SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne) # commented because unused
        lams = 2 * jnp.pi * self.C / omgs
        PsLam = PsOmg * 2 * jnp.pi * self.C / lams**2
        # PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2 # commented because unused
        formfactor = PsLam
        # formfactorE = PsLamE # commented because unused
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
        ax[0].plot(x, DF)
        ax[1].plot(fe_vphi[1, :, 0])
        plt.show()

        return formfactor, lams

    def calc_chi_vals(self, x, element, DF, xie_mag_at, klde_mag_at):
        # # create the grid in k-space
        # xp = x * jnp.cos(element) - y * jnp.sin(element)
        # yp = x * jnp.sin(element) + y * jnp.cos(element)
        #
        # # interpolate fe onto the grid aligned to k
        # # interp = Interpolator2D(
        # #    xp, yp, DF, method="nearest", derivative=0, extrap=(0, 0)
        # # )  # may need to switch into interpolation in the log space
        # # test = interp(0, 0)
        # # print(test)
        # newgrid = jnp.array(jnp.meshgrid(self.xi1, self.xi1))
        # # newpoints = jnp.reshape(newgrid, (2, -1))
        # if xp[0] > xp[-1]:
        #     xp = xp[::-1]
        #     DF = DF[::-1, :]
        # if yp[0] > yp[-1]:
        #     yp = yp[::-1]
        #     DF = DF[:, ::-1]
        # fe_2D_k = interp2d(newgrid[0, :, :].flatten(), newgrid[1, :, :].flatten(), xp, yp, jnp.log(DF), method="cubic")
        # # fe_2D_k = interp(newpoints[0, :], newpoints[1, :])  # this may need to be indexed 'ij'
        # fe_2D_k = jnp.exp(fe_2D_k.reshape(newgrid[0].shape))
        # fe_2D_k = fe_2D_k.at[jnp.isnan(fe_2D_k)].set(0.0)
        fe_2D_k = snd.rotate(DF, element * 180 / jnp.pi, reshape=False)
        fe_1D_k = jnp.sum(fe_2D_k, axis=1) * (x[1] - x[0])

        # # integrate over the k-perp direction
        # fe_1D_k = jnp.sum(fe_2D_k, axis=1) * (self.xi1[1] - self.xi1[0])  # this needs to be pulled from the fe object
        # # try renormalizing?
        # fe_1D_k = fe_1D_k / (jnp.sum(fe_1D_k) * (self.xi1[1] - self.xi1[0]))

        # from matplotlib import pyplot as plt
        #
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
        # ax[0].plot(self.xi1, fe_1D_k)
        # plt.show()

        # find the location of xie in axis array
        # loc = jnp.argmin(jnp.abs(self.xi1 - xie_mag_at))
        loc = jnp.argmin(jnp.abs(x - xie_mag_at))
        # add the value of fe to the fe container
        fe_vphi = fe_1D_k[loc]

        # derivative of f along k
        # df = jnp.diff(fe_1D_k) / self.h
        # df = jnp.real(jnp.gradient(fe_1D_k, self.xi1[1] - self.xi1[0]))
        df = jnp.real(jnp.gradient(fe_1D_k, x[1] - x[0]))
        # df = jnp.diff(fe_vphi, 1, 1) / jnp.diff(xie, 1, 1)
        # df = jnp.append(df, jnp.zeros((len(ne), 1, len(sa))), 1)

        # Chi is really chi evaluated at the points xie
        # so the imaginary part is
        chiEI = jnp.pi / (klde_mag_at**2) * df[loc]

        # the real part is solved with rational integration
        # giving the value at a single point where the pole is located at xie_mag[ind]
        chiERrat = (
            -1.0 / (klde_mag_at**2) * jnp.real(ratintn.ratintn(df, x - xie_mag_at, x))
        )  # this may need to be downsampled for run time
        return fe_vphi, chiEI[0], chiERrat[0]

    def calc_in_2D(self, params, ud_ang, va_ang, cur_ne, cur_Te, sa, f_and_v, lam):
        """
        NONMAXWTHOMSON calculates the Thomson spectral density function S(k,omg) and is capable of handeling multiple plasma conditions and scattering angles. The spectral density function is calculated with and without the ion contribution which can be set to an independent grid from the electron contribution. Distribution functions can be one or two dimensional and the appropriate susceptibility is calculated with the rational integration.



        :param Te: electron temperature in keV [1 by number of plasma conditions]
        :param Ti: ion temperature in keV [1 by number of ion species]
        :param Z: ionization state [1 by number of ion species]
        :param A: atomic mass [1 by number of ion species]
        :param fract: relative ion composition [1 by number of ion species]
        :param ne: electron density in 1e20 cm^-3 [1 by number of plasma conditions]
        :param Va: flow velocity 2D vector
        :param ud: drift velocity 2D vector
        :param lamrang: wavelength range in nm [1 by 2]
        :param lam: probe wavelength in nm
        :param sa: scattering angle in degrees [1 by n]
        :param fe: Distribution function (DF) and normalized velocity (x) for 1D distributions and
        Distribution function (DF), normalized velocity (x), and angles from k_L to f1 in radians
        :return:
        """
        # trying with rotations this time

        Te, Ti, Z, A, fract, ne, Va, ud, fe = (
            cur_Te,
            params["Ti"],
            params["Z"],
            params["A"],
            params["fract"],
            cur_ne,
            params["Va"],
            params["ud"],
            f_and_v,  # this is now a DistFunc object
        )

        Mi = jnp.array(A) * self.Mp  # ion mass
        re = 2.8179e-13  # classical electron radius cm
        Esq = self.Me * self.C**2 * re  # sq of the electron charge keV cm
        constants = jnp.sqrt(4 * jnp.pi * Esq / self.Me)
        sarad = sa * jnp.pi / 180  # scattering angle in radians
        sarad = jnp.reshape(sarad, [1, 1, -1])

        Va = Va * 1e6  # flow velocity in 1e6 cm/s
        # convert Va from mag, angle to x,y
        Va = (Va * jnp.cos(va_ang * jnp.pi / 180), Va * jnp.sin(va_ang * jnp.pi / 180))
        ud = ud * 1e6  # drift velocity in 1e6 cm/s
        # convert ua from mag, angle to x,y
        ud = (ud * jnp.cos(ud_ang * jnp.pi / 180), ud * jnp.sin(ud_ang * jnp.pi / 180))

        omgL, omgs, lamAxis, _ = lam_parse.lamParse(self.lamrang, lam, npts=self.npts)  # , True)

        # calculate k and omega vectors
        omgpe = constants * jnp.sqrt(ne[..., jnp.newaxis, jnp.newaxis])  # plasma frequency Rad/cm
        omgs = omgs[jnp.newaxis, ..., jnp.newaxis]
        omg = omgs - omgL

        kL = (jnp.sqrt(omgL**2 - omgpe**2) / self.C, jnp.zeros_like(omgpe))  # defined to be along the x axis
        ks_mag = jnp.sqrt(omgs**2 - omgpe**2) / self.C
        ks = (jnp.cos(sarad) * ks_mag, jnp.sin(sarad) * ks_mag)
        k = vsub(ks, kL)  # 2D
        k_mag = jnp.sqrt(vdot(k, k))  # 1D

        # kdotv = k * Va
        omgdop = omg - vdot(k, Va)  # 1D

        # plasma parameters

        # electrons
        vTe = jnp.sqrt(Te[..., jnp.newaxis, jnp.newaxis] / self.Me)  # electron thermal velocity
        klde_mag = (vTe / omgpe) * (k_mag[..., jnp.newaxis])  # 1D

        # ions
        Z = jnp.reshape(Z, [1, 1, 1, -1])
        Mi = jnp.reshape(Mi, [1, 1, 1, -1])
        fract = jnp.reshape(fract, [1, 1, 1, -1])
        Zbar = jnp.sum(Z * fract)
        ni = fract * ne[..., jnp.newaxis, jnp.newaxis, jnp.newaxis] / Zbar
        omgpi = constants * Z * jnp.sqrt(ni * self.Me / Mi)

        vTi = jnp.sqrt(Ti / Mi)  # ion thermal velocity
        kldi = (vTi / omgpi) * (k_mag[..., jnp.newaxis])
        # kldi = vdot((vTi / omgpi), v_add_dim(k))

        # ion susceptibilities
        # finding derivative of plasma dispersion function along xii array
        # proper handeling of multiple ion temperatures is not implemented
        xii = 1.0 / jnp.transpose((jnp.sqrt(2.0) * vTi), [1, 0, 2, 3]) * ((omgdop / k_mag)[..., jnp.newaxis])

        # probably should be generalized to an arbitrary distribtuion function but for now just assuming maxwellian
        ZpiR = jnp.interp(xii, self.xi2, self.Zpi[0, :], left=xii**-2, right=xii**-2)
        ZpiI = jnp.interp(xii, self.xi2, self.Zpi[1, :], left=0, right=0)
        chiI = jnp.sum(-0.5 / (kldi**2) * (ZpiR + jnp.sqrt(-1 + 0j) * ZpiI), 3)

        # electron susceptibility
        # calculating normilized phase velcoity(xi's) for electrons
        # xie = vsub(vdiv(omgdop, vdot(k, vTe)), vdiv(ud, vTe))
        xie = vdiv(vsub(vdot(omgdop / k_mag**2, k), ud), vTe)
        xie_mag = jnp.sqrt(vdot(xie, xie))
        DF, (x, y) = fe
        # print(DF)
        # print(x)
        # for a run or 2 can try plotiing out a histogram of all the angles to see if we can do some predefinitions
        # or calculate a smaller set and inperpolate from there

        # for each vector in xie
        # find the rotation angle beta, the heaviside changes the angles to [0, 2pi)
        beta = jnp.arctan(xie[1] / xie[0]) + jnp.pi * (-jnp.heaviside(xie[0], 1) + 1)

        # preallocate chiEI and chiER and fe
        # chiEI = jnp.zeros_like(beta)
        # chiERrat = jnp.zeros_like(beta)
        # fe_vphi = jnp.zeros_like(beta)
        x1D = x[0, :]

        # y1D = y[:, 0]
        #
        def this_ratintn(this_beta, this_xie_mag, this_klde_mag):
            return self.calc_chi_vals(
                x1D,
                this_beta,
                DF,
                this_xie_mag,
                this_klde_mag,
            )

        #
        # print(len(beta.flatten()))
        fe_vphi, chiEI, chiERrat = vmap(this_ratintn)(beta.flatten(), xie_mag.flatten(), klde_mag.flatten())

        # for each rotation or element of vector in xie
        # for ind, element in enumerate(beta.flatten()):
        #     v1, v2, v3 = self.calc_chi_vals(
        #         x[0, :],
        #         y[:, 0],
        #         element,
        #         DF,
        #         xie_mag[jnp.unravel_index(ind, beta.shape)],
        #         klde_mag[jnp.unravel_index(ind, beta.shape)],
        #     )
        #
        #     fe_vphi = fe_vphi.at[jnp.unravel_index(ind, beta.shape)].set(v1)
        #     chiEI = chiEI.at[jnp.unravel_index(ind, beta.shape)].set(v2)
        #     chiERrat = chiERrat.at[jnp.unravel_index(ind, beta.shape)].set(v3)
        #     print(ind)

        chiE = chiERrat + jnp.sqrt(-1 + 0j) * chiEI
        epsilon = 1.0 + chiE + chiI

        # This line needs to be changed if ion distribution is changed!!!
        ion_comp_fact = jnp.transpose(fract * Z**2 / Zbar / vTi, [1, 0, 2, 3])
        ion_comp = ion_comp_fact * (
            (jnp.abs(chiE[..., jnp.newaxis])) ** 2.0 * jnp.exp(-(xii**2)) / jnp.sqrt(2 * jnp.pi)
        )

        ele_comp = (jnp.abs(1.0 + chiI)) ** 2.0 * fe_vphi / vTe

        SKW_ion_omg = 1.0 / k_mag[..., jnp.newaxis] * ion_comp / ((jnp.abs(epsilon[..., jnp.newaxis])) ** 2)

        SKW_ion_omg = jnp.sum(SKW_ion_omg, 3)
        SKW_ele_omg = 1.0 / k_mag * (ele_comp) / ((jnp.abs(epsilon)) ** 2)
        # SKW_ele_omgE = 2 * jnp.pi * 1.0 / klde * (ele_compE) / ((jnp.abs(1 + (chiE))) ** 2) * vTe / omgpe # commented because unused

        PsOmg = (SKW_ion_omg + SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * ne[:, None, None]
        # PsOmgE = (SKW_ele_omg) * (1 + 2 * omgdop / omgL) * re**2.0 * jnp.transpose(ne) # commented because unused
        lams = 2 * jnp.pi * self.C / omgs
        PsLam = PsOmg * 2 * jnp.pi * self.C / lams**2
        # PsLamE = PsOmgE * 2 * jnp.pi * C / lams**2 # commented because unused
        formfactor = PsLam
        # formfactorE = PsLamE # commented because unused

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True, sharex=False)
        ax[0].plot(fe_vphi[1, :, 0])
        plt.show()

        return formfactor, lams
