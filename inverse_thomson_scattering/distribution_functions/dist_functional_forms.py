from jax.scipy.special import gamma
from jax import numpy as jnp
from inverse_thomson_scattering.misc.vector_tools import rotate


# we will probably want to add input checks to ensure the proper fields are defined
def DLM_1D(m, h):
    """
    Produces a 1-D Dum-Langdon-Matte distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        vx: normalized velocity grid
        fe_num: numerical distribution function
    """

    def SG(vx, m):
        x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
        return jnp.exp(-((jnp.abs(vx) / x0) ** m))

    vx = jnp.arange(-8, 8, h)
    fe_num = jnp.array([trapz(SG(jnp.sqrt(vx**2 + vz**2), m), h) for vz in vx])

    # x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    # fe_num = jnp.exp(-((jnp.abs(vx) / x0) ** m))
    fe_num = fe_num / trapz(fe_num, h)
    return vx, fe_num


def SG_1D(m, h):
    """
    Produces a 1-D Super-Gaussian distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        vx: normalized velocity grid
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0) ** m))
    fe_num = fe_num / trapz(fe_num, h)
    return vx, fe_num


# Warning: These super-gaussian orders do not follow Matte
def DLM_2D(m, h):
    """
    Produces a 2-D symmetric Dum-Langdon-Matte distribution parametrized by a super-gaussian order m.

    Args:
        m: (int) Super-Gaussian order
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.sqrt(vx**2 + vy**2) / x0) ** m))
    fe_num = fe_num / trapz(trapz(fe_num, h), h)
    return (vx, vy), fe_num


# Warning: These super-gaussian orders do not follow Matte
def BiDLM(mx, my, tasym, theta, h):
    """
    Produces a 2-D Dum-Langdon-Matte distribution that can have different widths and super-gaussian orders in the 2
    dimensions.

    Args:
        mx: (int) Super-Gaussian order for the x direction
        my: (int) Super-Gaussian order for the y direction
        tasym: (int) Temperature asymetry, where the y direction will have an effective temperature of Te*tasym. x
        direction will have an effective temperature of Te.
        theta: (int) counter-clockwise rotation of the distribution in radians
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0x = jnp.sqrt(3 * gamma(3 / mx) / gamma(5 / mx))
    x0y = jnp.sqrt(3 * gamma(3 / my) / gamma(5 / my))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0x) ** mx) - (jnp.abs(vy) / (x0y * jnp.sqrt(tasym))) ** my)
    fe_num = rotate(fe_num, theta)
    fe_num = fe_num / trapz(trapz(fe_num, h), h)
    return (vx, vy), fe_num


# not positive on the normalizations for f1 vs f0 so dt may not be =lambda_ei/LT
def Spitzer_3V(dt, vq, h):
    """
    Produces a 2-D Spitzer-Harm distribution with the f1 direction given in 3-space.

    Args:
        dt: (int) Knudsen number determining the magnitude of the perturbation
        vq: array or list with 3 elements giving the direction of the f1 perturbation in x,y,z
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    # likely to OOM (probably a shortcut by calculating the anlge out of the plane and multiplying f1 by cos of that angle)
    x = jnp.arange(-8, 8, h)
    y = jnp.arange(-8, 8, h)
    z = jnp.arange(-8, 8, h)
    vx, vy, vz = jnp.meshgrid(x, y, z)
    # vq = vq/jnp.sqrt(vq[0]**2 + vq[1]**2 + vq[2]**2)
    f0 = 1 / (2 * jnp.pi) ** (3 / 2) * jnp.exp(-(vx**2 + vy**2 + vz**2) / 2)
    f1 = (
        dt
        * jnp.sqrt(2 / (9 * jnp.pi))
        * (vx * vq[0] + vy * vq[1] + vz * vq[2]) ** 4
        * (4 - (vx * vq[0] + vy * vq[1] + vz * vq[2]) / 2)
        * f0
    )
    fe_num = f0 + f1
    fe_num = trapz(fe_num, h)  # integrate over z
    fe_num = fe_num / trapz(trapz(fe_num, h), h)  # renormalize

    # redefine to coordinates
    vx, vy = jnp.meshgrid(x, y)

    return (vx, vy), fe_num


def Spitzer_2V(dt, vq, h):
    """
    Produces a 2-D Spitzer-Harm distribution with the f1 direction given in the plane.

    Args:
        dt: (int) Knudsen number determining the magnitude of the perturbation
        vq: array or list with 2 elements giving the direction of the f1 perturbation in x,y
        h: (int) resolution of normalized velocity grid, i.e. spacing of the grid

    Returns:
        (vx, vy): tuple of the normalized velocity grids in x and y
        fe_num: numerical distribution function
    """
    x = jnp.arange(-8, 8, h)
    y = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(x, y)
    # vq = vq/jnp.sqrt(vq[0]**2 + vq[1]**2)
    f0 = 1 / (2 * jnp.pi) ** (3 / 2) * jnp.exp(-(vx**2 + vy**2) / 2)
    f1 = dt * jnp.sqrt(2 / (9 * jnp.pi)) * (vx * vq[0] + vy * vq[1]) ** 4 * (4 - (vx * vq[0] + vy * vq[1]) / 2) * f0
    fe_num = f0 + f1
    fe_num = fe_num / trapz(trapz(fe_num, h), h)  # renormalize

    return (vx, vy), fe_num


def trapz(y, dx):
    """
    JAX compatible trapizoidal intergration.

    Args:
        y: numerical array to be integrated
        dx: spacing of the associated x-axis

    Returns:
        z: integral of ydx
    """
    return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)


# def MoraYahi_3V(dt, vq, m, h)
