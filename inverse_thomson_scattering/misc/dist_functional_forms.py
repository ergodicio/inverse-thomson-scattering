from jax.scipy.special import gamma
from jax import numpy as jnp
from inverse_thomson_scattering.misc.vector_tools import rotate


# we will probably want to add input checks to ensure the proper fields are defined
def DLM_1D(m, h):
    vx = jnp.arange(-8, 8, h)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0) ** m))
    fe_num = fe_num / trapz(fe_num, h)
    return vx, fe_num


def DLM_2D(m, h):
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0 = jnp.sqrt(3 * gamma(3 / m) / gamma(5 / m))
    fe_num = jnp.exp(-((jnp.sqrt(vx**2 + vy**2) / x0) ** m))
    fe_num = fe_num / trapz(trapz(fe_num, h), h)
    return (vx, vy), fe_num


def BiDLM(mx, my, tasym, theta, h):
    vx = jnp.arange(-8, 8, h)
    vy = jnp.arange(-8, 8, h)
    vx, vy = jnp.meshgrid(vx, vy)
    x0x = jnp.sqrt(3 * gamma(3 / mx) / gamma(5 / mx))
    x0y = jnp.sqrt(3 * gamma(3 / my) / gamma(5 / my))
    fe_num = jnp.exp(-((jnp.abs(vx) / x0x) ** mx) - (jnp.abs(vy) / (x0y * jnp.sqrt(tasym))) ** my)
    fe_num = rotate(fe_num, -theta)
    fe_num = fe_num / trapz(trapz(fe_num, h), h)
    return (vx, vy), fe_num


# not positive on the normalizations for f1 vs f0 so dt may not be =lambda_ei/LT
def Spitzer_3V(dt, vq, h):
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
    return 0.5 * (dx * (y[..., 1:] + y[..., :-1])).sum(-1)


# def MoraYahi_3V(dt, vq, m, h)
