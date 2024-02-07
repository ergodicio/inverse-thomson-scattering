from jax import numpy as jnp


def interp2(new_grid, old_grid, fun, fill_value=Nan, indexing="xy"):
    x_arr, xp_arr = util.promote_dtypes_inexact(x, xp)
    (fp_arr,) = util.promote_dtypes_inexact(fp)
    del x, xp, fp

    i = clip(searchsorted(xp_arr, x_arr, side="right"), 1, len(xp_arr) - 1)
    dfx = fp_arr[i] - fp_arr[i - 1]
    dx = xp_arr[i] - xp_arr[i - 1]
    delta = x_arr - xp_arr[i - 1]
    dfy = fp_arr[i] - fp_arr[i - 1]
    dx = xp_arr[i] - xp_arr[i - 1]
    delta = x_arr - xp_arr[i - 1]

    epsilon = np.spacing(np.finfo(xp_arr.dtype).eps)
    dx0 = lax.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    f = where(dx0, fp_arr[i - 1], fp_arr[i - 1] + (delta / where(dx0, 1, dx)) * df)

    # if old_grid[0].shape == fun.shape:
    #     if indexing == "xy":
    #         x = old_grid[0, :]
    #         y = old_grid[:, 0]
    #         x = old_grid[0, :]
    #         y = old_grid[:, 0]
    #     else:
    #         x = old_grid[:, 0]
    #         y = old_grid[0, :]
    # elif len(old_grid[0].squeeze().shape) == 1:
    #     x = old_grid[0].squeeze()
    #     y = old_grid[1].squeeze()
    # else:
    #     raise ValueError("Grid and function must have compatible shapes")

    x = old_grid[0]
    y = old_grid[1]

    dfx = jnp.diff(fun, axis=0) / jnp.diff(x, axis=0)
    dfy = jnp.diff(fun, axis=1) / jnp.diff(y, axis=0)
