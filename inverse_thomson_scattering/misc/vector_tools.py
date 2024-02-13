from jax import numpy as jnp


def vadd(a, b):
    # custom function for vector addition where a and b are tuples of ND-arrays with the first element being the ND-array of x-values and the second element being the ND-array of y-values
    return (a[0] + b[0], a[1] + b[1])


def vsub(a, b):
    # custom function for vector subtraction where a and b are tuples of ND-arrays with the first element being the ND-array of x-values and the second element being the ND-array of y-values
    return (a[0] - b[0], a[1] - b[1])


def vdot(a, b):
    print(type(a))

    print(type(b))
    # custom function for vector dot product where a and b are tuples of ND-arrays with the first element being the ND-array of x-values and the second element being the ND-array of y-values
    if type(a) is tuple:
        print("here")
        if type(b) is tuple:
            return a[0] * b[0] + a[1] * b[1]
        else:
            return (a[0] * b, a[1] * b)
    else:
        return (a * b[0], a * b[1])


def vdiv(a, b):
    # custom function for vector divided by a scalar
    if type(a) is tuple:
        if type(b) is tuple:
            raise ValueError("vector must be divided by a scalar")
        else:
            return (a[0] / b, a[1] / b)
    else:
        raise ValueError("vector must be divided by a scalar")


def v_add_dim(a):
    return (a[0][..., jnp.newaxis], a[1][..., jnp.newaxis])


def rotate(A, theta):
    # create new grid
    rot_point = [A.shape[0] / 2, A.shape[1] / 2]
    x, y = jnp.meshgrid(jnp.arange(A.shape[0]), jnp.arange(A.shape[1]))
    R = jnp.array([[jnp.cos(-theta), -jnp.sin(-theta)], [jnp.sin(-theta), jnp.cos(-theta)]])
    origin_space = jnp.matmul(R, jnp.array([x - rot_point[0], y - rot_point[1]]).swapaxes(0, 1))
    or_x = jnp.squeeze(origin_space[:, 0, :])
    or_y = jnp.squeeze(origin_space[:, 1, :])
    # w11 = (x2-x)*(y2-y)/(x2-x1)(y2-y1)
    # doing this on the pixel grid the denominator is 1 and the distance to the next point is 1-current location modulus 1
    w11 = (1 - or_x % 1) * (1 - or_y % 1)
    w12 = (1 - or_x % 1) * (or_y % 1)
    w21 = (or_x % 1) * (1 - or_y % 1)
    w22 = (or_x % 1) * (or_y % 1)
    q11 = A[
        jnp.clip(jnp.asarray(or_x + rot_point[0], dtype=int), 0, A.shape[0]),
        jnp.clip(jnp.asarray(or_y + rot_point[1], dtype=int), 0, A.shape[1]),
    ]
    q12 = A[
        jnp.clip(jnp.asarray(or_x + rot_point[0], dtype=int), 0, A.shape[0]),
        jnp.clip(jnp.asarray(or_y + rot_point[1] + 1, dtype=int), 0, A.shape[1]),
    ]
    q21 = A[
        jnp.clip(jnp.asarray(or_x + rot_point[0] + 1, dtype=int), 0, A.shape[0]),
        jnp.clip(jnp.asarray(or_y + rot_point[1], dtype=int), 0, A.shape[1]),
    ]
    q22 = A[
        jnp.clip(jnp.asarray(or_x + rot_point[0] + 1, dtype=int), 0, A.shape[0]),
        jnp.clip(jnp.asarray(or_y + rot_point[1] + 1, dtype=int), 0, A.shape[1]),
    ]
    A_rotated = w11 * q11 + w12 * q12 + w21 * q21 + w22 * q22

    return A_rotated
