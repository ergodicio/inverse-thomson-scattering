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
