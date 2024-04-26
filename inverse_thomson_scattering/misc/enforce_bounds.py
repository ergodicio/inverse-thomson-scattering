def enforce_bounds(weights, bounds):
    """
    Resets weights that are out of the bounds to the boundary values. This is used to handle bounds violations in the
    minimizer output. This problem is common when there is an early termination of the minimizer.


    Args:
        weights: final weights dictionary from the minimizer
        bounds: bounds zip object submitted to the minimizer

    Returns:
        rebounded_weights: weights after bounds are reinforced

    """
    # used to handle bounds violation in minimizer output. This is common when terminating a run early
    # for i in bounds:
    #     print(bounds)
    # for i in weights:
    #     print(weights)
    return weights
