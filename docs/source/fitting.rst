Best practices for fitting
---------------------------------

It is recommended start fitting data with a coarse resolution in order to identify the rough plasma conditions. These conditions can then be used as the initial conditions for a fine resolution fit.




Background and lineout selection
---------------------------------

There are multiple options for background algorithms and types of fitting. These tend to be the best options for various data types. All of these options are editable in the input deck.

**Best operation for time resolved data:**

.. code-block:: yaml

    background:
        type: pixel
        slice: 900

**Best operation for spatially resolved data:**

.. code-block:: yaml

    background:
        type: fit
        slice: 900 <or background slice for IAW>

**Best operation for lineouts of angular:**

.. code-block:: yaml

    background:
	    type: fit
	    slice: <background shot number>

**Best operation for full angular:**

.. code-block:: yaml

    background:
	    type: fit
	    val: <background shot number>
    lineouts:
	    type: range
	    start: 90
    end: 950
