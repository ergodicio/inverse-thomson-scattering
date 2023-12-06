.. TSADAR documentation master file, created by
   sphinx-quickstart on Fri Nov 10 11:06:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TSADAR!
==================================

TSADAR is a Thomson scattering data analysis software written in Python. It helps determine plasma parameters given
Thomson scattering measurements by fitting the observed spectra to known parameterized functional forms. More detail on
the theory is provided elsewhere [1] and will be added to this documentation soon.

The fitting is performed via gradient descent. The gradients are acquired using automatic differentiation and JAX [2].
More details on the numerics will be added soon.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage

*What does TSADAR stand for?*

TBD







References
------------

[1] - Sheffield, J., Froula, D., Glenzer, S.H. and Luhmann Jr, N.C., 2010. Plasma scattering of electromagnetic radiation: theory and measurement techniques. Academic press.

[2] - https://github.com/google/jax


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
