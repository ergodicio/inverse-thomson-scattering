Configuring the inputs
========================================

This code utilizes 2 input deck called `inputs.yaml` and `defaults.yaml`. The primary input deck `inputs.yaml` contains
all the commonly altered options, things like which parameters are going to be fit, the initial values, the shotnumber
and the lineouts you want to fit. The secondary input deck `defaults.yaml` contains additional options that rarely need
to be altered and default values for all options.

All fitting parameters are found in the ``parameters:`` section of the input deck. Each parameter has at last 4
attributes. ``val`` is the initial value used as a starting condition for the minimizer. ``active`` is a boolean
determining if a parameter is to be fit, i.e ``active: True`` means a parameter with be fit and ``active: True`` means a
parameter with be held constant at ``val``. ``lb`` and ``ub`` are upper and lower bounds respectively for the parameters.

Parameter Description
---------------------

- ``amp1`` is the blue-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- ``amp2`` is the red-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- ``amp3`` is the IAW amplitude multiplier with 1 being the maxmimum of the data

- ``lam`` is the probe wavelength in nanometers, small shift (<5nm) can be used to mimic wavelength calibration uncertainty

- ``Te`` is the electron temperature in keV

- ``Ti`` is the ion temperature in keV

- ``Z`` is the average ionization state

- ``A`` is the atomic mass

- ``ud`` is the electron drift velocity (relative to the ions) in 10^6 cm/s

- ``Va`` is the plasma fluid velocity of flow velocity in 10^6 cm/s

- ``fract`` is the element ratio for multispecies (currently depreciated)

- ``ne`` is the electron density in 10^20 cm^-3

- ``m`` is the electron distribution function super-Gaussian parameter

- ``Te_gradient`` is the electron temperature spatial gradient in % of ``Te``. ``Te`` will take the form ``linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)``, A nonzero ``val`` will calculate the spectrum with a gradient.

- ``ne_gradient`` is the electron density spatial gradient in % of ``ne``. The same rules that apply to ``Te_gradient`` also apply here.





