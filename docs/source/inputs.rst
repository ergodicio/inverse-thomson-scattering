Configuring the inputs
========================================

This code utilizes 2 input deck called `inputs.yaml` and `defaults.yaml`. The primary input deck `inputs.yaml` contains
all the commonly altered options, things like which parameters are going to be fit, the initial values, the shotnumber
and the lineouts you want to fit. The secondary input deck `defaults.yaml` contains additional options that rarely need
to be altered and default values for all options.

Parameters
---------------------

All fitting parameters are found in the ``parameters:`` section of the input deck. Each parameter has at least 4
attributes. ``val`` is the initial value used as a starting condition for the minimizer. ``active`` is a boolean
determining if a parameter is to be fit, i.e ``active: True`` means a parameter with be fit and ``active: False`` means
a parameter with be held constant at ``val``. ``ub`` and ``lb`` are upper and lower bounds respectively for the
parameters.

- ``amp1`` is the blue-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- ``amp2`` is the red-shifted EPW amplitude multiplier with 1 being the maxmimum of the data

- ``amp3`` is the IAW amplitude multiplier with 1 being the maxmimum of the data

- ``lam`` is the probe wavelength in nanometers, small shift (<5nm) can be used to mimic wavelength calibration uncertainty

- ``Te`` is the electron temperature in keV

- ``Te_gradient`` is the electron temperature spatial gradient in % of ``Te``. ``Te`` will take the form ``linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)``, A ``val!=0`` will calculate the spectrum with a gradient.

- ``Ti`` is the ion temperature in keV

- ``Z`` is the average ionization state

- ``A`` is the atomic mass

- ``ud`` is the electron drift velocity (relative to the ions) in 10^6 cm/s

- ``Va`` is the plasma fluid velocity or flow velocity in 10^6 cm/s

- ``fract`` is the element ratio for multispecies

- ``ne`` is the electron density in 10^20 cm^-3

- ``ne_gradient`` is the electron density spatial gradient in % of ``ne``. ``ne`` will take the form ``linspace(ne-ne*ne_gradient.val/200, ne+ne*ne_gradient.val/200, ne_gradient.num_grad_points)``, A ``val!=0`` will calculate the spectrum with a gradient.

- ``m`` is the electron distribution function super-Gaussian parameter

- ``fe`` contains additional options for controlling the distribution function *more info to come*


MLFlow
--------------

When running all code output is managed by MLFlow. This included the fitted parameters as well as the automated plots.
A copy of the inputs decks will also be saved by MLFlow for easier reference. The MLFlow options can be found at the
end of ``inputs.yaml`` in the ``mlflow:`` section.

- ``experiment`` is the name of the experiment folder that the run will be associated with.

- ``run`` is the name of the analysis or forward model run. Run names do not need to be unique as many runs can be created with the same name. It is recomended that this is changed before each run.



Data
--------------
The ``data:`` section contains the specifics on which shot and what region of the shot should be analyzed.

- ``shotnum`` is the OMEGA shot number. For non-OMEGA data please contact the developers

- ``lineouts`` specifies the region of the data to take lineouts from

    - ``type`` specifies the units that the linout locations are in. Options are ``um`` for microns in imaging data, ``ps`` for picoseconds in time resolved data, ``pixel`` is the general option to specify locations in pixel numbers.

    - ``start`` the first location where a lineout will be taken.

    - ``end`` the last location where a lineout will be take

    - ``skip`` the distance between lineouts in the same units specified by ``type``

- ``background`` specifies the location where the background will be analyzed.

    - ``type`` there are multiple background algorithms availible. This field is used to select the approprate one. The options are ``Fit`` in order to fit a model to the background, ``Shot`` in order to subtract a background shot, and ``pixel`` to specify a location with background data to be subtracted.

    - ``slice`` is the location for the background algorithm. If ``Fit`` or ``pixel`` are used this is the pixel location, if ``Shot`` is used this is the shot number.


Other options
--------------------
 
The ``other:`` section includes options specifying the types of data that are being fit and other options
on how to perform the fit.

- ``load_ion_spec`` is a boolean determining if IAW data will be loaded.

- ``load_ele_spec`` is a boolean determining if EPW data will be loaded.

- ``fit_IAW`` is a boolean determining if IAW data will be fit by including it in the loss metric.

- ``fit_EPWb`` is a boolean determining if the blue shifted EPW data will be fit by including it in the loss metric.

- ``fit_EPWr`` is a boolean determining if the red shifted EPW data will be fit by including it in the loss metric.

- ``refit`` is a boolean determinging if poor fits will attempt to be refit.

- ``refit_thresh`` is the value of the loss metric below above which refits will be performed.

- ``calc_sigmas`` is a boolean determining if a Hessian will be computed to determine the uncertainty in fitted parameters.

