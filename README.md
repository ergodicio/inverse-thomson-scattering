# Thomson Scattering Fitting Code

## Instructions for running the code from a Juypiterhub terminal

**To initialize the enviornment:**
```
	bash init.sh
	mamaba env create -f env_gpu.yml
	mamba activate ts
```

**Then the code can be run with:**
```
	python3 <file to run>
```

**Tends to work better for ATS with:**
```
	XLA_PYTHON_CLIENT_PREALLOCATE=false python3 <file to run>
```

**To watch the processors and gpu (from thier own terminals):**
```
	htop
	watch -n1 nvidia-smi
```

**To copy files from MLflow to Juypiterhub:**
```
	aws s3 cp <filepath> <destination_filepath>
```

## Input deck

This code utilizes 2 input deck called `inputs.yaml` and `defaults.yaml`. The primary input deck `inputs.yaml` contains all the commonly altered options, things like which parameters are going to be fit, the initial values, the shotnumber and the lineouts you want to fit. The secondary input deck `defaults.yaml` contains additional options that rarely need to be altered and default values for all options.

All fitting parameters are found in the `parameters:` section of the input deck. Each parameter has at last 4 atributes. `val` is the intial value used as a starting condition for the minimizer. `active` is a boolean determining if a papeter is to be fit, i.e `active: True` means a parameter with be fit and `active: True` means a parameter with be held constant at `val`. `lb` and `ub` are upper and lower bounds respectively for the parameters.

The fitting parameters have the following meanings and normalization:

`amp1` is the blue-shifted EPW amplitude multiplier with 1 being the maxmimum of the data
 
`amp2` is the red-shifted EPW amplitude multiplier with 1 being the maxmimum of the data
  
`amp3` is the IAW amplitude multiplier with 1 being the maxmimum of the data
  
`lam` is the probe wavelength in nanometers, small shift (<5nm) can be used to mimic wavelength calibration uncertainty
  
`Te` is the electron temperature in keV
   
`Te_gradient` is the electron temperature spatial gradient in % of `Te`. `Te` will take the form `linspace(Te-Te*Te_gradient.val/200, Te+Te*Te_gradient.val/200, Te_gradient.num_grad_points)`, A `val!=0` will calculate the spectrum with a gradient.
   
`Ti` is the ion temperature in keV
    
`Z` is the average ionization state
     
`A` is the atomic mass

`ud` is the electron drift velocity (relative to the ions) in 10^6 cm/s
        
`Va` is the plasma fluid velocity of flow velocity in 10^6 cm/s
      	
`fract` is the element ratio for multispecies (currently depreciated)
        
`ne` is the electron density in 10^20 cm^-3
	
`ne_gradient` is the electron density spatial gradient in % of `ne`. `ne` will take the form `linspace(ne-ne*ne_gradient.val/200, ne+ne*ne_gradient.val/200, ne_gradient.num_grad_points)`, A `val!=0` will calculate the spectrum with a gradient.
 	
`m` is the electron distribtuion function super-Gaussian order
 

  
## Best practices for fitting

It is generarly recommended start fitting data with a coarse resolution in order to identify the rough plasma conditions. These conditions can then be used as the initial conditions for a fine resolution fit.

### Background and lineout selection

There are multiple options for background algorithms and types of fitting. These tend to be the best options for various data types. All of these options are editable in the input deck.

**Best operation for time resolved data:**
```
background:
	type: pixel
	slice: 900
```

**Best operation for spatialy resolved data:**
```
background:
	type: fit
	slice: 900 <or background slice for IAW>
```

**Best operation for lineouts of angular:**
```
background:
	type: fit
	slice: <background shot number>
```

**Best operation for full angular:**
```
background:
	type: fit
	val: <background shot number>
lineouts:
	type: range
	start: 90
  end: 950
```
