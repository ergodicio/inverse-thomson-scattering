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
	slice: []
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
