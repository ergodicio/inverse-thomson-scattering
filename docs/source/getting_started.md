# Getting Started

To use TSADAR:
	1. Clone the Github repository to the local or remote machine where you will be running analysis.
	2. Install
		a. Navigate to the code's base folder `inverse_thomson_scattering/`
		b. To run on a CPU: Create the TSADAR-cpu environment in a conda environment by running
			```conda env create --file env.yml ```
		c. To run on a GPU: Create the TSADAR-gpu environment in a conda environment by running
			```conda env create --file env_gpu.yml ```
	3. Run the code
		a. Activate the environment
			```conda activate tsadar-cpu```
			or
			```conda activate tsadar-gpu```
		b. Run the code
			```python3 run_tsadar.py --cfg <path> --mode <fit or forward>```
		c. If running on NERSC (additional setup required) jobs can be queued
			```python3 queue_tsadar.py --cfg <path> --mode <fit or forward>```

The inputs for the code are stored in an input deck. The default location for this input deck and therefore the starting path for running jobs is `inverse_thomson_scattering/config/1d`. These inputs should be modified to change the specific to fit your analysis need. More information on the Input deck can be found on the [Input Deck](input_deck.md) page.`
