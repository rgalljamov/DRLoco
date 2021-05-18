
## How to setup the Learn2Walk Repository on a new PC

1. Change `assets_path` in 'ref_trajecs.py'.

2. Change `is_remote()` in 'scripts/common/utils.py'

3. Install the gym_mimic_envs
	1. Open a terminal and navigate to the mujoco folder, where 'gym_mimic_envs' folder is located. 
	2. `pip install -e .` to install the environment
	    1. If you get the error "ERROR: Package 'imageio' requires a different Python: 2.7.17 not in '>=3.5'", use `pip3 install -e .` instead. 
    3. NOTE: Do not use `sudo` during installation.

4. Install Weights&Biases for logging training results
	1. `conda install -c conda-forge wandb`