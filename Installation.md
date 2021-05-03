
## How to setup the Learn2Walk Repository on a new PC

1. Change `assets_path` in 'ref_trajecs.py'.
2. Change `is_remote()` in 'scripts/common/utils.py'
3. Install the gym_mimic_envs
	1. Open a terminal and navigate to the mujoco folder, where 'gym_mimic_envs' folder is located. 
	2. `pip install -e .` to install the environment
