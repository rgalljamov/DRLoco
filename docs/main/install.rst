
.. _install:

Installation
******************

.. warning::
   
   @Guoping, I've not set up the environment on a purely new empty PC. Instead our DL PC already had some packages installed. Therefore, the installation guidelines below might not be sufficient. I'll go through all the steps on Oksanas Laptop and write a better documentation of the installation processs!


This page guides you command by command through the process to get up and running with DRLoco. It will take some time (1-2 hours or days in the worst case) and might get tricky at some point, but it's all worth it.

Setting up the DRL environment for Pytorch and SB3

1. Update your systems packages to the newest version: ``sudo apt update`` and ``sudo apt upgrade``

2. ``sudo apt-get install cuda-drivers-450``
	a. does work... there is now a huge list of cuda-packages that were automatically installed and are no longer required with the info to remove them with 'sudo apt autoremove'
	b. apt list --upgradable leads to 'Listing... Done.'

3. ``sudo apt autoremove``
	1. almost 5GB of storage will be freed. 
	2. lots of nvidia and cuda packages are getting deleted.
	3. apt update and apt list --upgradable does not show any open packages to insall



## Setting up the CONDA environment
---------------------------------------------

1. conda update -n base conda
	1. to update conda to the newest version

2. conda create -n torch python=3.8
	1. 'torch' here is the name of the conda environment. You can replace it.
	2. NOTE: If debugging is not working with _python=3.8_, just install python 3.7 after the environment is created with `conda install python=3.7`

3. conda activate torch; pip install stable-baselines3

4. install mujoco_py: ``pip3 install mujoco-py``

5. Try executing one of the scripts in the ``scripts/`` folder to see if there are any other packages missing in your environment and install them, too.

6. To record videos of the generated walking gait at the training's end, we need to further install ``ffmpeg``: ``conda install -c conda-forge ffmpeg``
	1. If you want to record the video on a remote server without a UI, in addtion install
		1. `sudo apt install xvfb`
		2. to execute the training script run `xvfb-run python /path/to/script.py`
		3. if the first two steps still not allow to record videos, use
			1. install `conda install -c conda-forge pyvirtualdisplay`
`