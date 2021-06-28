
.. _install:

Installation
******************

This page guides you through the installation process to get up and running with DRLoco. It should take about an hour, but might also get tricky depending on your system and require some googling to solve unexpected issues. So be prepared and best luck!

.. warning::

   This repository was developed and tested on Ubuntu 16.04 and Ubuntu 18.04 using Python 3.7. The installation instructions are the same for both Ubuntu versions and should also apply to newer Ubuntu versions.


#. Install Anaconda following the `official installation guide <https://docs.anaconda.com/anaconda/install/linux/#installation>`_.

	#. *Anaconda*, or *conda* for short, is a data science toolkit and package manager that allows you to easily install open-source python packages and organize them in separate environments.

#. Create a new conda environment with Python 3.7 and activate it (you can change the environment name *drloco* to evry other name you like)

	.. code-block:: console

	   conda create -n drloco python=3.7
	   conda activate drloco
	   

#. Install *PyTorch* using the `official installation guide <https://pytorch.org/get-started/locally/>`_ choosing the following options:

	.. image:: ../_static/images/pytorch_installation_configuration.png

#. Install *MuJoCo* and *mujoco-py* following `these instructions <https://github.com/openai/mujoco-py#install-mujoco>`_.

#. Install *Stable-Baselines 3* with the command below

	.. code-block:: console

	   pip install stable-baselines3[extra]

	#. If sth. goes wrong, remove the '[extra]'. This will leave out some packages, that you might need to install later separately, e.g. *tensorboard*.

	#. If there are still issues, please check the `official installation instructions of Stable Baselines <https://stable-baselines3.readthedocs.io/en/master/guide/install.html>`_.


#. Clone the *DRLoco* repository to your local PC (`Link to repository <https://github.com/rgalljamov/DRLoco>`_). 

	#. In the following, we refer to the absolute path of the local folder on your PC you cloned the repository to as ``/local/path/to/repo/``.

#. Install the gym_mimic_envs (this step will no longer be required in the future... in the moment it is required to register our mujoco environments.)

	1. Open a terminal and navigate to the ``mujoco`` folder, where the ``gym_mimic_envs`` folder is located. 
	2. ``pip install -e .`` to install the environment

	    1. If you get the error "ERROR: Package 'imageio' requires a different Python: 2.7.17 not in '>=3.5'", use ``pip3 install -e .`` instead. 

    	.. important:: It is important to **not** use *sudo* during installation!


#. Change :func:`is_remote` in ``scripts/common/utils.py``. Please refer to this function's doc-string for more details.

#. To check if installation was successful, run the script ``scripts/run.py``. To run the script from command line, use:

	.. code:: console
	
	   cd /local/path/to/repo/
	   python scripts/run.py

	#. If everything went well, a simulator window should appear. Press [Space] to start the simulation and you should see a MuJoCo model walking in circles for 10 seconds.

	#. If you get import errors, install the the missing packages in your conda environment.



Installing Additional Features
===============================

#. To record videos of the generated walking gait at the training's end, we need to further install ``ffmpeg``: ``conda install -c conda-forge ffmpeg``

	1. If you want to record the video on a remote server without a UI, in addtion install
		1. `sudo apt install xvfb`
		2. to execute the training script run `xvfb-run python /path/to/script.py`
		3. if the first two steps still not allow to record videos, use
			1. install `conda install -c conda-forge pyvirtualdisplay`

#. Install Weights&Biases for logging training results

	1. `conda install -c conda-forge wandb`