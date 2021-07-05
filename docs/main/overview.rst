
Repository Overview
***********************

.. note::
	**DRLoco = DeepMimic(MuJoCo, StableBaselines3)**

	Abbreviations:
		* SB3: `Stable-Baselines 3 <https://stable-baselines3.readthedocs.io>`_  

		* W&B: `Weights & Biases <https://wandb.ai>`_
	

This repository 

 * implements the :ref:`DeepMimic Framework <deepmim>` 
 * using the `PPO Algorithm <https://spinningup.openai.com/en/latest/algorithms/ppo.html>`_ from the `Stable-Baselines3 repository <https://stable-baselines3.readthedocs.io>`_(SB3)
 * to obtain (deep reinforcement) learning-based controllers 
 * for :ref:`robots <walkers>` simulated with the `MuJoCo Physics Engine <http://www.mujoco.org/>`_.

 
File Structure 
=================

In the following, the structure of the code is presented with a short description of what each individual script and folder is responsible for. The main folder :ref:`drloco <drloco_folder>` is explained in a separate section below.


.. _drloco_folder:

Folder *drloco*
=================

``drloco/`` is the main folder of the framework. It contains the three most important python scripts to train, run and evaluate models as well as all the most important modules the framework is build of. In the following, each subfolder and their files are presented shortly.

	* ``config/`` contains training configurations 

		* ``config.py`` is the main config file. It defines the meta data of training, like a description, as well as higher-level hyperparameters like the MuJoCo model to use, the simulation frequency, whether to use the CPU or GPU for training etc.

		* ``hypers.py`` define the hyperparameters on different lower levels, like DeepMimic configurations (e.g. imitation reward weights) and the PPO hyperparameters as well as common hyperparameters like the training duration and learning rate schedule parameters.


	* ``common/`` contains utility functions
		
		* ``callback.py`` defines a callback class that is triggered at every step taken in the environment as well as at the trainings beginning and the end. It is used to monitor the training performance and log it to Tensorboard and W&B. In addition, the training evaluation is implemented in this script.

		.. note::

		   We evaluate the training performance by pausing the training every N steps, and running the so far trained model on 20 consecutive evaluation episodes. The evaluation interval ``N`` depends on the training performance: N is 400k at the beginning (0 stable walks), changes to 200k after the agent reached the end of an evaluation episode (1-19 stable walks) and is again 400k after the agent converged to stable walking (20 stable walks). 

		   Each evaluation episode is initialized at a slightly different position on the reference trajectory. The desired velocity is set to a constant forward walking motion. We monitor the minimum and average walked distance across the 20 evaluation episodes, as well as number of stable walks defined as reaching a minimum distance without falling (e.g. 20m, defined in ``config/config.py``).

		* ``schedules.py`` contains the linear and exponential decay schedules. The linear scheduling is used for the learning rate in the moment, while the exponential decay is for *cliprage* scheduling.

		* ``utils.py`` contains helper functions like determening the project's path, whether the code is running on the remote or local server, smoothing functions etc.


	* ``custom/`` contains own implementations and extensions of the PPO algorithm and training procedure as defined in SB3. 

		* ``policies.py`` implements a custom actor critic policy to use with PPO. In the moment, the policy is just a re-implementation of the default policy in SB3 with the extra feature, that the policy and value function network architectures (number of hidden layers, hidden layer units and activation functions) can be specified in ``config/hypers.py``.


	* ``ref_trajecs/`` contains the necessary classes to provide a set of reference trajectories (e.g. motion capturing data) for imitation learning. 

		* ``base_ref_trajecs.py`` implements an interface, that is used in the mod:`MimicEnv` and provides the reference motion in an appropriate format during the training.

		* ``straight_walk_trajecs.py`` extends the base class to use the :ref:`straight walking trajectories <mocaps>`. The corresponding reference motion is a motion captured walk on a threadmill. 

		.. note:

		   There are two sets of straight walking trajectories stored in ``mocaps/straight_walking/``:

		   * ``Trajecs_Constant_Speed_400Hz.mat`` represent 30 steps of almost constant speed walking on a threadmill. 

		   * ``Trajecs_Ramp_Slow_400Hz[...].mat`` captures 250 steps on a threadmill which was accelerating half the time from 0.7 to 1.2m/s and descellerating back from 1.2 to 0.7m/s in the second half of the experiment.

		* ``loco3d_trajecs.py`` provides another set of a reference motion during the training procedure: walking an eight shape, walking straight, walking at constant speeds, accelerating and descellerating etc.


.. _mujoco_folder:

Folder *mujoco*
=================



Other Folders
=================

 * ``docs/`` contains the files needed to build this documentation

 * ``mocaps/`` contains our :ref:`motion capturing datasets <ref_trajecs>`

 	* Each dataset is explained in detail in the section :ref:`Reference Trajectories`

 * ``models/`` contains a few trained agents organized by the environment they were trained on as well as the made hyperparameter choices. 

 	* .. warning::

 		For each model, we save the PPO model as well as the **corresponding environment**. The environment is necessary, because we're normalizing the observations with a running mean and standard deviation which are maintained in the environment.

 	* **TODO: save at least one model here!**  

 * ``wandb/`` contains a few scripts to download the data logged to W&B for further postpocessing and analysis. It is optional. 

 