
Repository Overview
***********************

.. note::
	**DeepMimic(MuJoCo, Stable-Baselines3) = DRLoco**
	
This repository 

 * implements the :ref:`DeepMimic Framework <deepmim>` 
 * using the `PPO Algorithm <https://spinningup.openai.com/en/latest/algorithms/ppo.html>`_ from the `Stable-Baselines3 repository <https://stable-baselines3.readthedocs.io>`_
 * to obtain (deep reinforcement) learning-based controllers 
 * for :ref:`robots <walkers>` simulated with the `MuJoCo Physics Engine <http://www.mujoco.org/>`_.

 
File Structure
=================

In the following, the structure of the code is presented with a short description of what each individual script and folder is responsible for.

Folder *drloco*
------------------

``drloco/`` is the main folder of the framework. It contains the three most important python scripts to train, run and evaluate models as well as all the most important modules the framework is build of. 

Scripts
+++++++++

 * ``mocaps/`` contains our :ref:`motion capturing datasets <ref_trajecs>`

 	* Each dataset is explained in detail in the section :ref:`Reference Trajectories`
 
 * ``docs/`` contains the files needed to build this documentation

 * ``mujoco/`` maintains the MuJoCo environments we train our agents on.

 * ``models/`` contains a few trained agents organized by the environment they were trained on as well as the made hyperparameter choices. 

 	* .. warning::
 		For each model, we save the PPO model as well as the **corresponding environment**. The environment is necessary, because we're normalizing the observations with a running mean and standard deviation which are maintained in the environment.  


 