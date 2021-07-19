
.. _run:

Running an agent
**********************

To run a trained agent, specify the path to the agent of interest in ``run.py``. In this script, you in addition have multiple flags to set. Each line of the code is documented in detail by comments. The most important flags are still described in more detail here:

* :mod:`FROM_PATH` is a boolean flag allowing you to distinguish if you want to run a trained agent saved on your local PC (True) or just running a random agent on the environment specified in ``config/config.py``. The latter might be helpful for testing purposes.

* :mod:`PLAYBACK_TRAJECS` can be set to ``True`` to playback the corresponding reference trajectories on the current environment (set in ``config/config.py``). It is useful to test the configuration of mujoco models as well as their compatibility with the reference trajectories. Use this functionality every time you add a new or change existing mujoco models or reference trajectories.   

* :mod:`SPEED_CONTROL` allows you to specify a velocity profile for the walkers COM. Therefore, discrete desired velocities are specified in a list. The desired continuous COM velocity profile is constructed by linearly interpolating between the desired velocities.


.. important::

   To successfully run an agent, all configurations like the modifications to the PPO algorithm (variable *modification* in $config/hypers.py$), the observation and action spaces as well as other modifications to the environment have to be the same as during the training of the agent.

   Counterexample: The agent was trained by using the phase variable. During running the agent, instead of the true phase variable, its estimation from the hip joint is used. This setting will not result in any crashes, as the observation space dimensionality stays the same. However, the agent will very likely perform poorly.


WHEN do we save agents?
=======================

During the training of an agent, we save multiple checkpoints:

* initial agent: PPO with random networks directly after initialization and before training

* final agent: PPO agent after the end of the training

* agents reaching certain performance thresholds specified in ``common/callback.py``.

	* reaching 50, 60, ... 90% of maximum possible imitation reward

	* reaching 50, 60, ... 90% of maximum possible episode return

.. important::
   
   Each agent is always saved with a corresponding environment. This environment has to be loaded together with the agent to run the agent. This is done automatically within this framework. 

   **Details:** We train our agents on `VecNormalize` environment wrappers. These wrappers maintain a running mean and std of the observations (dimension-vise) to normalize the inputs into the policy and value function networks. Therefore, when loading an agent, we also have to wrap the corresponding environment in a VecNormalize object. This is done automatically within this framework. 
   

WHAT info do we save for each agent? 
======================================

We use the *save* and *load* functionalities of Stable-Baselines to save and load our agents. However, next to the agent, we also need to save the environment it was saved on. It is required to save the environment, because we are using the `VecNormalize` wrapper to noramlize the environment observations by their running mean and standard deviation. Both statistics are saved as part of the `VecNormalize` environment. 



WHERE do we save trained agents?
================================

Saving and loading of agents is implemented in ``common/utils.py``. 

At the beginning of the training, we create a folder for the corresponding agent (implementation in ``train.py``). The path to this folder is constructed from some of the used hyperparameters as well as a random run or agent id (random number between 0 and 999). The path is constructed in the end of ``config/hypers.py``.

The folder contains multiple sub-folders, e.g. ``models/`` for saving the checkpoints of the agent or ``envs/`` for saving the corresponding environments.
