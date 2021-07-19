
Configuration and Hyperparameters
****************************************

Our repository contains two main configuration files stored in the ``config/`` folder. ``config.py`` specifies the higher-level settings like whether to use the CPU or GPU for training, the trainining environment and the logging-details if W&B is used. hypers.py`` is focusing on the hyperparameters of the PPO algorithm including specified modifications to the algorithm like mirroring the policy. 

Training Configuration
========================

Implementation: ``config/config.py``

``config.py`` specifies the higher level details of training and running agents. The most important settings are described in detail below. Please also refer to the details comments in code.

* The main purpose of the file is the specification of a gym environment. Here, it is sufficient to only name the ID of the environment. The IDs of all available environments are specified in ``mujoco/config.py``.

* In addition, a control frequency can be specified as well as

* the settings for evaluating the walking performance of the agent 

* finally, the 

Hyperparameters and PPO modifications
============================================

Implementation: ``config/hypers.py``

In ``hypers.py``, we specify the hyperparameters of the PPO algorithm as well as possible modifications like mirroring the policy and using a schedule for PPO's cliprange.


Important Hyperparameters
-------------------------------

In the following, hyperparameters we found to be important in achieving stable walking are explained together with our insights. A detailed (technical) explanation of all PPO's hyperparameters can be found in the Stable-Baselines-Documentation of the algorithm `here <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters>`_. 

* we've found the **discount factor gamma** to be crucial. The discount factor balances the importance of the short- and long-term rewards. It can be interpreted as specifying the number of future timesteps to be considered in the return calculation of a single state-action pair. It is therefore highly dependend on the control frequency. We've found a value of 0.995 to work best for 200Hz control frequency. At this frequency, the factor considers 200 timesteps equivalent to 1 second of simulated time. Interestingly, this approximately corresponds to the duration of a whole gait cycle consisting of two steps. Please see `this stackoverflow post <https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning/428157#428157>`_ for details on how to determine how many steps a discounting factor corresponds to.

* another important hyperparameter is the logarithm of the **standard deviation of the initial policy** (Gaussian action distribution). As SB3 clips the actions to the range of [-1,1], we found it logical and useful to initialize the policy with a smaller standard deviation of 0.5 (or -0.7 for the logstd) instead of the default logstd of 0 corresponding to a standard deviation of 1. This avoids clipping of actions which results in different actions being applied in the same way in the environment (e.g. all actions above 1 are inputed into the environment as 1). This significantly improves the training signal and makes the calculation of the entropy and action probabilities more precise which improves the policy gradient estimation. 
	
	* even smaller logstds at policy initialization of up to -1.2 have been observed to result in even faster learning! But, systematic comparison of different values have not yet been 

* an **alive-bonus** of about 20% of the maximum possible reward at each interaction with the environment has shown significant increase in learning speed towards robust walking. This bonus is just added to the imitation reward.

* **learning rate** and **cliprange**: For both parameters we found it helpful to use scheduling. For the learning rate, we use a linear decay, while for the cliprange an exponential decay. We believe this way to give the agent the possibility to advance quickly at the beginning and go away from the random initial policy, while allowing fine-tuning once the policy generates better and better behavior.

	.. important:: 

	   All our schedules are implemented to decrease from a starting value to a final value over the course of the training duration. Therefore, by changing the training duration (e.g. from 8M steps to 16M steps) the slope of the decay is also changing. 

	   An agent that has almost reached stable walking when training it for 4M steps, might not reach stable walking when increasing the training duration to 8M steps as the learning rate during the first 4M steps also changes. (As a hotfix to this curcumstances, we've introduced a scaling factor to increase or decrease the slope of a linear schedule. However, a scalar of 2 will decay the learning rate from a start to a final value in half the training time and continue training with the final learning rate value until the end of the training.) 

* **entropy coefficient** allows to punish and encourage exploration. In the PPO implementation of SB3 it is set to zero. Increasing it, encourages exploration and was proposed by the PPO developers to avoid collapsing of the action distribution to a single value (std = 0). However, we've not observed collapsing but on the contrary found it helpful to punish entropy. This way, the agent converges to a more stable policy. 

* **number of epochs per batch of experiences** (*noptepochs* in our code) is the number of times, each sample from a batch of experiences is used for policy optimization. While the default value in SB3 is 10, we found smaller values like 4 to result in faster learning.


PPO Modifications
-------------------------

To increase the sample-efficiency of learning stable walking, we've implemented multiple modifications to the original PPO implementation in SB3.

* **Cliprange Scheduling**: actually just a hyperparameter choice, making the cliprange decrease exponentially from 0.55 to 0.1 instead of using a constant value of 0.15. This way, we allow bigger changes to the poor performing random initial policy and limit the difference of consecutive policies in later phases of the training.

* **Mirroring Policy**: The idea is to let the agent learn to step only with the left foot. The actions for stepping with the right foot are symmetric. Therefore, for a right foot step, we mirror the observations (to get corresponding observations of stepping with the left foot), get the actions for the left foot and then mirror the actions (applying the left hip torques on the right hip etc.).
	
	.. important:: Policy mirroring is currently only implemented for the Straight Walker which gets a 1D desired COM velocity (in x direction) and a 1D phase variable. 

	The information about the side of the step (whether it is a step with the left or the right foot) has to be provided from the reference trajectories in the moment, which is also a major drawback for using the trained policy on a real robot where no ref data is available.

* **Custom Policy**: This mode just makes the PPO agent custom hidden layers for the policy and value function networks as defined in ``custom/policies.py``. In the moment however, the custom hidden layers are identical to the hidden layer implementation of SB3. But, the code is well documented, so it should be easy to use custom network architectures for each of the networks. 