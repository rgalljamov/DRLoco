import numpy as np
import torch as th
from drloco.common import utils
from drloco.config import config as cfg

# ---------------------------------------------------
# MODIFICATIONS of the PPO algorithm,
# e.g. to achieve better sample efficiency
# ---------------------------------------------------

# use our own policy extending the ActorCriticPolicy
# to change network topology etc. Used as default mode!
MOD_CUSTOM_POLICY = 'cstm_pi'

# todo: remove? Otherwise, just use as a boolean flag
MOD_CLIPRANGE_SCHED = 'clip_sched'

# todo: shift it to the straight walker env
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_POLICY = 'mirr_py'

# specify modifications to the baseline algorithm, e.g. mirroring policy
modifications_list = [MOD_CUSTOM_POLICY, MOD_MIRR_POLICY]
modification = '/'.join(modifications_list)

def is_mod(mod_str):
    '''Simple check if a certain modification to the baseline algorithm is used,
       e.g. is_mod(MOD_MIRR_POLICY) is true, when we mirror the policy. '''
    return mod_str in modification

# Mirroring Policy only works with the Straight Walker and
# the observation space having a 1D desired velocity, a 1D phase and the COM Y position being included!
# These requirements are satisfied in the code in the moment. Once, sth. is changed,
# the mirroring functions in mujoco/mimic_env.py have to be adjusted, which is quite simple.
# If changes are made to the observation space but mirroring functions are not changed,
# the following lines should be uncommented.
# assert not is_mod(MOD_MIRR_POLICY), \
#     'Mirroring Policy can only be used with the StraightWalker. ' \
#     'AND only after changing the mirroring functions! '


# ---------------------------------------------------
# DEEPMIMIC HYPERPARAMETERS
# ---------------------------------------------------

# weights for the imitation reward
# [joint positions, joint velocities, com reward, energy]
rew_weights = [0.8, 0.2, 0, 0]

# reward scaling (set to 1 to disable)
rew_scale = 1

# alive bonus is provided as reward for each step
# the agent hasn't entered a terminal state
alive_bonus = 0.2 * rew_scale

# Early Termination: maximum steps in the environment per episode
ep_dur_max = 3000

# ---------------------------------------------------
# (PPO) HYPERPARAMETERS
#
# Detailed documentation of the PPO implementation:
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
# ---------------------------------------------------

# The discount factor we found to be optimal for different control frequencies
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[cfg.CTRL_FREQ]

# PPO samples the actions from a Gaussian Distribution.
# This hyperparameter specifies the the log standard deviation
# of the initial Gaussian Distribution at the trainings beginning
# NOTE: e^(-0.7) i  s about 0.5 which we found to be optimal for
# a normalized action space with a range of [-1,1]
init_logstd = -0.75

# batch and minibatch size
minibatch_size = 512 * 4
batch_size = (4096 * 4) if not cfg.DEBUG else 2 * minibatch_size

# we schedule the learning rate to start from lr_start
# and decrease to lr_final over the course of the training
# when lr_scale is 1. Otherwise, the slope of the schedule is
# slope = lr_scale * (lr_final - lr_start)
lr_start = 500 * (1e-6)
lr_final = 1 * (1e-6)
lr_scale = 1

# number of experiences to collect [in Millions]
mio_samples = 8

# how many parallel environments should be used to collect samples
n_envs = 8 if utils.is_remote() and not cfg.DEBUG else 1

# Neural Network hidden layer sizes
# and the corresponding activation functions
# length of the lists = number of hidden layers
hid_layer_sizes = [512]*2
activation_fns = [th.nn.Tanh]*2

# the cliprange specifies the maximum allowed distance between
# consequtive policies in order to avoid disruptive policy updates
if is_mod(MOD_CLIPRANGE_SCHED):
    clip_start = 0.55
    clip_end = 0.1
    clip_exp_slope = 5
else:
    cliprange = 0.15

# a negative entropy coefficient punishes exploration,
# a positive one encourages it
ent_coef = -0.0075

# num of times a batch of experiences
# is used to update the policy
noptepochs = 4


# ---------------------------------------------------
# BUILD MODEL PATH
#
# Based on the made hyperparameter choices,
# we construct a path to save the trained model at.
# ---------------------------------------------------

# run aka. agent id to avoid overwriting agents with same hyperparameter choices
run_id = str(np.random.randint(0, 1000))
# the part of the path constructed from hyperparameters
_mod_path = ('cfg.DEBUG/' if cfg.DEBUG else '') + \
            f'train/{modification}/{cfg.ENV_ID}/{n_envs}envs/' \
            f'{mio_samples}mio/'
# the final saving path of the current agent
save_path = utils.get_project_path() + 'models/' + _mod_path + f'{run_id}/'
