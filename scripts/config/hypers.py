# add current working directory to the system path 
import sys
from os import getcwd
sys.path.append(getcwd())

import numpy as np
import torch as th
from scripts.common import utils
from scripts.config import config as cfg

# todo: move to train.py
# make torch using the CPU instead of the GPU by default
if cfg.USE_CPU:
    from os import environ
    # fool python to think there is no CUDA device
    environ["CUDA_VISIBLE_DEVICES"]=""
    # to avoid massive slow-down when using torch with cpu
    import torch
    n_envs = cfg.N_PARALLEL_ENVS
    torch.set_num_threads(n_envs if n_envs<=16 else 8)

def is_mod(mod_str):
    '''Simple check if a certain modification to the baseline algorithm is used,
       e.g. is_mod(MOD_MIRR_POLICY) is true, when we mirror the policy. '''
    return mod_str in modification

# MODIFICATIONS of the PPO algorithm to achieve better sample efficiency

# todo: choice of mocaps should NOT be a modification
#  Create separate classes for both refs, possibly simply in the straight-walking-refs!
MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

# use our own policy extending the ActorCriticPolicy
# to change network topology etc. Used as default mode!
MOD_CUSTOM_POLICY = 'cstm_pi'

# todo: remove? Otherwise, just use as a boolean flag
MOD_CLIPRANGE_SCHED = 'clip_sched'

# todo: shift it to the straight walker env
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_POLICY = 'mirr_py'

# ------------------

# todo: directly access these constants from the config file
SIM_FREQ = cfg.SIM_FREQ
CTRL_FREQ = cfg.CTRL_FREQ

# todo: change all modifications to simple flags
# specify modifications to the baseline algorithm, e.g. mirroring policy
modifications_list = [MOD_CUSTOM_POLICY]
modification = '/'.join(modifications_list)

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = cfg.DEBUG_TRAINING or not sys.gettrace() is None
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8200'
# should that really matter? I think not
ent_coef = {100: -0.0075, 200: -0.0075, 400: -0.00375}[CTRL_FREQ]
init_logstd = -0.7
# todo: put all cliprange settings in the same short block
cliprange = 0.15
# only for cliprange scheduling
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

# just for logging to wandb
walker_xml_file = cfg.ENV_ID

hid_layer_sizes = cfg.hid_layer_sizes
activation_fns = [th.nn.Tanh]*2
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[CTRL_FREQ]
rew_scale = 10
alive_bonus = 0.2 * rew_scale
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# num of times a batch of experiences is used
noptepochs = 4

# ----------------------------------------------------------------------------------

# number of experiences to collect, not training steps.
mio_samples = cfg.MIO_SAMPLES
n_envs = cfg.N_PARALLEL_ENVS if utils.is_remote() and not DEBUG else 1
minibatch_size = 512 * 4
batch_size = (4096 * 4) if not DEBUG else 2*minibatch_size
# to make PHASE based mirroring comparable with DUP, reduce the batch size
if is_mod(MOD_MIRR_POLICY): batch_size = int(batch_size / 2)

lr_start = 500 * (1e-6)
lr_final = 1 * (1e-6)
ep_dur_max = cfg.MAX_EPISODE_STEPS # int(_ep_dur_in_k * 1e3)
max_distance = cfg.MAX_WALKING_DISTANCE

# todo: move to train.py
# construct the paths to store the models at
run_id = str(np.random.randint(0, 1000))
_mod_path = ('debug/' if DEBUG else '') + \
            f'train/{modification}/{cfg.ENV_ID}/{n_envs}envs/' \
            f'{mio_samples}mio/'
save_path_norun= utils.get_absolute_project_path() + 'models/' + _mod_path
save_path = save_path_norun + f'{run_id}/'

if __name__ == '__main__':
    from scripts.train import train
    train()