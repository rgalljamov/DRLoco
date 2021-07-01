import sys
import numpy as np
import torch as th
from drloco.common import utils
from drloco.config import config as cfg

# ---------------------------------------------
# MODIFICATIONS of the PPO algorithm,
# e.g. to achieve better sample efficiency
# ---------------------------------------------

# use our own policy extending the ActorCriticPolicy
# to change network topology etc. Used as default mode!
MOD_CUSTOM_POLICY = 'cstm_pi'

# todo: remove? Otherwise, just use as a boolean flag
MOD_CLIPRANGE_SCHED = 'clip_sched'

# todo: shift it to the straight walker env
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_POLICY = 'mirr_py'

# specify modifications to the baseline algorithm, e.g. mirroring policy
modifications_list = [MOD_CUSTOM_POLICY]
modification = '/'.join(modifications_list)

def is_mod(mod_str):
    '''Simple check if a certain modification to the baseline algorithm is used,
       e.g. is_mod(MOD_MIRR_POLICY) is true, when we mirror the policy. '''
    return mod_str in modification

assert not is_mod(MOD_MIRR_POLICY), \
    'Mirroring Policy can only be used with the StraightWalker. ' \
    'AND only after changing the mirroring functions! '

# ---------------------------------------------

# delete and directly ask for cfg.DEBUG
DEBUG = cfg.DEBUG or not sys.gettrace() is None
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8200'
ent_coef = -0.0075
init_logstd = -0.7
# todo: put all cliprange settings in the same short block
cliprange = 0.15
# only for cliprange scheduling
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

# network hidden layer sizes
hid_layer_sizes = [512]*2
activation_fns = [th.nn.Tanh]*2
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[cfg.CTRL_FREQ]
rew_scale = 1
alive_bonus = 0.2 * rew_scale
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# num of times a batch of experiences is used
noptepochs = 4

# ----------------------------------------------------------------------------------

# number of experiences to collect [in Millions]
mio_samples = 8
# how many parallel environments should be used to collect samples
n_envs = 8 if utils.is_remote() and not DEBUG else 1
minibatch_size = 512 * 4
batch_size = (4096 * 4) if not DEBUG else 2*minibatch_size
# to make PHASE based mirroring comparable with DUP, reduce the batch size
if is_mod(MOD_MIRR_POLICY): batch_size = int(batch_size / 2)

lr_start = 500 * (1e-6)
lr_final = 1 * (1e-6)
# LR decay slope scaling: slope = lr_scale * (lr_final - lr_start)
# the decay is linear from lr_start to lr_final
lr_scale = 1
# maximum steps in the environment per episode
ep_dur_max = 1000

# todo: move to train.py or utils
# construct the paths to store the models at
run_id = str(np.random.randint(0, 1000))
_mod_path = ('debug/' if DEBUG else '') + \
            f'train/{modification}/{cfg.ENV_ID}/{n_envs}envs/' \
            f'{mio_samples}mio/'
save_path_norun= utils.get_project_path() + 'models/' + _mod_path
save_path = save_path_norun + f'{run_id}/'
