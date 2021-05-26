# workaround to start scripts from cmd on any remote server
import sys

sys.path.append('/home/rustam/code/torch/')

import numpy as np
import torch as th
from scripts.common import utils
from scripts.config import config as cfg

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

# todo: do we need it here?
# get the absolute path of the current project
abs_project_path = utils.get_absolute_project_path()

# modifications / modes of the approach
MOD_FLY = 'fly'


# todo: choice of mocaps should be a modification
MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

# use our own policy extending the ActorCriticPolicy
# to change network topology etc. Used as default mode!
MOD_CUSTOM_POLICY = 'cstm_pi'

MOD_CLIPRANGE_SCHED = 'clip_sched'

# use symmetrized mocap data for imitation reward
MOD_SYMMETRIC_WALK = 'sym_walk'

# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_POLICY = 'mirr_py'

# todo: do we need it? Especially when we'll use other mocaps soon?
# only when training to accelerate with the velocity ramp trajectories
SKIP_N_STEPS = 1
STEPS_PER_VEL = 1

# ------------------

# todo: directly access these constants from the config file
SIM_FREQ = cfg.SIM_FREQ
CTRL_FREQ = cfg.CTRL_FREQ

# specify modifications to the baseline algorithm, e.g. mirroring policy
modifications_list = [MOD_CUSTOM_POLICY, MOD_REFS_RAMP, MOD_MIRR_POLICY]
modification = '/'.join(modifications_list)

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = cfg.DEBUG_TRAINING or not sys.gettrace() is None
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!

rew_weights = '8200'
ent_coef = {200: -0.0075, 400: -0.00375}[CTRL_FREQ]
init_logstd = -0.7
pi_out_init_scale = 0.001
cliprange = 0.15
# only for cliprange scheduling
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

# just for logging to wandb
peak_joint_torques = cfg.PEAK_JOINT_TORQUES
walker_xml_file = cfg.WALKER_MJC_XML_FILE

hid_layer_sizes = cfg.hid_layer_sizes
activation_fns = [th.nn.Tanh]*2
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[CTRL_FREQ]
rew_scale = 1
alive_bonus = 0.2 * rew_scale
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# num of times a batch of experiences is used
noptepochs = 4

# ----------------------------------------------------------------------------------

# choose environment
if cfg.ENV_ID is not None:
    env_id = cfg.ENV_ID
    # short description of the env used in the save path
    env_abbrev = env_id
    env_is3d = True
    env_out_torque = cfg.ENV_OUT_TORQUE
    env_ids = ['MimicWalker2d-v0', 'MimicWalker2d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
    env_abbrevs = ['mim2d', 'mim_trq2d', 'mim3d', 'mim_trq3d', 'mim_trq_ff3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']

else:
    env_ids = ['MimicWalker2d-v0', 'MimicWalker2d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'MimicWalker3d-v0', 'Walker2d-v2', 'Walker2d-v3', 'Humanoid-v3', 'Blind-BipedalWalker-v2', 'BipedalWalker-v2']
    env_abbrevs = ['mim2d', 'mim_trq2d', 'mim3d', 'mim_trq3d', 'mim_trq_ff3d', 'walker2dv2', 'walker2dv3', 'humanoid', 'blind_walker', 'walker']
    env_index = 4
    env_id = env_ids[env_index]
    # used in the save path (e.g. 'wlk2d')
    env_abbrev = env_abbrevs[env_index]
    env_out_torque = True
    env_is3d = True

# choose hyperparams
algo = 'ppo2'
# number of experiences to collect, not training steps.
mio_samples = cfg.MIO_SAMPLES
n_envs = cfg.N_PARALLEL_ENVS if utils.is_remote() and not DEBUG else 2
minibatch_size = 512 * 4
batch_size = (4096 * 2) if not DEBUG else 2*minibatch_size
# to make PHASE based mirroring comparable with DUP, reduce the batch size
if is_mod(MOD_MIRR_POLICY): batch_size = int(batch_size / 2)

lr_start = 500 * (1e-6)
lr_final = 1 * (1e-6)
_ep_dur_in_k = {400: 6, 200: 3, 100: 1.5, 50: 0.75}[CTRL_FREQ]
ep_dur_max = cfg.MAX_EPISODE_STEPS # int(_ep_dur_in_k * 1e3)
max_distance = cfg.MAX_WALKING_DISTANCE

run_id = str(np.random.randint(0, 1000))
info_baseline_hyp_tune = f'hl{str(hid_layer_sizes)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew{rew_weights}_gam{int(gamma*1e3)}'

# construct the paths to store the models at
_mod_path = ('debug/' if DEBUG else '') + \
            f'train/{modification}/{env_abbrev}/{n_envs}envs/' \
            f'{algo}/{mio_samples}mio/'
save_path_norun= abs_project_path + 'models/' + _mod_path
save_path = save_path_norun + f'{run_id}/'

if __name__ == '__main__':
    from scripts.train import train
    train()