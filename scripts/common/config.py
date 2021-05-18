# suppress the annoying TF Warnings at startup
import warnings, sys
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# workaround to start scripts from cmd on remote server
sys.path.append('/home/rustam/code/torch/')

import numpy as np
from scripts.common import utils
from scripts import config_light as cfgl

# make torch using the CPU instead of the GPU by default
if cfgl.USE_CPU:
    from os import environ
    # fool python to think there is no CUDA device
    environ["CUDA_VISIBLE_DEVICES"]=""
    # to avoid massive slow-down when using torch with cpu
    import torch
    n_envs = cfgl.N_PARALLEL_ENVS
    torch.set_num_threads(n_envs if n_envs<=16 else 8)


def s(input):
    """ improves conversion of digits to strings """
    if isinstance(input, list):
        str_list = [str(item) for item in input]
        res = ''.join(str_list)
        return res
    return str(input).replace('.','')

def mod(mods:list):
    modification = ''
    for mod in mods:
        modification += mod + '/'
    # remove last /
    modification = modification[:-1]
    return modification

def get_torque_ranges(hip_sag, hip_front, knee, ankle):
    torque_ranges = np.ones((8,2))
    peaks = np.array([hip_sag, hip_front, knee, ankle] * 2)
    torque_ranges[:,0] = -peaks
    torque_ranges[:,1] = peaks
    # print('Torque ranges (hip_sag, hip_front, knee, ankle): ', torque_ranges)
    return torque_ranges

def is_mod(mod_str):
    return mod_str in modification

# get the absolute path of the current project
abs_project_path = utils.get_absolute_project_path()

# approaches
AP_DEEPMIMIC = 'dmm'

# modifications / modes of the approach
MOD_FLY = 'fly'
MOD_ORIG = 'orig'

MOD_REFS_CONST = 'refs_const'
MOD_REFS_RAMP = 'refs_ramp'

# use our own policy extending the ActorCriticPolicy
# to change network topology etc. Used as default mode!
MOD_CUSTOM_POLICY = 'cstm_pi'
# let the policy output deltas to current angle
MOD_PI_OUT_DELTAS = 'pi_deltas'

# use a tanh activation function at the output layer
MOD_BOUND_MEAN = 'tanh_mean'
# bound actions as done in SAC: apply a tanh to sampled actions
# and consider that squashing in the prob distribution, e.g. logpi calculation
MOD_SAC_ACTS = 'sac_acts'
# use running statistics from previous runs
MOD_LOAD_OBS_RMS = 'obs_rms'
init_obs_rms_path = abs_project_path + 'models/behav_clone/models/rms/env_999'

# load pretrained policy (behavior cloning)
MOD_PRETRAIN_PI = 'pretrain_pi'
# init the weights in the output layer of the value function to all zeros
MOD_VF_ZERO = 'vf_zero'

# mirror experiences # TODO: Will this still be possible with domain randomization?
MOD_MIRROR_EXPS = 'mirr_exps'
# query the policy and the value functions to get neglogpacs and values
MOD_QUERY_NETS = 'query_nets'
# use linear instead of exponential reward to have better gradient away from trajecs
MOD_QUERY_VF_ONLY = 'query_vf_only'

# multiply the individual reward components instead of using a weighted sum
MOD_REW_MULT = 'rew_mult'
# use linear instead of exponential function in the reward calculation
MOD_LIN_REW = 'lin_rew'
# use com x velocity instead of x position for com reward
MOD_COM_X_VEL = 'com_x_vel'

# use reference trajectories as a replay buffer
MOD_REFS_REPLAY = 'ref_replay'

# train multiple networks for different phases (left/right step, double stance)
MOD_GROUND_CONTACT_NNS = 'grnd_contact_nns'
MOD_3_PHASES = '3_phases'
MOD_CLIPRANGE_SCHED = 'clip_sched'
# use symmetrized mocap data for imitation reward
MOD_SYMMETRIC_WALK = 'sym_walk'
# reduce input dimensionality with an end-to-end encoder network of the observations
# e2e means here that we don't separately train the encoder to reconstruct the observations
MOD_E2E_ENC_OBS = 'e2e_enc_obs'
MOD_L2_REG = 'l2_reg'
l2_coef = 5e-4
# set a fixed logstd of the policy
MOD_CONST_EXPLORE = 'const_explor'
# learn policy for right step only, mirror states and actions for the left step
MOD_MIRR_POLICY = 'mirr_py'
MOD_EXP_REPLAY = 'exp_replay'
replay_buf_size = 1

# only when training to accelerate with the velocity ramp trajectories
SKIP_N_STEPS = 1
STEPS_PER_VEL = 1

# ------------------
approach = AP_DEEPMIMIC
SIM_FREQ = cfgl.SIM_FREQ
CTRL_FREQ = cfgl.CTRL_FREQ
# DO NOT CHANGE default modifications
modification = MOD_CUSTOM_POLICY + '/'
# HERE modifications can be added
modification += mod([MOD_MIRROR_EXPS])
modification = 'sb3/' + mod([MOD_MIRR_POLICY]) # mod([MOD_CLIPRANGE_SCHED]) #

# ----------------------------------------------------------------------------------
# Weights and Biases
# ----------------------------------------------------------------------------------
DEBUG = cfgl.DEBUG_TRAINING or not sys.gettrace() is None
MAX_DEBUG_STEPS = int(2e4) # stop training thereafter!
TORQUE_RANGES = get_torque_ranges(*cfgl.PEAK_JOINT_TORQUES)

rew_weights = '8200' if not is_mod(MOD_FLY) else '7300'
ent_coef = {200: -0.0075, 400: -0.00375}[CTRL_FREQ]
init_logstd = -0.7
pi_out_init_scale = 0.001
cliprange = 0.15
clip_start = 0.55 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_end = 0.1 if is_mod(MOD_CLIPRANGE_SCHED) else cliprange
clip_exp_slope = 5

# just for logging to wandb
peak_joint_torques = cfgl.PEAK_JOINT_TORQUES
walker_xml_file = cfgl.WALKER_MJC_XML_FILE

enc_layer_sizes = [512]*2 + [16]
hid_layer_sizes_vf = cfgl.hid_layer_sizes_vf
hid_layer_sizes_pi = cfgl.hid_layer_sizes_pi
gamma = {50:0.99, 100: 0.99, 200:0.995, 400:0.998}[CTRL_FREQ]
rew_scale = 1
alive_bonus = 0.2 * rew_scale
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# num of times a batch of experiences is used
noptepochs = 4

wb_project_name = cfgl.WB_PROJECT_NAME
wb_run_name = ('SYM ' if is_mod(MOD_SYMMETRIC_WALK) else '') + \
               cfgl.WB_EXPERIMENT_NAME
wb_run_notes = cfgl.WB_EXPERIMENT_DESCRIPTION
# ----------------------------------------------------------------------------------

# choose environment
if cfgl.ENV_ID is not None:
    env_id = cfgl.ENV_ID
    # short description of the env used in the save path
    env_abbrev = env_id
    env_is3d = True
    env_out_torque = cfgl.ENV_OUT_TORQUE
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
# In case of mirroring, during 4M training steps, we collect 8M samples.
mirr_exps = is_mod(MOD_MIRROR_EXPS)
exp_replay = is_mod(MOD_EXP_REPLAY)
mio_samples = cfgl.MIO_SAMPLES
if mirr_exps: mio_samples *= 2
n_envs = cfgl.N_PARALLEL_ENVS if utils.is_remote() and not DEBUG else 2
minibatch_size = 512 * 4
batch_size = (4096 * 2 * (2 if not mirr_exps else 1)) if not DEBUG else 2*minibatch_size
# to make PHASE based mirroring comparable with DUP, reduce the batch size
if is_mod(MOD_MIRR_POLICY): batch_size = int(batch_size / 2)
# if using a replay buffer, we have to collect less experiences
# to reach the same batch size
if exp_replay: batch_size = int(batch_size/(replay_buf_size+1))

lr_start = 500 * (1e-6)
lr_final = 1 * (1e-6)
_ep_dur_in_k = {400: 6, 200: 3, 100: 1.5, 50: 0.75}[CTRL_FREQ]
ep_dur_max = cfgl.MAX_EPISODE_STEPS # int(_ep_dur_in_k * 1e3)
max_distance = cfgl.MAX_WALKING_DISTANCE

run_id = s(np.random.random_integers(0, 1000))
info_baseline_hyp_tune = f'hl{s(hid_layer_sizes_vf)}_ent{int(ent_coef * 1000)}_lr{lr_start}to{lr_final}_epdur{_ep_dur_in_k}_' \
       f'bs{int(batch_size/1000)}_imrew{rew_weights}_gam{int(gamma*1e3)}'

# construct the paths to store the models at
_mod_path = ('debug/' if DEBUG else '') + \
            f'{approach}/{modification}/{env_abbrev}/{n_envs}envs/' \
            f'{algo}/{mio_samples}mio/'
save_path_norun= abs_project_path + 'models/' + _mod_path
save_path = save_path_norun + f'{run_id}/'

# wandb
def get_wb_run_name():
    return wb_run_name

# names of saved model before and after training
init_checkpoint = 'init'
final_checkpoint = 'final'

if __name__ == '__main__':
    from scripts.train import train
    train()