"""
Loads a specified model (by path or from config) and executes it.
The policy can be used sarcastically and deterministically.
"""
import gym, time, mujoco_py
# necessary to import custom gym environments
import gym_mimic_envs
from gym_mimic_envs.monitor import Monitor
from stable_baselines3 import PPO
from scripts.common.utils import load_env
from scripts.config import hypers as cfg
from scripts.config import config as cfgl

# paths
# PD baseline
path_pd_baseline = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                   'cstm_pi/mim3d/8envs/ppo2/16mio/918-evaled-ret71'
path_pd_normed_deltas = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                        'pi_deltas/norm_acts/cstm_pi/mim3d/8envs/ppo2/16mio/431-evaled-ret81'
path_trq_baseline = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                    'cstm_pi/mim_trq_ff3d/8envs/ppo2/8mio/296-evaled-ret79'
path_mirr_steps = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                  'steps_mirr/cstm_pi/mim_trq_ff3d/8envs/ppo2/8mio/280'
path_mirr_exps = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/' \
                 'mirr_exps/cstm_pi/mim_trq_ff3d/8envs/ppo2/16mio/331-evaled-ret86'
path_guoping = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/cstm_pi/' \
               'mirr_exps/MimicWalker3d-v0/8envs/ppo2/8mio/361'
path_140cm_40kg = '/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/cstm_pi/' \
                  'refs_ramp/mirr_exps/MimicWalker3d-v0/8envs/ppo2/16mio/197-evaled-ret78'
path_agent = cfg.abs_project_path + 'models/dmm/cstm_pi/mim_trq_ff3d/8envs/ppo2/8mio/296-evaled-ret79'
path_agent = '/mnt/88E4BD3EE4BD2EF6/Users/Sony/Google Drive/WORK/DRL/CodeTorch/models/train/' \
             'cstm_pi/refs_ramp/mirr_py/MimicWalker3d-v0/8envs/ppo2/4mio/885'

DETERMINISTIC_ACTIONS = True
RENDER = True

if cfg.env_out_torque:
    cfg.env_id = cfg.env_ids[4]
else:
    cfg.env_id = cfg.env_ids[2]

SPEED_CONTROL = False
speeds = [0.5, 1, 1.25, 1.25]
duration_secs = 8

PLAYBACK_TRAJECS = True

# which model would you like to run
FROM_PATH = False
PATH = path_agent
checkpoint = 'final' # 'ep_ret2100_20M' # '33_min24mean24' # 'ep_ret2000_7M' #'mean_rew60'

if FROM_PATH:
    if not PATH.endswith('/'): PATH += '/'

    # check if correct reference trajectories are used
    if cfg.MOD_REFS_RAMP in PATH and not cfg.is_mod(cfg.MOD_REFS_RAMP):
        raise AssertionError('Model trained on ramp-trajecs but is used with constant speed trajecs!')

    # load model
    model_path = PATH + f'models/model_{checkpoint}'
    model = PPO.load(path=model_path)
    print('\nModel:\n', model_path + '\n')

    env = load_env(checkpoint, PATH, cfg.env_id)
else:
    env_id = 'MimicWalker3d-v0' # 'MimicWalker165cm65kg-v0' # 'MimicWalker3dHip-v0' # cfg.env_id
    env = gym.make(env_id)
    env = Monitor(env)
    vec_env = env
    if PLAYBACK_TRAJECS:
        obs = vec_env.reset()
        env.activate_evaluation()
        env.playback_ref_trajectories(2000)

if not isinstance(env, Monitor):
    # VecNormalize wrapped DummyVecEnv
    vec_env = env
    env = env.venv.envs[0]

if SPEED_CONTROL:
    env.activate_speed_control(speeds, duration_secs)
    cfg.ep_dur_max = duration_secs * cfgl.CTRL_FREQ

obs = vec_env.reset()
env.activate_evaluation()

des_speeds = []
com_speeds = []

for i in range(10000):

    if FROM_PATH:
        action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
        obs, reward, done, _ = vec_env.step(action)
    else:
        if cfg.env_out_torque:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
        else:
            # try to follow desired trajecs with PD Position Controllers
            des_qpos = env.get_ref_qpos(exclude_not_actuated_joints=True)
            obs, reward, done, _ = env.step(des_qpos)

    # only stop episode when agent has fallen
    done = False and env.data.qpos[env.env._get_COM_indices()[-1]] < 0.5

    if SPEED_CONTROL:
        des_speeds.append(env.desired_walking_speed)
        com_speeds.append(env.get_qvel()[0])

    if RENDER: env.render()
    if done: env.reset()
    if SPEED_CONTROL and i >= cfg.ep_dur_max:
        from matplotlib import pyplot as plt
        plt.plot(des_speeds)
        plt.plot(com_speeds)
        plt.legend(['Desired Walking Speed', 'COM X Velocity'])
        plt.show()
        exit(33)

env.close()
