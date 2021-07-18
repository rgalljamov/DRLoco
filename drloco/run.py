"""
Loads a specified model (by path or from config) and executes it.
The policy can be used sarcastically and deterministically.
"""
# add current working directory to the system path
import sys
from os import getcwd
sys.path.append(getcwd())

from drloco.mujoco.monitor_wrapper import Monitor
from stable_baselines3 import PPO
from drloco.common.utils import load_env, get_project_path
from drloco.config import hypers as cfg
from drloco.config import config as cfgl

# do you want to run a trained agent from a given path (FROM_PATH = True)
# or do you want to run a random agent (can be used for testing purposes)
#   ..important:
#       If you load an agent from path, and the agent was trained using some modifications
#       (variable 'modifications' in hypers.py was not an empty string,
#       but e.g. MOD_MIRR_POLICY in hypers.py for policy mirroring), it is important,
#       that the modifications variable is again the same as at the moment the agent was trained!
FROM_PATH = False
# absolute path to the trained agent
#   ..important: just specify the path to the folder,
#   where the model/ and env/ folders of your agent are
path_agent = get_project_path() + 'models/dmm/cstm_pi/mim_trq_ff3d/8envs/ppo2/8mio/296-evaled-ret79'
PATH = path_agent
# we save multiple checkpoints during training. Which one should be used?
# The name of the checkpoint is the text after 'model_' or 'env_'
#   ..info: The agent saved at the end of the training has the name 'model_final'.
checkpoint = 'final' # 'ep_ret2100_20M' # '33_min24mean24' # 'ep_ret2000_7M' #'mean_rew60'


# should the agents behavior be rendered?
RENDER = True
# should the agent just play back the reference trajectories?
# Can be used for testing the reference trajectories.
PLAYBACK_TRAJECS = True

# set to True, to apply a velocity profile as desired walking speed
# during running the agent
SPEED_CONTROL = False
# specify the desired COM velocities on the profile in m/s
# we'll linearly interpolate between these velocities
# the velocities can be 1d (x-direction) or 2d: [(x1,y1), (x2,y2)],...]
speeds = [0.5, 1, 1.25, 1.25]
# how long should the desired walking speed profile be?
duration_secs = 8

assert not (PLAYBACK_TRAJECS and SPEED_CONTROL), \
    'Controlling the COM speed while playing back trajectories is not supported!'


if FROM_PATH and not PLAYBACK_TRAJECS:
    # correct path if necessary
    if not PATH.endswith('/'): PATH += '/'
    # load the trained agent from path
    model_path = PATH + f'models/model_{checkpoint}'
    model = PPO.load(path=model_path)
    # load the corresponding environment from path
    # necessary, to load the running mean and std of the observations.
    env = load_env(checkpoint, PATH, cfg.env_id)
else:
    # create a new environment as specified in the config.py
    from drloco.mujoco.config import env_map
    env = env_map[cfgl.ENV_ID]()
    env = Monitor(env)
    vec_env = env
    if PLAYBACK_TRAJECS:
        obs = vec_env.reset()
        env.activate_evaluation()
        env.playback_ref_trajectories(2000)


if SPEED_CONTROL:
    # set some flags and variables in the environment
    # to have the agent follow a desired COM velocity profile
    env.activate_speed_control(speeds, duration_secs)
    cfg.ep_dur_max = duration_secs * cfgl.CTRL_FREQ
    des_speeds = []
    com_speeds = []

# reset environment to get current state
obs = vec_env.reset()
# env.activate_evaluation()

# run the agent for 10k steps in the environment
for i in range(10000):

    # get the actions and take a step in the environment
    if FROM_PATH:
        # get actions from the loaded agent
        action, hid_states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
    else:
        # get a random action, if no agent was loaded
        # here, you can also specify your own custom actions (e.g. all zeros)
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

    # only stop episode when agent has fallen
    com_z_pos = env.get_COM_Z_position()
    done = com_z_pos < 0.5

    # log desired and actual COM speeds if speed control is active
    if SPEED_CONTROL:
        des_speeds.append(env.desired_walking_speed)
        com_speeds.append(env.get_qvel()[0])

    if RENDER: env.render()
    if done: env.reset()

    # in case speed control was active, compare the desired and actual velocities
    if SPEED_CONTROL and i >= cfg.ep_dur_max:
        from matplotlib import pyplot as plt
        plt.plot(des_speeds)
        plt.plot(com_speeds)
        plt.legend(['Desired Walking Speed', 'COM X Velocity'])
        plt.show()
        exit(33)

env.close()
