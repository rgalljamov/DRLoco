import os.path
import glob, wandb
import numpy as np

import drloco.config.config
import drloco.train
from drloco.common import utils
from drloco.config import hypers as cfg

from stable_baselines3 import PPO
plt = utils.import_pyplot()

RENDER = True and not utils.is_remote()
NO_ET = True
PLOT_RESULTS = False
DETERMINISTIC_ACTIONS = True

FROM_PATH = True
PATH = "/mnt/88E4BD3EE4BD2EF6/Masters/M.Sc. Thesis/Code/models/dmm/mirr_exps/cstm_pi/pi_deltas/norm_acts/" \
       "mim3d/8envs/ppo2/1.0mio/672-evaled-ret-12901"
if not PATH.endswith('/'): PATH += '/'

# evaluate for n episodes
n_eps = 10
# how many actions to record in each episode
rec_n_steps = 1000

def eval_model(run_id, checkpoint):

    utils.log('MODEL EVALUATION STARTED')

    # change save_path to specified model
    if FROM_PATH:
        save_path = PATH
    else:
        save_path = cfg.save_path

    env = utils.load_env(checkpoint, save_path, cfg.env_id)
    mimic_env = env.venv.envs[0]
    mimic_env.activate_evaluation()

    # load model
    model_path = save_path + f'models/model_{checkpoint}.zip'
    model = PPO.load(model_path, env)

    print('\nModel:\n', model_path + '\n')

    ep_rewards, all_returns, ep_durations = [], [], []
    all_rewards = np.zeros((n_eps, rec_n_steps))
    all_actions = np.zeros((n_eps, env.action_space.shape[0], rec_n_steps))

    ep_count, ep_dur = 0, 0
    obs = env.reset()

    if not RENDER: print('Episodes finished:\n0 ', end='')

    while True:
        ep_dur += 1
        action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
        if ep_dur <= rec_n_steps:
            all_actions[ep_count, :, ep_dur - 1] = action
        obs, reward, done, info = env.step(action)
        ep_rewards += [reward[0] if isinstance(reward,list) else reward]
        done_is_scalar = isinstance(done, bool) or \
                         isinstance(done, np.bool_) or isinstance(done, np.bool)
        done_is_list = isinstance(done, list)
        done = (done_is_scalar and done) or (done_is_list and done.any())
        # end the episode also after a max amount of steps
        done = done or (ep_dur > 2000)
        if done:
            ep_durations.append(ep_dur)
            # clip ep_dur to max number of steps to save
            ep_dur = min(ep_dur, rec_n_steps)
            all_rewards[ep_count,:ep_dur] = np.asarray(ep_rewards)[:ep_dur].flatten()
            ep_return = np.sum(ep_rewards)
            all_returns.append(ep_return)
            if RENDER: print('ep_return: ', ep_return)
            # reset all episode specific containers and counters
            ep_rewards = []
            ep_dur = 0
            ep_count += 1
            # stop evaluation after enough episodes were observed
            if ep_count >= n_eps: break
            elif ep_count % 5 == 0:
                print(f'-> {ep_count}', end=' ', flush=True)
            env.reset()

        if RENDER: env.render()
    env.close()

    mean_return = np.mean(all_returns)
    print('\n\nAverage episode return was: ', mean_return)

    # create the metrics folder
    metrics_path = save_path + f'metrics/model_{checkpoint}/'
    os.makedirs(metrics_path, exist_ok=True)

    np.save(metrics_path + '/{}_mean_ret_on_{}eps'.format(int(mean_return), n_eps), mean_return)
    np.save(metrics_path + '/rewards', all_rewards)
    np.save(metrics_path + '/actions', all_actions)
    np.save(metrics_path + '/ep_returns', all_returns)
    np.save(metrics_path + '/ep_durations_' + str(int(np.mean(ep_durations))), ep_durations)

    SAVE_NUM_OF_PARAMS = False
    if SAVE_NUM_OF_PARAMS:
        # count and save number of parameters in the model
        num_policy_params = np.sum([np.prod(tensor.get_shape().as_list())
                                    for tensor in model.params if 'pi' in tensor.name])
        num_valfunc_params = np.sum([np.prod(tensor.get_shape().as_list())
                                    for tensor in model.params if 'vf' in tensor.name])
        num_params = np.sum([np.prod(tensor.get_shape().as_list())
                             for tensor in model.params])

        count_params_dict = {'n_pi_params': num_policy_params, 'n_vf_params':num_valfunc_params,
                             'n_params': num_params}

        np.save(metrics_path + '/weights_count', count_params_dict)
        print(count_params_dict)

    # mark special episodes
    ep_best = np.argmax(all_returns)
    ep_worst = np.argmin(all_returns)
    ep_average = np.argmin(np.abs(all_returns - mean_return))
    relevant_eps = [ep_best, ep_worst, ep_average]

    if PLOT_RESULTS:
        plt.plot(all_returns)
        plt.plot(np.arange(len(all_returns)),
                 np.ones_like(all_returns)*mean_return, '--')
        plt.vlines(relevant_eps, ymin=min(all_returns), ymax=max(all_returns),
                   colors='#cccccc', linestyles='dashed')
        plt.title(f"Returns of {n_eps} epochs")
        plt.show()

    record_video(model, checkpoint, all_returns, relevant_eps)

def record_video(model, checkpoint, all_returns, relevant_eps):
    utils.log("Preparing video recording!")

    if utils.is_remote():
        import pyvirtualdisplay
        # Creates a virtual display for OpenAI gym
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

    import gym
    from stable_baselines3.common.vec_env import VecVideoRecorder

    # load the environment
    env = utils.load_env(checkpoint, cfg.save_path, cfg.env_id)
    mimic_env = env.venv.envs[0].env
    mimic_env.activate_evaluation()

    # build the video path
    pi_string = 'determin' if DETERMINISTIC_ACTIONS else 'stochastic'
    video_path = cfg.save_path + 'videos_' + pi_string

    # setup the video recording
    env = VecVideoRecorder(env, video_path,
                           record_video_trigger=lambda x: x >= 0,
                           video_length=100)

    obs = env.reset()
    for _ in range(101):
        action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
        action =  [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        env.render()
        # only reset when agent has fallen
        # if has_fallen(mimic_env):
        #     video_env.reset()

    # Save the video
    env.close()





def record_video_OLD(model, checkpoint, all_returns, relevant_eps):
    """ GENERATE VIDEOS of different performances (best, worst, mean)

    # The idea is to understand the agent by observing his behavior
    # during the best, worst episode and an episode with a close to average return.
    # Therefore, we reload the environment to have same behavior as in the evaluation episodes
    # and record the video only for the interesting episodes.
    """

    # import the video recorder
    from stable_baselines3.common.vec_env import VecVideoRecorder

    utils.log("Preparing video recording!")

    # which episodes are interesting to record a video of
    relevant_eps_returns = [max(all_returns), min(all_returns), np.mean(all_returns)]
    relevant_eps_names = ['best', 'worst', 'mean']

    # reload environment to replicate behavior of evaluation episodes (determinism tested)
    if FROM_PATH:
        save_path = PATH
    else:
        save_path = cfg.save_path
    env = utils.load_env(checkpoint, save_path, cfg.env_id)
    obs = env.reset()

    ep_count, step = 0, 0

    # determine video duration
    fps = env.venv.metadata['video.frames_per_second']
    video_len_secs = 10
    video_n_steps = video_len_secs * fps

    # build the video path
    pi_string = 'determin' if DETERMINISTIC_ACTIONS else 'stochastic'
    video_path = save_path + 'videos_' + pi_string

    # repeat only as much episodes as necessary
    while ep_count <= max(relevant_eps):

        if ep_count in relevant_eps:
            ep_index = relevant_eps.index(ep_count)
            ep_ret = relevant_eps_returns[ep_index]
            ep_name = relevant_eps_names[ep_index]

            # create an environment that captures performance on video
            video_env = VecVideoRecorder(env, video_path,
                                         record_video_trigger=lambda x: x > 0,
                                         video_length=video_n_steps,
                                         name_prefix=f'{ep_name}_{int(ep_ret)}_')
            # access the wrapped mimic environment
            mimic_env = video_env.env.venv.envs[0].env
            mimic_env.activate_evaluation()

            obs = video_env.reset()

            while step <= video_n_steps:
                action, hid_states = model.predict(obs, deterministic=DETERMINISTIC_ACTIONS)
                obs, reward, done, info = video_env.step(action)
                step += 1
                # only reset when agent has fallen
                if has_fallen(mimic_env):
                    video_env.reset()

            video_env.close()
            utils.log(f"Saved performance video after {step} steps.")
            step = 0

        # irrelevant episode, just reset the environment
        else:
            env.reset()

        # log progress
        if ep_count % 10 == 0:
            print(f'{ep_count} episodes finished', flush=True)

        ep_count += 1
    env.close()

    # rename folder to mark it as evaluated
    path_evaled = save_path[:-1] + f'-evaled-ret{int(np.mean(all_returns))}'
    os.rename(save_path[:-1], path_evaled)

    # upload videos to wandb
    mp4_paths_all = glob.glob(path_evaled+f'/videos_{pi_string}/*.mp4')
    # filter out broken videos, filesize < 1MB
    mp4_paths = [path for path in mp4_paths_all if os.path.getsize(path)>1024**2]
    utils.log('MP4 Paths:', mp4_paths)
    wandb.log({"video": wandb.Video(mp4_paths[0], fps=16, format='gif')})
    # wandb.log({"video": wandb.Video(mp4_paths[1], fps=4, format='mp4')})


def has_fallen(mimic_env):
    com_z_pos = mimic_env.data.qpos[mimic_env._get_COM_indices()[-1]]
    return com_z_pos < 0.25


if __name__ == "__main__":
    pass
# eval.py was called from another script
else:
    RENDER = False
    FROM_PATH = False