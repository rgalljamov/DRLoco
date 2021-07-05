# add current working directory to the system path
import sys
from os import getcwd
sys.path.append(getcwd())

# import required modules
import os.path
import wandb

import torch as th

from drloco.config import config as cfgl
from drloco.config import hypers as cfg
from drloco.common import utils
from drloco.common.schedules import LinearDecay, ExponentialSchedule
from drloco.common.callback import TrainingMonitor

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from drloco.custom.policies import CustomActorCriticPolicy

# determine the name of saved models before (init) and after training (final)
INIT_CHECKPOINT_SUFFIX = 'init'
FINAL_CHECKPOINT_SUFFIX = 'final'

def use_cpu():
    """
    Force PyTorch to use CPU instead of GPU.
    In some cases, e.g. training many agents in parallel on a CPU cluster,
    it might be useful to use CPU instead of GPU. This function fools PyTorch
    to think there is no GPU available on the PC, so that it uses the CPU.
    """
    from os import environ
    # fool python to think there is no CUDA device
    environ["CUDA_VISIBLE_DEVICES"] = ""
    # to avoid massive slow-down when using torch with cpu
    import torch
    n_envs = cfg.n_envs
    torch.set_num_threads(n_envs if n_envs <= 16 else 8)


def init_wandb(model):
    batch_size = model.n_steps * model.n_envs
    params = {
        "path": cfg.save_path,
        "env_id": cfgl.ENV_ID,
        "mod": cfg.modification,
        "lr0": cfg.lr_start,
        "lr1": cfg.lr_final,
        'hid_sizes': cfg.hid_layer_sizes,
        "noptepochs": cfg.noptepochs,
        "batch_size": batch_size,
        "cfg.batch_size": cfg.batch_size,
        "cfg.minibatch_size": cfg.minibatch_size,
        "mio_steps": cfg.mio_samples,
        "ent_coef": model.ent_coef,
        "ep_dur": cfg.ep_dur_max,
        "imit_rew": cfg.rew_weights,
        "logstd": cfg.init_logstd,
        "gam": model.gamma,
        "lam": model.gae_lambda,
        "n_envs": model.n_envs,
        "seed": model.seed,
        "policy": model.policy,
        "n_steps": model.n_steps,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        }

    if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED):
        params.update({"clip0": cfg.clip_start, "clip1": cfg.clip_end})

    wandb.init(config=params, sync_tensorboard=True, name=cfgl.WB_RUN_NAME,
               project=cfgl.WB_PROJECT_NAME, notes=cfgl.WB_RUN_DESCRIPTION)


def train():

    # make torch using the CPU instead of the GPU
    if cfgl.USE_CPU: use_cpu()

    # create model directories
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
        os.makedirs(cfg.save_path + 'metrics')
        os.makedirs(cfg.save_path + 'models')
        os.makedirs(cfg.save_path + 'models/params')
        os.makedirs(cfg.save_path + 'envs')

    # setup environment
    env = utils.vec_env(cfgl.ENV_ID, norm_rew=True, num_envs=cfg.n_envs)

    # setup model/algorithm
    training_timesteps = int(cfg.mio_samples * 1e6)
    lr_start = cfg.lr_start
    lr_end = cfg.lr_final

    learning_rate_schedule = LinearDecay(lr_start, lr_end).value
    if cfg.is_mod(cfg.MOD_CLIPRANGE_SCHED):
        clip_schedule = ExponentialSchedule(cfg.clip_start, cfg.clip_end, cfg.clip_exp_slope)
        clip_range = clip_schedule.value
    else:
        clip_range = cfg.cliprange

    use_custom_policy = cfg.is_mod(cfg.MOD_CUSTOM_POLICY)
    policy_kwargs = {'log_std_init':cfg.init_logstd} if use_custom_policy else \
                    {'net_arch': [{'vf': cfg.hid_layer_sizes, 'pi': cfg.hid_layer_sizes}],
                    'activation_fn': th.nn.Tanh, 'log_std_init':cfg.init_logstd}

    model = PPO(CustomActorCriticPolicy if use_custom_policy else MlpPolicy,
                       env, verbose=1,
                       n_steps = cfg.batch_size//cfg.n_envs, # num of steps per env per update
                       batch_size=cfg.minibatch_size, # minibatch size (batch size per training step)
                       policy_kwargs=policy_kwargs,
                       learning_rate=learning_rate_schedule, ent_coef=cfg.ent_coef,
                       gamma=cfg.gamma, n_epochs=cfg.noptepochs,
                       clip_range_vf=clip_range, clip_range=clip_range,
                       tensorboard_log=cfg.save_path + 'tb_logs/')

    # init wandb when not debugging
    if not cfgl.DEBUG: init_wandb(model)

    # print model path and modification parameters
    utils.log('RUN DESCRIPTION: \n' + cfgl.WB_RUN_DESCRIPTION)
    utils.log('Training started',
              ['Model: ' + cfg.save_path, 'Modifications: ' + cfg.modification])

    # save model and weights before training
    if not cfgl.DEBUG:
        utils.save_model(model, cfg.save_path, INIT_CHECKPOINT_SUFFIX)

    # train model
    model.learn(total_timesteps=training_timesteps, callback=TrainingMonitor())

    # save model after training
    utils.save_model(model, cfg.save_path, FINAL_CHECKPOINT_SUFFIX)

    # close environment
    env.close()

    # evaluate the trained model
    # eval.eval_model()


if __name__ == '__main__':
    train()
