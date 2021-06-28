# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# determine if Pytorch should use CPU or GPU
USE_CPU = True
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 150 # testing if we really need this
# maximum length of an episode
MAX_EPISODE_STEPS = 1000

# configure Weights & Biases
WB_PROJECT_NAME = 'debug_loco3d'
# todo: definitely also try increasing the velocity reward weight
WB_RUN_NAME = f'straight + 512 + vel05 + rew_scale10'
WB_RUN_DESCRIPTION = 'Training the straight walker without policy mirroring. ' \
                     'Set max walking distance to 150m to check if we really need it. ' \
                     'Max spisode steps reduced to 1000 as we are now training at 100Hz' \
                     'Calculating desired velocity from 0.5 seconds of future mocaps. ' \
                     'Scaling the reward by 100. ' \
                     'Fixed a lot of issues in monitoring. ' \
                     'Evaluate the model by letting it walk straight. ' \
                     'This way, we can retain all our previous evaluation metrics. ' \
                     'Starting training of the 2seg upper body model ' \
                     'AND loco3d mocaps without any further adjustments.' \
                     '' \
                     'Estimate the phase variable from the hip joint phase plot angle. ' \
                     'Use the so far best batch and minibatch sizes. ' \
                     'Implemented a custom policy so far only replicating the same properties as the MLP policy. ' \
                     'Baseline with the current PPO parameters. ' \
                            '' \
                            'Use default entropy coefficient. ' \
                            'Use tanh instead of relu for hidden layer activations. ' \
                            'Training longer but reducing the LR faster!' \
                            'Mirroring the policy and training only for 4M steps! ' \
                            'Halved the batchsize to 16k.'

# -----------------------------
# Simulation Environment
# -----------------------------

# the registered gym environment id, e.g. 'Walker2d-v2'
ENV_ID = 'MimicWalker165cm65kg-v0' # 'MimicWalker3d-v0' #
# simulation frequency... overwrites the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = {'MimicWalker3d-v0': 200,
             'MimicWalker165cm65kg-v0': 100}[ENV_ID]
# does the model uses joint torques (True) or target angles (False)?
# todo: remove this flag as all our models output torque
ENV_OUT_TORQUE = True

# -----------------------------
# Algorithm Hyperparameters
# -----------------------------

# number of training steps = samples to collect [in Millions]
MIO_SAMPLES = 8
# how many parallel environments should be used to collect samples
N_PARALLEL_ENVS = 8
# network hidden layer sizes
hid_layer_sizes = [512]*2
# LR decay slope scaling: slope = lr_scale * (lr_final - lr_start)
# the decay is linear from lr_start to lr_final
lr_scale = 1

# names of saved model before and after training
init_checkpoint = 'init'
final_checkpoint = 'final'