# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# determine if Pytorch should use CPU or GPU
USE_CPU = True
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 15 # testing if we really need this
# maximum length of an episode
MAX_EPISODE_STEPS = 3000

# configure Weights & Biases
WB_PROJECT_NAME = 'restruct'
# todo: definitely also try increasing the velocity reward weight
WB_RUN_NAME = f'Straight NO_MIRR'
#                  'Set max walking distance to 150m to check if we really need it. ' \
# 'Max spisode steps reduced to 1000 as we are now training at 100Hz' \
# 'Calculating desired velocity from 0.5 seconds of future mocaps. ' \
#                      'Scaling the reward by 100. ' \
# 'Starting training of the 2seg upper body model ' \
#                      'AND loco3d mocaps without any further adjustments.' \
#                      '' \
WB_RUN_DESCRIPTION = 'Training the straight walker WITH policy mirroring. ' \
                 'Fixed a lot of issues in monitoring. ' \
                 'Evaluate the model by letting it walk straight. ' \
                 'This way, we can retain all our previous evaluation metrics. ' \
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

# specify the environment you want to use
ENV_ID = 'StraightMimicWalker' # 'MimicWalker165cm65kg' #
# simulation frequency... overwrites the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = {'StraightMimicWalker': 200,
             'MimicWalker165cm65kg': 100}[ENV_ID]

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