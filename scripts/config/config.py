# -----------------------------
# Experiment Specification
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG_TRAINING = False
# determine if Pytorch should use CPU or GPU
USE_CPU = True
# maximum walking distance after which the episode is terminated
MAX_WALKING_DISTANCE = 15
# maximum length of an episode
MAX_EPISODE_STEPS = 3000

# TODO: remove COM reward, train longer with smaller LR decay, use exp clip_range sched
# configure Weights & Biases
WB_PROJECT_NAME = 'no_phase'
WB_RUN_NAME = 'phase_ang'
WB_RUN_DESCRIPTION = 'Estimate the phase variable from the hip joint phase plot angle. ' \
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
ENV_ID = 'MimicWalker3d-v0'
# walker XML file
WALKER_MJC_XML_FILE = 'walker3d_flat_feet.xml' # 'walker3d_flat_feet_lowmass.xml' # 'walker3d_flat_feet_40kg_140cm.xml' #
# simulation frequency... overwrite the frequency specified in the xml file
SIM_FREQ = 1000
# control frequency in Hz
CTRL_FREQ = 200
# does the model uses joint torques (True) or target angles (False)?
ENV_OUT_TORQUE = True
# peak joint torques [hip_sag, hip_front, knee_sag, ank_sag], same for both sides
PEAK_JOINT_TORQUES = [300]*4 # [50]*3 + [5] # [300, 300, 300, 300] #


# -----------------------------
# Algorithm Hyperparameters
# -----------------------------

# number of training steps = samples to collect [in Millions]
MIO_SAMPLES = 4
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