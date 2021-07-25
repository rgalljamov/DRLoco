# -----------------------------
# Experiment
# -----------------------------

# don't sync with W&B in debug mode, log additional information etc.
DEBUG = False
# when debugging, stop training early after a few steps
MAX_DEBUG_STEPS = int(2e4)
# determine if PyTorch should use CPU or GPU
USE_CPU = True

# -----------------------------
# Simulation Environment
# -----------------------------

# specify the environment you want to use
# the IDs of all available environments are listed in mujoco/config.py
ENV_ID = 'StraightMimicWalker' # 'MimicWalker165cm65kg' #
# specify control frequency in Hz (policy queries per second)
CTRL_FREQ = {'StraightMimicWalker': 200,
             'MimicWalker165cm65kg': 100}[ENV_ID]
# number of episodes per model evaluation
EVAL_N_TIMES = 20
# minimum distance [m] to walk to label the gait as stable
MIN_STABLE_DISTANCE = 15

# --------------------------------
# Weights & Biases
# --------------------------------

# do you want to use Weights & Biases (WB)? I love it!
# It syncs the tensorboard logs to WB, logs all hyperparameters,
# and allows you to effectively manage, monitor and compare your agents!
USE_WANDB = True

if USE_WANDB:
    # give your project a name
    WB_PROJECT_NAME = 'drloco_models'
    # give the current run a name
    WB_RUN_NAME = f'CPU, PHASE, MIRR_PY'
    # describe your ideas/goals/changes related to the current run
    WB_RUN_DESCRIPTION = 'Training the straight walker WITHOUT policy mirroring but with the real PHASE. ' \
                         '1D desired velocity from refs, ' \
                         'DO USE COM Y position in obs! ' \
                         'Fixed a lot of issues in monitoring. ' \
                     'Evaluate the model by letting it walk straight. ' \
                     'This way, we can retain all our previous evaluation metrics. ' \
                     ''