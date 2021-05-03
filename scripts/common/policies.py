from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3.common.utils import linear, ortho_init
from scripts.behavior_cloning.models import load_weights, load_encoder_weights
from scripts.common.utils import log
from scripts.common import config as cfg
import tensorflow as tf
import numpy as np


class CustomPolicy(ActorCriticPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, **kwargs)

        # log("Using CustomPolicy.")

        with tf.variable_scope("model", reuse=reuse):
            obs = self.processed_obs # shape: (?, obs_dim)
            act_func_hid = tf.nn.relu

            # reduce dim of observations
            if cfg.is_mod(cfg.MOD_E2E_ENC_OBS):
                log('Building an encoder to reduce PI input dimensionality.\n'
                    f'Input dim original: {obs.shape[1]}\n'
                    f'Hidden Layer Sizes (E2E): {cfg.enc_layer_sizes + cfg.hid_layer_sizes_pi}')
                obs_reduced = self.fc_hidden_layers('obs_enc_hid', obs, cfg.enc_layer_sizes, act_func_hid)


            # build the policy network's hidden layers
            if cfg.is_mod(cfg.MOD_PRETRAIN_PI):
                pi_h = self.load_pretrained_policy_hid_layers('pi_fc_hid', obs, act_func_hid)
                log('Loading pretrained policy HIDDEN LAYER weights!')
            elif cfg.is_mod(cfg.MOD_GROUND_CONTACT_NNS):
                log('Constructing multiple networks for different gait phases!')
                pi_left = self.fc_hidden_layers('pi_left_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                pi_right = self.fc_hidden_layers('pi_right_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                pi_double = self.fc_hidden_layers('pi_double_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                has_ground_contact_left = tf.stack([obs[:,0]] * cfg.hid_layer_sizes_vf[-1], axis=1)
                has_ground_contact_right = tf.stack([obs[:,1]] * cfg.hid_layer_sizes_vf[-1], axis=1)
                has_ground_contact_both = tf.stack([obs[:,2]] * cfg.hid_layer_sizes_vf[-1], axis=1)
                pi_h = tf.divide(
                    (tf.multiply(has_ground_contact_left, pi_left)
                     + tf.multiply(has_ground_contact_right, pi_right)
                     + tf.multiply(has_ground_contact_both, pi_double)),
                    (has_ground_contact_left+has_ground_contact_right+has_ground_contact_both))
            else:
                # simple two hidden layer fully connected policy network
                pi_obs_input = obs if not cfg.is_mod(cfg.MOD_E2E_ENC_OBS) else obs_reduced
                pi_h = self.fc_hidden_layers('pi_fc_hid', pi_obs_input, cfg.hid_layer_sizes_pi, act_func_hid)
            # build the value network's hidden layers
            if cfg.is_mod(cfg.MOD_GROUND_CONTACT_NNS):
                vf_left = self.fc_hidden_layers('vf_left_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                vf_right = self.fc_hidden_layers('vf_right_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                vf_double = self.fc_hidden_layers('vf_double_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
                vf_h = tf.divide(
                (tf.multiply(has_ground_contact_left, vf_left)
                 + tf.multiply(has_ground_contact_right, vf_right)
                 + tf.multiply(has_ground_contact_both, vf_double)),
                (has_ground_contact_left + has_ground_contact_right + has_ground_contact_both))
            else:
                vf_h = self.fc_hidden_layers('vf_fc_hid', obs, cfg.hid_layer_sizes_vf, act_func_hid)
            # build the output layer of the policy (init_scale as proposed by stable-baselines)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_h, vf_h, init_scale=0.01)
            # build the output layer of the value function
            vf_out = self.fc('vf_out', vf_h, 1, zero=cfg.is_mod(cfg.MOD_VF_ZERO))
            self._value_fn = vf_out
            # required to set up additional attributes
            self._setup_init()

    def fc_hidden_layers(self, name, input, hid_sizes, act_func):
        """Fully connected MLP. Number of layers determined by len(hid_sizes)."""
        hid = input
        for i, size in enumerate(hid_sizes):
            hid = act_func(self.fc(f'{name}{i}', hid, size))
        return hid

    def build_linear_layer(self, input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
        """
        Creates a fully connected layer for TensorFlow

        :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer
        :param scope: (str) The TensorFlow variable scope
        :param n_hidden: (int) The number of hidden neurons
        :param init_scale: (int) The initialization scale
        :param init_bias: (int) The initialization offset bias
        :return: (TensorFlow Tensor) fully connected layer
        """
        with tf.variable_scope(scope):
            n_input = input_tensor.get_shape()[1].value
            weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale),
                                     regularizer= (tf.keras.regularizers.l2(cfg.l2_coef)
                                     if cfg.is_mod(cfg.MOD_L2_REG) else None))
            bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
            return tf.matmul(input_tensor, weight) + bias

    def fc(self, name, input, size, zero=False):
        """
        Builds a single fully connected layer. Initial values taken from stable-baselines.
        :param zero: if True, scale down init weights and biases by a small value of 0.01
        """
        return self.build_linear_layer(input, name, size, init_scale=0.01 if zero else np.sqrt(2), init_bias=0)

    def load_pretrained_policy_hid_layers(self, scope, input, act_func=tf.nn.relu):
        """
        Loads the hidden layers of the pretrained policy model (Behavior Cloning).
        Adopted linear layer from stable_baselines/a2c/utils.py.
        :param input: (tf tensor) The input tensor for the fully connected layer
        :param scope: (str) The TensorFlow variable scope
        """

        input_dim = input.get_shape()[1].value
        # load weights
        # [w_hid1, w_hid2, w_out], [b_hid1, b_hid2, b_out]
        ws, bs = load_weights()
        # check dimensions
        assert input_dim == ws[0].shape[0]
        hid_layer_sizes = [bs[0].size, bs[1].size]
        assert cfg.hid_layer_sizes_pi == hid_layer_sizes
        # organize weights and biases in lists

        # construct the hidden layers with relu activation
        for i, n_hidden in enumerate(hid_layer_sizes):
            with tf.variable_scope(f'{scope}{i}'):
                weight = tf.get_variable("w", initializer=ws[i])
                bias = tf.get_variable("b",initializer=bs[i])
                output = act_func(tf.matmul(input, weight) + bias)
                input = output

        return output

    # ----------------------------------------------
    # Obligatory method definitions from stable-baselines (proba_step, step, value)
    # cf https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
    # ----------------------------------------------

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

# Register the policy
register_policy('CustomPolicy', CustomPolicy)