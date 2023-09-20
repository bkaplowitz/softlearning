import tensorflow as tf
import tensorflow_probability as tfp
import tree

from .base_policy import ContinuousPolicy


class UniformPolicyMixin:
    @tf.function(experimental_relax_shapes=True)
    def actions(self, observations):
        first_observation = tree.flatten(observations)[0]
        first_input_rank = tf.size(tree.flatten(self._input_shapes)[0])
        batch_shape = tf.shape(first_observation)[:-first_input_rank]

        return self.distribution.sample(batch_shape)

    @tf.function(experimental_relax_shapes=True)
    def log_probs(self, observations, actions):
        return self.distribution.log_prob(actions)[..., tf.newaxis]

    @tf.function(experimental_relax_shapes=True)
    def probs(self, observations, actions):
        return self.distribution.prob(actions)[..., tf.newaxis]


class ContinuousUniformPolicy(UniformPolicyMixin, ContinuousPolicy):
    def __init__(self, *args, **kwargs):
        super(ContinuousUniformPolicy, self).__init__(*args, **kwargs)
        low, high = self._action_range
        self.distribution = tfp.distributions.Independent(
            tfp.distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1)
