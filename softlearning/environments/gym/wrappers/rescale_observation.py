import numpy as np

import gym
from gym import spaces


def rescale_values(values, old_low, old_high, new_low, new_high):
    rescaled_values = new_low + (new_high - new_low) * (
        (values - old_low) / (old_high - old_low))
    rescaled_values = np.clip(rescaled_values, new_low, new_high)
    return rescaled_values


class RescaleObservation(gym.ObservationWrapper):
    def __init__(self, env, low, high):
        r"""Rescale observation space to a range [`low`, `high`].
        Example:
            >>> RescaleObservation(env, low, high).observation_space == Box(low, high)
            True
        Raises:
            TypeError: If `not isinstance(environment.observation_space, spaces.Box)`.
            ValueError: If either `low` or `high` is not finite.
            ValueError: If any of `observation_space.{low,high}` is not finite.
            ValueError: If `high <= low`.
        TODO(hartikainen): This should be extended to work with Dict and Tuple spaces.
        """
        if np.any(~np.isfinite((low, high))):
            raise ValueError(
                f"Arguments 'low' and 'high' need to be finite. Got: low={low}, high={high}"
            )

        if np.any(high <= low):
            raise ValueError(
                f"Argument `low` must be smaller than `high` Got: low={low}, high="
            )

        super(RescaleObservation, self).__init__(env)

        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                f"Expected Box observation space. Got: {type(env.observation_space)}"
            )

        if np.any(~np.isfinite((
                env.observation_space.low, env.observation_space.high))):
            raise ValueError(
                f"Observation space 'low' and 'high' need to be finite. Got: observation_space.low={env.observation_space.low}, observation_space.high={env.observation_space.high}"
            )

        shape = env.observation_space.shape
        dtype = env.observation_space.dtype

        self.low = low + np.zeros(shape, dtype=dtype)
        self.high = high + np.zeros(shape, dtype=dtype)
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, shape=shape, dtype=dtype)

    def observation(self, observation):
        return rescale_values(
            observation,
            old_low=self.env.observation_space.low,
            old_high=self.env.observation_space.high,
            new_low=self.low,
            new_high=self.high,
        )
