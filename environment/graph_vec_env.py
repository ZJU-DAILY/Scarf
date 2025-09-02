import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


class GraphVecEnv(VecEnv):
    """
    Vectorized environment for variable-shape graph observations.
    Observations from all envs are returned as: dict[str, list[np.ndarray]], i.e.,
    for each field (e.g. 'node_features'), a python list whose i-th entry is the obs from env i.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        env_fns: list of env constructors, one per environment
        """
        self.envs = [_patch_env(fn()) for fn in env_fns]
        ids = [id(env.unwrapped) for env in self.envs]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        self.metadata = env.metadata

        # Figure out observation keys for dict obs space
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            self.obs_keys = list(obs_space.spaces.keys())
        else:
            # For non-dict obs, treat as key None
            self.obs_keys = [None]

        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float64)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_infos: List[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self):
        """
        Step each env one by one.
        Returns:
            obs (Dict[str, List[np.ndarray]]): For each key, a list of that field's observation for each env.
            rews (np.ndarray): [n_envs]
            dones (np.ndarray): [n_envs]
            infos (List[dict]): [n_envs]
        """
        obs_list: List[Any] = []
        for env_idx in range(self.num_envs):
            (
                obs,
                self.buf_rews[env_idx],
                terminated,
                truncated,
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            self.buf_dones[env_idx] = terminated or truncated
            # TimeLimit.truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = (
                truncated and not terminated
            )
            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            obs_list.append(obs)
        # Transpose obs_list into dict[str, list[np.ndarray]]
        obs_dict = _list_of_obs_to_dict_of_list(obs_list)
        return (
            obs_dict,
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(self):
        """
        Reset all envs.
        Returns:
            obs (Dict[str, List[np.ndarray]]): For each key, a list of that field's observation for each env.
        """
        obs_list: List[Any] = []
        for env_idx in range(self.num_envs):
            maybe_options = (
                {"options": self._options[env_idx]} if self._options[env_idx] else {}
            )
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options
            )
            obs_list.append(obs)
        self._reset_seeds()
        self._reset_options()
        obs_dict = _list_of_obs_to_dict_of_list(obs_list)
        return obs_dict

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        # Optional: implement a tiled rendering, if needed
        return super().render(mode=mode)

    def get_attr(
        self, attr_name: str, indices: Optional[Union[None, int, List[int]]] = None
    ) -> List[Any]:
        indices = self._get_indices(indices)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(
        self,
        attr_name: str,
        value: Any,
        indices: Optional[Union[None, int, List[int]]] = None,
    ) -> None:
        indices = self._get_indices(indices)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(
        self, method_name: str, *method_args, indices=None, **method_kwargs
    ) -> List[Any]:
        indices = self._get_indices(indices)
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in indices
        ]

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> List[bool]:
        indices = self._get_indices(indices)
        return [isinstance(self.envs[i], wrapper_class) for i in indices]


# --- Helper for graph observations: transpose list of obs dicts into dict of lists
def _list_of_obs_to_dict_of_list(obs_list: List[Any]) -> Dict[str, List[np.ndarray]]:
    """
    Args:
        obs_list: length-N list, each is obs for one env, could be dict[str, np.ndarray] or ndarray.
        obs_keys: list of observation keys, or [None] for non-dict obs.

    Returns:
        dict of lists: key->list of obs for each env.
    """
    result = {}
    for obs in obs_list:
        if not isinstance(obs, dict):
            raise ValueError("Expected obs to be a dict, but got: %s" % type(obs))
        for k in obs:
            if k not in result:
                result[k] = []
            result[k].append(obs[k])

    length = None
    for item in result.values():
        if length is None:
            length = len(item)
        else:
            if length != len(item):
                raise ValueError(
                    "All lists in the dict must have the same length, but got: %s"
                    % [len(item) for item in result.values()]
                )
    return result
