# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/22.
#


from pysc2.lib.actions import FUNCTIONS, FunctionCall
import numpy as np

from .config import SPATIAL_ARG_TYPES


class Adapter(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultObservationAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, timesteps):
        '''
        :param timesteps:
        :return: state, next_state, reward, done, timesteps
        '''
        slice_len = 1
        if isinstance(timesteps[0], list):
            slice_len = len(timesteps[0])
            for i in range(1, len(timesteps)):
                timesteps[0].extend(timesteps[i])
            timesteps = timesteps[0]

        states, dones, rewards = None, [], []
        screen_feature_indexes = \
            self._config._screen_feature_indexes
        minimap_feature_indexes = \
            self._config._minimap_feature_indexes
        nonspatial_features = \
            self._config._non_spatial_features
        action_indexes = self._config._action_indexes
        screens, minimaps = [], []
        nonspatials = [[] for _ in range(len(nonspatial_features))]
        for timestep in timesteps:
            rewards.append(timestep.reward)
            dones.append(timestep.last())
            observation = timestep.observation
            screen_features = observation.screen_feature[
                screen_feature_indexes]
            minimap_features = observation.minimap_feature[
                minimap_feature_indexes]

            screens.append(screen_features)
            minimaps.append(minimap_features)

            for i, field in enumerate(nonspatial_features):
                feature = observation[field]
                if field == 'available_actions':
                    feature = np.zeros(len(FUNCTIONS))
                    feature[observation[field]] = 1
                    feature = feature[action_indexes]
                nonspatials[i].append(feature)

        screen = np.array(screens).transpose((0, 2, 3, 1))
        minimap = np.array(minimaps).transpose((0, 2, 3, 1))
        nonspatial = [np.array(e) for e in nonspatials]

        states = [screen, minimap] + nonspatial
        return ([s[:-slice_len] for s in states],
                [s[slice_len:] for s in states],
                np.array(rewards[slice_len:]),
                np.array(dones[slice_len:]),
                timesteps[:-slice_len])

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultActionAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, func_calls):
        values = []
        size = self._config._size
        action_index_table = \
            self._config._action_index_table
        action_args_index_table = \
            self._config._action_args_index_table
        length = len(func_calls)
        for dim, _ in self._config.policy_dims:
            values.append(np.zeros(shape=(length, dim), dtype=np.int32))
        act_values, arg_values = [values[0]], values[1:]
        for i, func_call in enumerate(func_calls):
            act_id = action_index_table[int(func_call.function)]
            act_values[0][i][act_id] = 1
            for i, arg_type in enumerate(FUNCTIONS[act_id].args):
                arg = func_call.arguments[i]
                if arg_type.name in SPATIAL_ARG_TYPES:
                    arg = arg[0] + arg[1] * size
                else:
                    arg = int(arg[0])
                arg_idx = action_args_index_table[arg_type.name]
                arg_values[arg_idx][i][arg] = 1

        return values

    def reverse(self, actions):
        action_indexes = self._config._action_indexes
        action_arg_table = self._config._action_args_index_table
        size = self._config._size
        function_calls = []
        for action in zip(*actions):
            act_id = action_indexes[action[0]]
            act_args = []
            for arg_type in FUNCTIONS[act_id].args:
                arg_index = action_arg_table[arg_type.name]
                arg_value = action[1:][arg_index]
                if arg_type.name in SPATIAL_ARG_TYPES:
                    act_args.append(
                        [arg_value % size,
                         arg_value // size])
                else:
                    act_args.append([arg_value])
            function_call = FunctionCall(act_id, act_args)
            function_calls.append(function_call)
        return function_calls
