# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/22.
#


import numpy as np
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from .config import SPATIAL_ARG_TYPES


class Adapter(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultObservationAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, timesteps, state_only=False):
        '''
        :param timesteps:
        :return: state, next_state, reward, done, timesteps
        '''
        states, dones, rewards = None, [], []
        screen_feature_indexes = \
            self._config._screen_feature_indexes
        minimap_feature_indexes = \
            self._config._minimap_feature_indexes
        nonspatial_features = \
            self._config._non_spatial_features

        screens, minimaps = [], []
        nonspatials = [[] for _ in range(len(nonspatial_features))]
        for timestep in timesteps:
            rewards.append(timestep.reward)
            dones.append(timestep.last())
            observation = timestep.observation
            screen_features = observation.feature_screen[
                screen_feature_indexes]
            minimap_features = observation.feature_minimap[
                minimap_feature_indexes]

            screens.append(screen_features)
            minimaps.append(minimap_features)

            for i, field in enumerate(nonspatial_features):
                feature = observation[field]
                if field == 'available_actions':
                    feature = self._available_actions(timestep)
                nonspatials[i].append(feature)

        screen = np.array(screens).transpose((0, 2, 3, 1))
        minimap = np.array(minimaps).transpose((0, 2, 3, 1))
        nonspatial = [np.array(e) for e in nonspatials]
        states = [screen, minimap] + nonspatial
        return (states, np.array(rewards), np.array(dones), timesteps)

    def _available_actions(self, timestep):
        action_indexes = self._config._action_indexes
        observation = timestep.observation
        feature = np.zeros(len(FUNCTIONS))
        feature[observation['available_actions']] = 1
        return feature[action_indexes]

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
            values.append(np.zeros(shape=(length), dtype=np.int32))
        act_values, arg_values = [values[0]], values[1:]
        for i, func_call in enumerate(func_calls):
            act_id = action_index_table[int(func_call.function)]
            act_values[0][i] = act_id
            for j, arg_type in enumerate(FUNCTIONS[act_id].args):
                arg = func_call.arguments[j]
                if arg_type.name in SPATIAL_ARG_TYPES:
                    arg = arg[0] + arg[1] * size
                else:
                    arg = int(arg[0])
                arg_idx = action_args_index_table[arg_type.name]
                arg_values[arg_idx][i] = arg

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
