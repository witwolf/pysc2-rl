# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/22.
#


import itertools
from pysc2.lib.actions import FUNCTIONS, FunctionCall
import numpy as np
from lib.protoss_macro import PROTOSS_MACROS
from .config import SPATIAL_ARG_TYPES
from lib.protoss_unit import _PROTOSS_UNITS, _PROTOSS_UNITS_MACROS, _PROTOSS_BUILDINGS, _PROTOSS_BUILDINGS_MACROS
from pysc2.lib import units

class Adapter(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()

class DefaultActionRewardAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def get_feature_vector(self, timestep):
        features = {}
        for unit in timestep.observation.raw_units:
            if unit.alliance == 1:
                if unit.unit_type not in features:
                    features[unit.unit_type] = 1
                else:
                    features[unit.unit_type] += 1
        return features, timestep.observation.player.minerals, timestep.observation.player.vespene,\
               timestep.observation.player.food_cap - timestep.observation.player.food_used

    def get_reward(self, timestep, action):
        features, minerals, gas, food = self.get_feature_vector(timestep)
        # check unit requirements
        if action.id in _PROTOSS_UNITS_MACROS:
            unit = _PROTOSS_UNITS_MACROS[action.id]
            # if resource not enough
            if minerals < unit.minerals or gas < unit.gas or food < unit.food:
                return -1
            for requirement in unit.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            return float(unit.minerals + unit.gas + unit.gas) / 1000.0

        # check building requirements
        if action.id in _PROTOSS_BUILDINGS_MACROS:
            building = _PROTOSS_BUILDINGS_MACROS[action.id]
            # if resource not enough
            if minerals < building.minerals or gas < building.gas:
                return -1
            for requirement in building.requirement_types:
                # if requirements miss
                if requirement not in features:
                    return -1
            # first building return 1
            if building.unit_type not in features:
                return 1
            # if need food
            elif building.unit_type == units.Protoss.Pylon and food < 2:
                return 1
            else:
                return np.exp(-features[building.unit_type])

        # other function return 0
        return 0

    def transform(self, timesteps, actions):
        '''
        :param timesteps:
        :param actions:
        :return:
        '''
        rewards = []
        for timestep, action in zip(timesteps, actions):
            rewards.append(self.get_reward(timestep, action))
        return rewards

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
        slice_len = 1
        _timesteps = timesteps
        if isinstance(timesteps[0], list):
            slice_len = len(timesteps[0])
            _timesteps = itertools.chain(*timesteps)
            _timesteps = list(_timesteps)

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
        for timestep in _timesteps:
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
                    feature = np.zeros(len(FUNCTIONS))
                    feature[observation[field]] = 1
                    feature = feature[action_indexes]
                nonspatials[i].append(feature)

        screen = np.array(screens).transpose((0, 2, 3, 1))
        minimap = np.array(minimaps).transpose((0, 2, 3, 1))
        nonspatial = [np.array(e) for e in nonspatials]

        states = [screen, minimap] + nonspatial
        if state_only:
            return states
        return ([s[:-slice_len] for s in states],
                [s[slice_len:] for s in states],
                np.array(rewards[slice_len:]),
                np.array(dones[slice_len:]),
                _timesteps[:-slice_len])

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


class DefaultProtossMacroAdapter(Adapter):
    def __init__(self, config):
        self._config = config

    def transform(self, macros):
        values = []

        action_index_table = \
            self._config._action_index_table

        length = len(macros)
        for dim, _ in self._config.policy_dims:
            values.append(np.zeros(shape=(length), dtype=np.int32))
        for i, macro in enumerate(macros):
            macro_id = action_index_table[macro.macro]
            values[0][i] = macro_id

    def reverse(self, actions):
        action_indexes = self._config._action_indexes
        macros = []
        for action in actions[0]:
            macro_id = action_indexes[action]
            macros.append(PROTOSS_MACROS[macro_id]())
        return macros
