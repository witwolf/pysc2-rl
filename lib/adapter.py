# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/22.
#


import numpy as np
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from .config import SPATIAL_ARG_TYPES
from lib.protoss_unit import *
from lib.protoss_unit import _PROTOSS_UNITS_MACROS
from lib.protoss_unit import _PROTOSS_BUILDINGS_MACROS
from pysc2.lib import units


class Adapter(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultActionRewardAdapter(Adapter):
    def __init__(self, config, memory_step_n = 8):
        self._config = config
        self._memory_step_n = memory_step_n
        self._memory_step_obs = [None] * memory_step_n

    def get_feature_vector(self, timestep):
        features = {}
        for unit in timestep.observation.raw_units:
            if unit.alliance == 1:
                if unit.unit_type not in features:
                    features[unit.unit_type] = 1
                else:
                    features[unit.unit_type] += 1
        return (features,
                timestep.observation.player.minerals,
                timestep.observation.player.vespene,
                timestep.observation.player.food_cap -
                timestep.observation.player.food_used)

    def get_reward(self, timestep, success, action):
        if not success:
            return 0
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

        if (action.id == PROTOSS_MACROS.Collect_Mineral) or (
                action.id == PROTOSS_MACROS.Callback_Idle_Workers):
            full_harvesters = [
                unit.assigned_harvesters >= unit.ideal_harvesters
                for unit in timestep.observation.feature_units
                if unit.unit_type == units.Protoss.Nexus]
            # not enough mineral harvester, positive reward
            for full in full_harvesters:
                if not full:
                    return 1
            # full mineral harvester, negative reward
            return -1

        if action.id == PROTOSS_MACROS.Collect_Gas:
            full_harvesters = [unit.assigned_harvesters >= unit.ideal_harvesters
                               for unit in timestep.observation.feature_units
                               if unit.unit_type == units.Protoss.Assimilator]
            # not enough mineral harvester, positive reward
            for full in full_harvesters:
                if not full:
                    return 1
            # full mineral harvester, negative reward
            return -1

        # other function return 0
        return 0

    def get_damage(self, timestep, last_timestep):
        if not last_timestep:
            return 0
        # self health
        self_hp = np.sum([
            unit.health + unit.shield for unit
            in timestep.observation.raw_units
            if unit.alliance == 1])
        self_last_hp = np.sum([
            unit.health + unit.shield
            for unit in last_timestep.observation.raw_units
            if unit.alliance == 1])
        # enemy health
        enemy_hp = np.sum([
            unit.health + unit.shield
            for unit in timestep.observation.raw_units
            if unit.alliance == 4])
        enemy_last_hp = np.sum([
            unit.health + unit.shield
            for unit in last_timestep.observation.raw_units
            if unit.alliance == 4])
        # health delta
        self_hp_delta = self_hp - self_last_hp
        enemy_hp_delta = enemy_hp - enemy_last_hp
        return np.clip(self_hp_delta - enemy_hp_delta, -5, 5)

    def transform(self, timesteps, successes, actions):
        '''
        :param timesteps:
        :param actions:
        :return:
        '''
        step_i = 0
        rewards = []
        game_end = False
        for timestep, success, action in zip(timesteps, successes, actions):
            if timestep.last():
                game_end = True
            last_timestep = None
            if success:
                if step_i < self._memory_step_n:
                    last_timestep = self._memory_step_obs[step_i]
                else:
                    last_timestep = timesteps[step_i - self._memory_step_n]
            rewards.append(self.get_reward(timestep, success, action) +
                           self.get_damage(timestep, last_timestep) +
                            timestep.reward)
            step_i += 1

        if not game_end:
            self._memory_step_obs = timesteps[-self._memory_step_n:]
        else:
            self._memory_step_obs = [None] * self._memory_step_n
        return np.array(rewards)

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
        action_indexes = self._config._action_indexes
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
                    feature = np.zeros(len(FUNCTIONS))
                    feature[observation[field]] = 1
                    feature = feature[action_indexes]
                nonspatials[i].append(feature)

        screen = np.array(screens).transpose((0, 2, 3, 1))
        minimap = np.array(minimaps).transpose((0, 2, 3, 1))
        nonspatial = [np.array(e) for e in nonspatials]
        states = [screen, minimap] + nonspatial
        return (states, np.array(rewards), np.array(dones), timesteps)

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
            macro_id = action_index_table[macro.id]
            values[0][i] = macro_id

    def reverse(self, actions):
        action_indexes = self._config._action_indexes
        macros = []
        for action in actions[0]:
            macro_id = action_indexes[action]
            macros.append(PROTOSS_MACROS[macro_id]())
        return macros
