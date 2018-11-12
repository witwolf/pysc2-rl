# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/22.
#


import numpy as np
from pysc2.lib.actions import FUNCTIONS, FunctionCall
from .config import SPATIAL_ARG_TYPES
from lib.protoss_unit import *
from lib.protoss_unit import _PROTOSS_UNITS_MACROS, _PROTOSS_UNITS_FUNCTIONS
from lib.protoss_unit import _PROTOSS_BUILDINGS_MACROS, _PROTOSS_BUILDINGS_FUNCTIONS
from pysc2.lib import units


class Adapter(object):
    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def reverse(self, *args, **kwargs):
        raise NotImplementedError()


class DefaultActionRewardAdapter(Adapter):
    def __init__(self, config, memory_step_n=8):
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

    def get_reward(self, timestep, action):
        if not timestep.macro_success:
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

        if action.id == PROTOSS_MACROS.Callback_Idle_Workers:
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

    def transform(self, timesteps, actions):
        '''
        :param timesteps:
        :param actions:
        :return:
        '''
        step_i = 0
        rewards = []
        game_end = False
        for timestep, action in zip(timesteps, actions):
            if timestep.last():
                game_end = True
            last_timestep = None
            if timestep.macro_success:
                if step_i < self._memory_step_n:
                    last_timestep = self._memory_step_obs[step_i]
                else:
                    last_timestep = timesteps[step_i - self._memory_step_n]
            rewards.append(self.get_reward(timestep, action) +
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


class DefaultInformationAdapter(Adapter):
    class DefaultInformation:
        def __init__(self):
            self._frame_pass = 0         # second pass
            self._self_bases = {}       # (location, worker assigned)
            self._enemy_bases = {}      # (location, worker assigned)
            self._neutral_bases = {}    # (location, true/false)
            self._self_units = {}       # (type, count)
            self._enemy_units = {}      # (tag, type)
            self._self_buildings = {}   # (type, count)
            self._self_upgrades = {}    # (type, true/false)
            self._training_queues = {}  # (type, [frames])
            self._building_queues = {}  # (type, queued_count)

        def is_unit(self, unit_type):
            return unit_type in [units.Protoss.Zealot, units.Protoss.Stalker, units.Protoss.Adept,
                                 units.Protoss.Probe, units.Protoss.Sentry]

        def is_building(self, unit_type):
            return unit_type in [units.Protoss.Nexus, units.Protoss.Gateway, units.Protoss.Assimilator,
                                 units.Protoss.CyberneticsCore, units.Protoss.Forge, units.Protoss.Pylon]

        def is_upgrade(self, action_id):
            if action_id == FUNCTIONS.Research_Blink_quick.id:
                return "blink"
            if action_id == FUNCTIONS.Research_ProtossGroundArmorLevel1_quick.id:
                return "ground armor 1"
            if action_id == FUNCTIONS.Research_ProtossGroundWeaponsLevel1_quick.id:
                return "ground weapon 1"
            return None

        def update(self, timestep):
            self._frame_pass += 1
            self._self_bases.clear()
            self._enemy_bases.clear()
            self._neutral_bases.clear()
            self._self_units.clear()
            self._self_buildings.clear()

            # check building queue
            for build in self._building_queues:
                if self._building_queues[build] > 0:
                    # last build count
                    last_build_count = self._self_buildings.get(build, 0)
                    # current build count
                    build_count = len([unit for unit in timestep.observation.raw_units
                                       if unit.unit_type == build and unit.alliance == 1])
                    if build_count > last_build_count:
                        self._building_queues[build] -= (build_count - last_build_count)
                    if self._building_queues[build] < 0:
                        self._building_queues[build] = 0

            # update training queue
            for unit in self._training_queues:
                index = 0
                training_queue = self._training_queues[unit]
                while index < len(training_queue):
                    training_queue[index] -= 1
                    if training_queue[index] == 0:
                        del training_queue[index]
                    else:
                        index += 1

            last_actions = timestep.observation.last_actions
            for last_action in last_actions:
                # update building queue
                if last_action in _PROTOSS_BUILDINGS_FUNCTIONS:
                    unit = _PROTOSS_BUILDINGS_FUNCTIONS[last_action].unit_type
                    self._building_queues[unit] = self._building_queues.get(unit, 0) + 1
                # update training queue
                elif last_action in _PROTOSS_UNITS_FUNCTIONS:
                    unit = _PROTOSS_UNITS_FUNCTIONS[last_action].unit_type
                    time = _PROTOSS_UNITS_FUNCTIONS[last_action].time
                    if unit in self._training_queues:
                        self._training_queues[unit].append(time)
                    else:
                        self._training_queues[unit] = [time]
                # update upgrades
                upgrade = self.is_upgrade(last_action)
                if upgrade:
                    self._self_upgrades[upgrade] = True

            # update unit/building count
            for unit in timestep.observation.raw_units:
                if unit.alliance == 1:
                    # update unit count
                    if self.is_unit(unit.unit_type):
                        self._self_units[unit.unit_type] =\
                            self._self_units.get(unit.unit_type, 0) + 1
                    # update building count
                    elif self.is_building(unit.unit_type):
                        self._self_buildings[unit.unit_type] =\
                            self._self_buildings.get(unit.unit_type, 0) + 1

        def to_feature(self):
            self_bases = [(key, self._self_bases[key]) for key in sorted(self._self_bases.keys())]
            enemy_bases = [(key, self._enemy_bases[key]) for key in sorted(self._enemy_bases.keys())]
            neutral_bases = [(key, self._neutral_bases[key]) for key in sorted(self._neutral_bases.keys())]
            self_units = [(key, self._self_units[key]) for key in sorted(self._self_units.keys())
                          if self._self_units[key] > 0]
            enemy_units = [(key, self._enemy_units[key]) for key in sorted(self._enemy_units.keys())
                           if self._enemy_units[key] > 0]
            self_buildings = [(key, self._self_buildings[key]) for key in sorted(self._self_buildings.keys())
                              if self._self_buildings[key] > 0]
            self_upgrades = [(key, self._self_upgrades[key]) for key in sorted(self._self_upgrades.keys())]
            training_queues = [(key, self._training_queues[key]) for key in sorted(self._training_queues.keys())
                               if len(self._training_queues[key]) > 0]
            building_queues = [(key, self._building_queues[key]) for key in sorted(self._building_queues.keys())
                               if self._building_queues[key] > 0]
            return [self._frame_pass] + self_bases + enemy_bases + neutral_bases + self_units +\
                   enemy_units + self_buildings + self_upgrades + training_queues + building_queues

    def __init__(self, config):
        self._config = config
        self._informations = []
        for _ in range(8):
            self._informations.append(self.DefaultInformation())

    def update(self, timesteps):
        for index, timestep in zip(range(len(timesteps)), timesteps):
            self._informations[index].update(timestep)

    def transform(self, timesteps):
        return [information.to_feature() for information in self._informations[:len(timesteps)]]

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
