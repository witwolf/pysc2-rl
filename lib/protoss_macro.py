# Copyright (c) 2018 horizon robotics. All rights reserved.


import enum
import collections
import numbers
import numpy as np
import logging

from numpy.random import randint
from pysc2.lib import features

from lib.protoss_unit import *
from pysc2.lib.actions import FUNCTIONS


class U(object):
    @staticmethod
    def screen_size(obs):
        feature_screen = obs.observation.feature_screen
        screen_shape = feature_screen.shape
        h = screen_shape[-2]
        w = screen_shape[-1]
        return w, h

    @staticmethod
    def minimap_size(obs):
        feature_minimap = obs.observation.feature_minimap
        minimap_shape = feature_minimap.shape
        h = minimap_shape[-2]
        w = minimap_shape[-1]
        return w, h

    @staticmethod
    def base_minimap_location(obs):
        player_relative = \
            obs.observation.feature_minimap.player_relative
        player_self = features.PlayerRelative.SELF
        player_y, player_x = (
                player_relative == player_self).nonzero()
        if player_y.mean() > 31 and player_x.mean() > 31:
            x, y = 40, 46
        else:
            x, y = 19, 24
        return x, y

    @staticmethod
    def enemy_minimap_location(obs):
        base_x, base_y = U.base_minimap_location(obs)
        (x, y) = (40, 46) if base_x < 31 else (19, 24)
        return x, y

    @staticmethod
    def worker_type(obs):
        return units.Protoss.Probe

    @staticmethod
    def geyser_type(obs):
        return units.Protoss.Assimilator

    @staticmethod
    def locations_by_type(obs, unit_type):
        screen_w, screen_h = U.screen_size(obs)
        _feature_units = obs.observation.feature_units
        locations = [(unit.x, unit.y) for unit in _feature_units
                     if unit.unit_type == unit_type]
        return list(filter(
            lambda p: (p[0] > 0) and
                      (p[0] < screen_w) and
                      (p[1] > 0) and
                      p[1] < screen_h, locations))

    @staticmethod
    def random_unit_location(obs, unit_type):
        feature_units = obs.observation.feature_units
        units_pos = [(unit.x, unit.y)
                     for unit in feature_units if
                     unit.unit_type == unit_type]
        # todo check
        if (len(units_pos)) == 0:
            logging.warning("No units, unit_type:%d" % unit_type)
            return 0, 0
        random_pos = randint(0, len(units_pos))
        x, y = units_pos[random_pos]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def bad_worker_location(obs):
        assimilator_type = units.Protoss.Assimilator
        _feature_units = obs.observation.feature_units
        overload_geysers = [
            (unit.x, unit.y) for unit in _feature_units if
            unit.unit_type == assimilator_type and unit.assigned_harvesters >= 3]
        if len(overload_geysers) > 0:
            geyser_x, geyser_y = overload_geysers[0]
            probes = U.locations_by_type(
                units.Protoss.Probe)
            dists = np.ndarray(shape=[len(probes)], dtype=np.float32)
            for i, (x, y) in zip(range(len(probes)), probes):
                dists[i] = abs(x - geyser_x) + abs(y - geyser_y)
            pos = np.argmin(dists)
            x, y = probes[pos]
            return U._valid_screen_x_y(x, y, obs)
        return 1, 1

    @staticmethod
    def new_pylon_location(obs):
        screen_w, screen_h = U.screen_size(obs)
        x = randint(0, screen_w)
        y = randint(0, screen_h)
        return x, y

    @staticmethod
    def new_gateway_location(obs):
        screen_w, screen_h = U.screen_size(obs)
        x = randint(0, screen_w)
        y = randint(0, screen_h)
        return x, y

    @staticmethod
    def new_assimilator_location(obs):
        vesps = U.locations_by_type(obs, units.Neutral.VespeneGeyser)
        assimilators = U.locations_by_type(obs, units.Protoss.Assimilator)
        valid_vespenes = list(set(vesps) - set(assimilators))

        if len(valid_vespenes) == 0:
            return 1, 1
        x, y = valid_vespenes[randint(0, len(valid_vespenes))]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def new_cyberneticscore_location(obs):
        screen_w, screen_h = U.screen_size(obs)
        x = randint(0, screen_w)
        y = randint(0, screen_h)
        return x, y

    @staticmethod
    def mineral_location(obs):
        minerals = U.locations_by_type(obs, units.Neutral.MineralField)
        # if no mineral, go to center
        if len(minerals) == 0:
            return 1, 1
        x, y = minerals[randint(0, len(minerals))]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def gas_location(obs):
        _feature_units = obs.observation.feature_units
        assimilator_type = units.Protoss.Assimilator
        assimilators = [
            unit for unit in _feature_units if
            unit.unit_type == assimilator_type]
        for assimilator in assimilators:
            if assimilator.assigned_harvesters < 3:
                return U._valid_screen_x_y(
                    assimilator.x, assimilator.y, obs)
        # if no mineral, go to center
        return 1, 1

    @staticmethod
    def screen_top(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w / 2, 0

    @staticmethod
    def screen_topright(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, 0

    @staticmethod
    def screen_right(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, screen_h / 2

    @staticmethod
    def screen_bottomright(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w - 1, screen_h - 1

    @staticmethod
    def screen_bottom(obs):
        screen_w, screen_h = U.screen_size(obs)
        return screen_w / 2, screen_h - 1

    @staticmethod
    def screen_bottomleft(obs):
        screen_w, screen_h = U.screen_size(obs)
        return 0, screen_h - 1

    @staticmethod
    def screen_left(obs):
        screen_w, screen_h = U.screen_size(obs)
        return 0, screen_h / 2

    @staticmethod
    def screen_topleft(obs):
        return 0, 0

    @staticmethod
    def army_minimap_location(obs):
        player_relative = \
            obs.observation.feature_minimap.player_relative
        player_self = features.PlayerRelative.SELF
        ys, xs = \
            (player_relative == player_self).nonzero()
        if len(xs) == 0:
            return U.base_minimap_location(obs)
        enemy_x, enemy_y = U.enemy_minimap_location(obs)
        dists = np.ndarray(shape=[len(xs)], dtype=np.float32)
        for (i, x, y) in zip(range(len(xs)), xs, ys):
            dists[i] = abs(x - enemy_x) + abs(y - enemy_y)
        pos = np.argmin(dists)
        x, y = xs[pos], ys[pos]
        return U._valid_minimap_x_y(x, y, obs)

    @staticmethod
    def attack_location(obs):
        player_relative = \
            obs.observation.feature_screen.player_relative
        player_enemy = features.PlayerRelative.ENEMY
        player_self = features.PlayerRelative.SELF
        enemy_ys, enemy_xs = \
            (player_relative == player_enemy).nonzero()
        if len(enemy_xs) == 0:
            # if no enemy on screen follow army-enemy direction
            return U.attack_location_army2enemy(obs)

        # if enemy on screen follow enemy closest
        army_ys, army_xs = (player_relative == player_self).nonzero()
        army_c_x, army_c_y = (army_xs.mean(), army_ys.mean())
        distances = np.ndarray(shape=[len(enemy_xs)], dtype=np.float32)
        for i, x, y in zip(range(len(enemy_xs)), enemy_xs, enemy_ys):
            distances[i] = abs(x - army_c_x) + abs(y - army_c_y)
        pos = np.argmin(distances)
        x, y = enemy_xs[pos], enemy_ys[pos]
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def attack_location_army2enemy(obs):
        army_front = U.army_minimap_location(obs)
        enemy_base = U.enemy_minimap_location(obs)
        screen_w, screen_h = U.screen_size(obs)
        attack_direction = (
            enemy_base[0] - army_front[0],
            enemy_base[1] - army_front[1])
        radius = attack_direction[0] * attack_direction[0] \
                 + attack_direction[1] * attack_direction[1]

        radius = np.clip(radius, 1e-1, None)
        ratio = float(min(screen_w, screen_h) / 2.5) / float(radius)
        x, y = (attack_direction[0] * ratio - screen_h / 2,
                attack_direction[1] * ratio - screen_h / 2)
        return U._valid_screen_x_y(x, y, obs)

    @staticmethod
    def _valid_screen_x_y(x, y, obs):
        screen_w, screen_h = U.screen_size(obs)
        _x = np.clip(x, 0, screen_w - 1)
        _y = np.clip(y, 0, screen_h - 1)
        return _x, _y

    @staticmethod
    def _valid_minimap_x_y(x, y, obs):
        minimap_w, minimap_h = U.minimap_size(obs)
        _x = np.clip(x, 0, minimap_w - 1)
        _y = np.clip(y, 0, minimap_h - 1)
        return _x, _y

    @staticmethod
    def _can_build_pylon(obs):
        return obs.observation.player.minerals >= Pylon.minerals

    @staticmethod
    def _can_build_gateway(obs):
        return obs.observation.player.minerals >= Gateway.minerals

    @staticmethod
    def _can_build_assimilator(obs):
        return obs.observation.player.minerals >= Assimilator.minerals

    @staticmethod
    def _can_build_cyberneticscore(obs):
        return obs.observation.player.minerals >= CyberneticsCore.minerals

    @staticmethod
    def _can_training_probe(obs):
        # TODO
        return True

    @staticmethod
    def _can_train_zealot(obs):
        # TODO
        return True

    @staticmethod
    def _can_train_stalker(obs):
        # TODO
        return True

    @staticmethod
    def _can_select_army(obs):
        return obs.observation.player.army_count > 0


def build_a_pylon():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # build a pylon
        FUNCTIONS.Build_Pylon_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.new_pylon_location(obs))]
    cond = U._can_build_pylon
    return cond, funcs, funcs_args


def build_a_gateway():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # build a gateway
        FUNCTIONS.Build_Gateway_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.new_gateway_location(obs))]
    cond = U._can_build_gateway
    return cond, funcs, funcs_args


def build_a_assimilator():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # build a assimilator
        FUNCTIONS.Build_Assimilator_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.new_assimilator_location(obs))]
    cond = U._can_build_assimilator
    return cond, funcs, funcs_args


def build_a_cyberneticscore():
    cond = U._can_build_cyberneticscore
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all workers
        FUNCTIONS.select_point,
        # build a CyberneticsCore
        FUNCTIONS.Build_CyberneticsCore_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.new_cyberneticscore_location(obs))]
    return cond, funcs, funcs_args


def training_a_probe():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a zealot
        FUNCTIONS.Train_Probe_quick]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, units.Protoss.Nexus)),
        lambda obs: ("now",)]
    cond = U._can_training_probe
    return cond, funcs, funcs_args


def training_a_zealot():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a zealot
        FUNCTIONS.Train_Zealot_quick]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, units.Protoss.Gateway)),
        lambda obs: ("now",)]
    cond = U._can_train_zealot
    return cond, funcs, funcs_args


def training_a_stalker():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select all gateways
        FUNCTIONS.select_point,
        # train a stalker
        FUNCTIONS.Train_Stalker_quick]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select_all_type", U.random_unit_location(
            obs, units.Protoss.Gateway)),
        lambda obs: ("now",)]
    cond = U._can_train_stalker
    return cond, funcs, funcs_args


def callback_idle_workers():
    cond = lambda obs: obs.observation.player.idle_worker_count > 0
    funcs = [
        # select idle workers
        FUNCTIONS.select_idle_worker,
        # move camera to base
        FUNCTIONS.move_camera,
        # collect minerals
        FUNCTIONS.Harvest_Gather_screen]
    funcs_args = [
        lambda obs: ("select_all",),
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("now", U.mineral_location(obs))]

    return cond, funcs, funcs_args


def collect_minerals():
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select a worker
        FUNCTIONS.select_point,
        # collect minerals
        FUNCTIONS.Smart_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select", U.bad_worker_location(obs)),
        lambda obs: ("now", U.gas_location(obs))]
    cond = lambda obs: True
    return cond, funcs, funcs_args


def collect_gas():
    cond = lambda obs: True
    funcs = [
        # move camera to base
        FUNCTIONS.move_camera,
        # select a worker
        FUNCTIONS.select_point,
        # collect gas
        FUNCTIONS.Harvest_Gather_screen]
    funcs_args = [
        lambda obs: (U.base_minimap_location(obs),),
        lambda obs: ("select", U.random_unit_location(
            obs, U.worker_type(obs))),
        lambda obs: ("now", U.gas_location(obs))]

    return cond, funcs, funcs_args


def move_screen_topleft():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_topleft(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_top():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_top(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_topright():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_topright(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_right():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_right(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_bottomright():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottomright(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_bottom():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottom(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_bottomleft():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_bottomleft(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def move_screen_left():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # move to one corner
        FUNCTIONS.Move_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.screen_left(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


def attack_enemy():
    funcs = [
        # select all army
        FUNCTIONS.select_army,
        # move camera to army
        FUNCTIONS.move_camera,
        # attack enemy
        FUNCTIONS.Attack_screen]
    funcs_args = [
        lambda obs: ("select",),
        lambda obs: (U.army_minimap_location(obs),),
        lambda obs: ("now", U.attack_location(obs))]
    cond = U._can_select_army
    return cond, funcs, funcs_args


class ProtossMacro(collections.namedtuple(
    "ProtossMacro", ["id", "name", "ability_id",
                     "general_id", "cond", "func", "args"])):
    __slots__ = ()

    @classmethod
    def ability(cls, id_, name, func, ability_id, general_id=0):
        values = func()
        cond = values[0]
        func = list(zip(*values[1:]))
        return cls(id_, name, ability_id, general_id, cond, func, [])

    def __hash__(self):
        return self.id

    def __call__(self):
        return MacroCall(self)


_PROTOSS_MACROS = [
    ProtossMacro.ability(0, "Build_Pylon", build_a_pylon, 0),
    ProtossMacro.ability(1, "Build_Gateway", build_a_gateway, 1),
    ProtossMacro.ability(2, "Build_Assimilator", build_a_assimilator, 2),
    ProtossMacro.ability(3, "Build_CyberneticsCore", build_a_cyberneticscore, 3),
    ProtossMacro.ability(4, "Train_Probe", training_a_probe, 4),
    ProtossMacro.ability(5, "Train_Zealot", training_a_zealot, 5),
    ProtossMacro.ability(6, "Train_Stalker", training_a_stalker, 6),
    ProtossMacro.ability(7, "Callback_Idle_Workers", callback_idle_workers, 7),
    ProtossMacro.ability(8, "Collect_Gas", collect_gas, 8),
    ProtossMacro.ability(9, "Move_TopLeft", move_screen_topleft, 9),
    ProtossMacro.ability(10, "Move_Top", move_screen_top, 10),
    ProtossMacro.ability(11, "Move_TopRight", move_screen_topright, 11),
    ProtossMacro.ability(12, "Move_Right", move_screen_right, 12),
    ProtossMacro.ability(13, "Move_BottomRight", move_screen_bottomright, 13),
    ProtossMacro.ability(14, "Move_Bottom", move_screen_bottom, 14),
    ProtossMacro.ability(15, "Move_BottomLeft", move_screen_bottomleft, 15),
    ProtossMacro.ability(16, "Move_Left", move_screen_left, 16),
    ProtossMacro.ability(17, "Attack_Enemy", attack_enemy, 17)
    # ProtossMacro.ability(8, "Collect_Mineral", collect_minerals, 8),
]


class MacroCall(object):

    def __init__(self, macro):
        func = macro.func
        self.id = macro.id
        self.func = func

    def __getitem__(self, item):
        return self.func[item]

    def __iter__(self):
        return iter(self.func)

    def __len__(self):
        return len(self.func)

    def __str__(self):
        return str(self.id)


class ProtossMacros(object):
    def __init__(self, macros):
        macros = sorted(macros, key=lambda f: f.id)
        macros = [f._replace(id=_ProtossMacros(f.id))
                  for f in macros]

        self._macro_list = macros
        self._macro_dict = {f.name: f for f in macros}
        if len(self._macro_dict) != len(self._macro_list):
            raise ValueError("Macro names must be unique.")

    def __getattr__(self, name):
        return self._macro_dict.get(name, None)

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self._macro_list[key]
        return self._macro_dict[key]

    def __getstate__(self):
        return self._macro_list

    def __setstate__(self, macros):
        self.__init__(macros)

    def __iter__(self):
        return iter(self._macro_list)

    def __len__(self):
        return len(self._macro_list)

    def __eq__(self, other):
        return self._macro_list == other._macro_list


_ProtossMacros = enum.IntEnum(
    "_ProtossMacros", {f.name: f.id for f in _PROTOSS_MACROS})

PROTOSS_MACROS = ProtossMacros(_PROTOSS_MACROS)

_PROTOSS_UNITS = [
    Zealot, Stalker, Sentry, Adept,
    HighTemplar, DarkTemplar, Archon, Observer,
    WarpPrism, Immortal,
    Colossus, Disruptor, Phoenix, VoidRay, Oracle,
    Tempest, Carrier, Interceptor, Mothership]

_PROTOSS_UNITS_MACROS = {
    PROTOSS_MACROS.Train_Probe.id: Probe,
    PROTOSS_MACROS.Train_Zealot.id: Zealot,
    PROTOSS_MACROS.Train_Stalker.id: Stalker
}

_PROTOSS_UNITS_FUNCTIONS = {
    FUNCTIONS.Train_Probe_quick.id: Probe,
    FUNCTIONS.Train_Zealot_quick.id: Zealot,
    FUNCTIONS.Train_Stalker_quick.id: Stalker
}

_PROTOSS_BUILDINGS = [
    Nexus, Pylon, Assimilator, Gateway,
    WarpGate, Forge, CyberneticsCore,
    PhotonCannon, ShieldBattery,
    RoboticsFacility, Stargate, TwilightCouncil,
    RoboticsBay, FleetBeacon, TemplarArchive, DarkShrine,
    StasisTrap]

_PROTOSS_BUILDINGS_MACROS = {
    PROTOSS_MACROS.Build_Pylon.id: Pylon,
    PROTOSS_MACROS.Build_Gateway.id: Gateway,
    PROTOSS_MACROS.Build_Assimilator.id: Assimilator,
    PROTOSS_MACROS.Build_CyberneticsCore.id: CyberneticsCore
}

_PROTOSS_BUILDINGS_FUNCTIONS = {
    FUNCTIONS.Build_Pylon_screen.id: Pylon,
    FUNCTIONS.Build_Gateway_screen.id: Gateway,
    FUNCTIONS.Build_Assimilator_screen.id: Assimilator,
    FUNCTIONS.Build_CyberneticsCore_screen.id: CyberneticsCore
}
