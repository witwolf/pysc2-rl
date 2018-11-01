# Copyright (c) 2018 horizon robotics. All rights reserved.


import enum
import collections
import numbers
import numpy as np

from numpy.random import randint
from pysc2.lib import units
from pysc2.lib import features
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
    def base_minimap_location(obs):
        player_relative = \
            obs.observation.feature_minimap.player_relative
        player_self = features.PlayerRelative.SELF
        player_y, player_x = (
                player_relative == player_self).nonzero()
        if player_y.mean() > 31 and player_x.mean() > 31:
            x, y = 51, 49
        else:
            x, y = 10, 15
        return x, y

    @staticmethod
    def enemy_minimap_location(obs):
        base_x, base_y = U.base_minimap_location(obs)
        (x, y) = (49, 51) if base_x < 31 else (15, 10)
        return x, y

    @staticmethod
    def worker_type(obs):
        return units.Protoss.Probe

    @staticmethod
    def geyser_type(obs):
        return units.Protoss.Assimilator

    @staticmethod
    def random_unit_location(obs, unit_type):
        feature_units = obs.observation.feature_units
        units_pos = [(unit.x, unit.y)
                     for unit in feature_units if
                     unit.unit_type == unit_type]
        random_pos = randint(0, len(units_pos))
        return units_pos[random_pos]

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
        screen_w, screen_h = U.screen_size(obs)
        x = randint(0, screen_w)
        y = randint(0, screen_h)
        return x, y

    @staticmethod
    def new_cyberneticscore_location(obs):
        screen_w, screen_h = U.screen_size(obs)
        x = randint(0, screen_w)
        y = randint(0, screen_h)
        return x, y

    @staticmethod
    def minieral_location(obs):
        unit_type = obs.observation.feature_screen.unit_type
        mineral_type = units.Neutral.MineralFiel
        xs, ys = (unit_type == mineral_type).nonzero()
        # if no mineral, go to center
        if len(xs) == 0:
            screen_w, screen_h = U.screen_size(obs)
            return screen_w / 2, screen_h / 2
        mineral_id = randint(0, len(xs))
        return xs[mineral_id], ys[mineral_id]

    @staticmethod
    def gas_location(obs):
        unit_type = obs.observation.feature_screen.unit_type
        gas_type = U.geyser_type(obs)
        xs, ys = (unit_type == gas_type).nonzero()
        # if no gas, go to center
        if len(xs) == 0:
            screen_w, screen_h = U.screen_size(obs)
            return screen_w / 2, screen_h / 2
        gas_id = randint(0, len(xs))
        return xs[gas_id], ys[gas_id]

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
        xs, ys = \
            (player_relative == player_self).nonzero()
        if len(xs) == 0:
            return U.base_minimap_location(obs)
        enemy_x, enemy_y = U.enemy_minimap_location(obs)
        dists = np.ndarray(shape=[len(xs)], dtype=np.float32)
        for (i, x, y) in zip(range(len(xs)), xs, ys):
            dists[i] = abs(x - enemy_x) + abs(y - enemy_y)
        pos = np.argmin(dists)
        return xs[pos], ys[pos]

    @staticmethod
    def attack_location(obs):
        player_relative = \
            obs.observation.feature_screen.player_relative
        player_enemy = features.PlayerRelative.ENEMY
        player_self = features.PlayerRelative.SELF
        enemy_xs, enemy_ys = \
            (player_relative == player_enemy).nonzero()
        if len(enemy_xs[0]) == 0:
            # if no enemy on screen follow army-enemy direction
            return U.attack_location_army2enemy(obs)

        # if enemy on screen follow enemy closest
        army_xs, army_ys = (player_relative == player_self).nonzero()
        army_c_x, army_c_y = (army_xs.mean(), army_ys.mean())
        distances = np.ndarray(shape=[len(enemy_xs)], dtype=np.float32)
        for i, x, y in zip(range(len(enemy_xs)), enemy_xs, enemy_ys):
            distances[i] = abs(x - army_c_x) + abs(y - army_c_y)
        pos = np.argmin(distances)
        return enemy_xs[pos], enemy_ys[pos]

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
        ratio = float(min(screen_w, screen_h) / 2.5) / float(radius)
        return (attack_direction[0] * ratio - screen_h / 2,
                attack_direction[1] * ratio - screen_h / 2)


def _move_to_base(obs, *args):
    # move camera to base
    base_location = U.base_minimap_location(obs)
    move_to_base = FUNCTIONS.move_camera((base_location,))
    return [move_to_base]


def _select_worker(obs, *args):
    # select worker
    _worker_type = U.worker_type(obs)
    worker_location = U.random_unit_location(obs, _worker_type)
    select_worker = FUNCTIONS.select_point(
        "select_all_type", worker_location)
    return [select_worker]


def _select_gateway(obs, *args):
    # select gateway
    gateway_type = units.Protoss.Gateway
    gateway_location = U.random_unit_location(obs, gateway_type)
    select_gateway = FUNCTIONS.select_point(
        "select_all_type", gateway_location)
    return [select_gateway]


# macros

def build_a_pylon(obs, *args):
    move_to_base = _move_to_base(obs, *args)
    select_worker = _select_worker(obs, *args)
    # build pylon
    pylon_location = U.new_pylon_location(obs)
    build_pylon = FUNCTIONS.Build_Pylon_screen(
        "now", pylon_location)
    return move_to_base + select_worker + [build_pylon]


def build_a_gateway(obs, *args):
    move_to_base = _move_to_base(obs, *args)
    select_worker = _select_worker(obs, *args)
    # build gateway
    gateway_location = U.new_gateway_location(obs)
    build_gateway = FUNCTIONS.Build_Gateway_screen(
        "now", gateway_location)
    return move_to_base + select_worker + [build_gateway]


def build_a_assimilator(obs, *args):
    move_to_base = _move_to_base(obs, *args)
    select_worker = _select_worker(obs, *args)
    # build assimilator
    assimilator_location = U.new_assimilator_location(obs)
    build_assimilator = FUNCTIONS.Build_Assimilator_screen(
        "now", assimilator_location)
    return move_to_base + select_worker + [build_assimilator]


def build_a_CyberneticsCore(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs, *args)
    # select worker
    select_worker = _select_worker(obs, *args)
    # build cyberneticscore
    cybernet_location = U.new_cyberneticscore_location(obs)
    build_cybernet = FUNCTIONS.Build_CyberneticsCore_screen(
        "now", cybernet_location)
    return move_to_base + select_worker + [build_cybernet]


def training_a_probe(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs, *args)
    # select worker
    worker_type = units.Protoss.Nexus
    worker_location = U.random_unit_location(obs, worker_type)
    select_worker = FUNCTIONS.select_point(
        "select_all_type", worker_location)
    # train probe
    train_probe = FUNCTIONS.Train_Probe_quick("now", )
    return move_to_base + [select_worker, train_probe]


def train_a_zealot(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs, *args)
    # select gateway
    select_gateway = _select_gateway(obs, *args)
    # train zealot
    train_zealot = FUNCTIONS.Train_Zealot_quick("now", )
    return move_to_base + select_gateway + [train_zealot]


def training_a_stalker(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs, *args)
    # select gateway
    select_gateway = _select_gateway(obs, *args)
    # train stalker
    train_stalker = FUNCTIONS.Train_Stalker_quick("now", )
    return move_to_base + select_gateway + [train_stalker]


def collect_minerals(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs, *args)
    # select idle worker
    select_idle_worker = FUNCTIONS.select_idle_worker("select_all")
    # collect mineral
    mineral_location = U.minieral_location(obs)
    collect_mineral = FUNCTIONS.Smart_screen("now", mineral_location)
    return move_to_base + [select_idle_worker, collect_mineral]


def collect_gas(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select worker
    _worker_type = U.worker_type(obs)
    worker_location = U.random_unit_location(obs, _worker_type)
    select_worker = FUNCTIONS.select_point("select", worker_location)
    # collect gas
    _gas_location = U.gas_location(obs)
    collect_gas = FUNCTIONS.Smart_screen("now", _gas_location)
    return move_to_base + [select_worker, collect_gas]


def move_screen_topleft(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    top_left = U.screen_topleft(obs)
    move_screen = FUNCTIONS.Move_screen("now", top_left)
    return move_to_base + [select_army, move_screen]


def move_screen_top(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    top = U.screen_top(obs)
    move_screen = FUNCTIONS.Move_screen("now", top)
    return move_to_base + [select_army, move_screen]


def move_screen_topright(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    top_right = U.screen_topright(obs)
    move_screen = FUNCTIONS.Move_screen("now", top_right)
    return move_to_base + [select_army, move_screen]


def move_screen_right(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    right = U.screen_right(obs)
    move_screen = FUNCTIONS.Move_screen("now", right)
    return move_to_base + [select_army, move_screen]


def move_screen_bottomright(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    bottom_right = U.screen_bottomright(obs)
    move_screen = FUNCTIONS.Move_screen("now", bottom_right)
    return move_to_base + [select_army, move_screen]


def move_screen_bottom(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    bottom = U.screen_bottom(obs)
    move_screen = FUNCTIONS.Move_screen("now", bottom)
    return move_to_base + [select_army, move_screen]


def move_screen_bottomleft(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    bottom_left = U.screen_bottomleft(obs)
    move_screen = FUNCTIONS.Move_screen("now", bottom_left)
    return move_to_base + [select_army, move_screen]


def move_screen_left(obs, *args):
    # move to base
    move_to_base = _move_to_base(obs)
    # select army
    select_army = FUNCTIONS.select_army("select", )
    # move screen
    left = U.screen_left(obs)
    move_screen = FUNCTIONS.Move_screen("now", left)
    return move_to_base + [select_army, move_screen]


def attack_enemy(obs, *args):
    # move to army
    army_location = U.army_minimap_location(obs)
    move_to_army = FUNCTIONS.move_camera(army_location)
    # select all army
    select_army = FUNCTIONS.select_army("select")
    # attack enemy
    attack_location = U.attack_location(obs)
    attack_enemy = FUNCTIONS.Attack_screen("now", attack_location)
    return [move_to_army, select_army, attack_enemy]


class ProtossMacro(collections.namedtuple(
    "ProtossMacro", ["id", "name", "ability_id",
                     "general_id", "function_calls", "args"])):
    __slots__ = ()

    @classmethod
    def ability(cls, id_, name, function_calls, ability_id, general_id=0):
        return cls(id_, name, ability_id, general_id, function_calls, [])

    def __hash__(self):
        return self.id

    def __call__(self, *args):
        return MacroCall.init(self.id, args)


_PROTOSS_MACROS = [
    ProtossMacro.ability(0, "Build_Pylon", build_a_pylon, 0),
    ProtossMacro.ability(1, "Build_Gateway", build_a_gateway, 1),
    ProtossMacro.ability(2, "Build_Assimilator", build_a_assimilator, 2),
    ProtossMacro.ability(3, "Build_CyberneticsCore", build_a_CyberneticsCore, 3),
    ProtossMacro.ability(4, "Train_Probe", training_a_probe, 4),
    ProtossMacro.ability(5, "Train_Zealot", train_a_zealot, 5),
    ProtossMacro.ability(6, "Train_Stalker", training_a_stalker, 6),
    ProtossMacro.ability(7, "Collect_Mineral", collect_minerals, 7),
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
]


class MacroCall(collections.namedtuple(
    "MacroCall", ["macro", "arguments"])):
    __slots__ = ()

    @classmethod
    def init(cls, macro, arguments):
        macro = PROTOSS_MACROS[macro]
        return cls(macro.id, arguments)


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
        return self._macro_dict[name]

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
