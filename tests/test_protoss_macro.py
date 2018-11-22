# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/1.
#

import sys

sys.path.append('.')
from pysc2.lib import units
from pysc2.tests import utils
from absl.testing import absltest
from lib.protoss_macro import U
from lib.protoss_macro import PROTOSS_MACROS
from environment.macro_env import default_macro_env_maker
from numpy.random import randint


class TestProtossMacro(utils.TestCase):

    def test_train_probe(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_train_probe(obs):
                    act = PROTOSS_MACROS.Train_Probe()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_build_pylon(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_build_pylon(obs):
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_build_gateway(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_build_gateway(obs):
                    act = PROTOSS_MACROS.Build_Gateway()
                elif U.can_build_pylon(obs) and randint(0, 4) == 0:
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_train_zealot(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_train_zealot(obs):
                    act = PROTOSS_MACROS.Train_Zealot()
                elif U.can_build_gateway(obs) and obs._unit_counts.get(
                        units.Protoss.Gateway, 0) < 1:
                    act = PROTOSS_MACROS.Build_Gateway()
                elif U.can_build_pylon(obs) and obs._unit_counts.get(
                        units.Protoss.Pylon, 0) < 1:
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_build_assimilator(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_build_assimilator(obs):
                    act = PROTOSS_MACROS.Build_Assimilator()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_build_cyberneticscore(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_build_cyberneticscore(obs):
                    act = PROTOSS_MACROS.Build_CyberneticsCore()
                elif U.can_build_gateway(obs) and obs._unit_counts.get(
                        units.Protoss.Gateway, 0) < 1:
                    act = PROTOSS_MACROS.Build_Gateway()
                elif U.can_build_pylon(obs) and obs._unit_counts.get(
                        units.Protoss.Pylon, 0) < 1:
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    def test_train_stalker(self):
        with TestProtossMacro._create_macro_env() as env:
            obs = env.reset()[0]
            while True:
                act = PROTOSS_MACROS.No_op()
                if U.can_train_stalker(obs):
                    act = PROTOSS_MACROS.Train_Stalker()
                elif U.can_build_cyberneticscore(obs) and obs._unit_counts.get(
                        units.Protoss.CyberneticsCore, 0) < 1:
                    act = PROTOSS_MACROS.Build_CyberneticsCore()
                elif U.can_build_gateway(obs) and obs._unit_counts.get(
                        units.Protoss.Gateway, 0) < 1:
                    act = PROTOSS_MACROS.Build_Gateway()
                elif U.can_build_assimilator(obs):
                    act = PROTOSS_MACROS.Build_Assimilator()
                elif U.can_build_pylon(obs) and obs._unit_counts.get(
                        units.Protoss.Pylon, 0) < 1:
                    act = PROTOSS_MACROS.Build_Pylon()
                obs = env.step((act,))[0]
                if obs.last():
                    break

    @staticmethod
    def _create_macro_env():
        env = default_macro_env_maker(
            dict(map_name='Simple64', visualize=False))
        return env


if __name__ == "__main__":
    absltest.main()
