# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import sys

sys.path.append('.')

import argparse
import ast
import tensorflow as tf
from experiments.experiment import Experiment
from environment.parallel_env import ParallelEnvs
from environment.env_runner import EnvRunner
from lib.adapter import DefaultObservationAdapter as ObservationAdapter
from lib.adapter import DefaultActionAdapter as ActionAdapter
from lib.adapter import DefaultProtossMacroAdapter as MacroAdapter
from algorithm.a2c import A2C
from lib.config import Config

from experiments.exp_bc import network_creator
from lib.protoss_macro import PROTOSS_MACROS


# a2c train
class A2CExperiment(Experiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="MoveToBeacon")
        parser.add_argument("--env_num", type=int, default=32, help='env parallel run')
        parser.add_argument("--epoch", type=int, default=1024),
        parser.add_argument("--batch", type=int, default=256),
        parser.add_argument("--td_step", type=int, default=16, help='td(n)')
        parser.add_argument("--logdir", type=str, default='log/a2c')
        parser.add_argument("--v_coef", type=float, default=0.25)
        parser.add_argument("--ent_coef", type=float, default=1e-3)
        parser.add_argument("--lr", type=float, default=7e-4)
        parser.add_argument("--train", type=ast.literal_eval, default=True)
        parser.add_argument("--restore", type=ast.literal_eval, default=False)
        args, _ = parser.parse_known_args()
        self._args = args

    def run(self):
        args = self._args
        with tf.Session() as sess:
            config = Config()
            agent = A2C(
                network_creator=network_creator(config),
                lr=args.lr,
                td_step=args.td_step,
                ent_coef=args.ent_coef,
                v_coef=args.v_coef,
                sess=sess)
            env_args = {'map_name': args.map_name}
            if args.train:
                env = ParallelEnvs(
                    env_num=args.env_num,
                    env_args=env_args)
            else:
                env = None
            test_env = ParallelEnvs(
                env_num=1,
                env_args=env_args)
            obs_adapter = ObservationAdapter(config)
            act_adapter = ActionAdapter(config)
            env_runner = EnvRunner(
                agent=agent,
                env=env,
                test_env=test_env,
                observation_adapter=obs_adapter,
                action_adapter=act_adapter,
                epoch_n=args.epoch,
                batch_n=args.batch,
                step_n=args.td_step,
                restore=args.restore,
                test_after_epoch=True,
                logdir=args.logdir)
            env_runner.run()


Experiment.register(A2CExperiment, "A2C training")


class A2CProtossExperiment(A2CExperiment):

    def __init__(self):
        super().__init__()

    def run(self):
        args = self._args
        from environment.macro_env import default_macro_env_maker
        with tf.Session() as sess:
            config = Config(
                available_actions=PROTOSS_MACROS._macro_list,
                non_spatial_features=['player'])
            agent = A2C(
                network_creator=network_creator(config),
                lr=args.lr,
                td_step=args.td_step,
                ent_coef=args.ent_coef,
                v_coef=args.v_coef,
                sess=sess)
            env_args = {'map_name': "Simple64"}
            if args.train:
                env = ParallelEnvs(
                    env_makers=default_macro_env_maker,
                    env_num=args.env_num,
                    env_args=env_args)
            else:
                env = None
            test_env = ParallelEnvs(
                env_makers=default_macro_env_maker,
                env_num=1,
                env_args=env_args)
            obs_adapter = ObservationAdapter(config)
            act_adapter = MacroAdapter(config)
            env_runner = EnvRunner(
                agent=agent,
                env=env,
                test_env=test_env,
                observation_adapter=obs_adapter,
                action_adapter=act_adapter,
                epoch_n=args.epoch,
                batch_n=args.batch,
                step_n=args.td_step,
                restore=args.restore,
                test_after_epoch=True,
                logdir=args.logdir)
            env_runner.run()


Experiment.register(A2CProtossExperiment, "A2C training protoss")

if __name__ == '__main__':
    Experiment.main()
