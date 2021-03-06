# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import sys

sys.path.append('.')

import argparse
import ast
import tensorflow as tf
from experiments.experiment import Experiment
from experiments.experiment import DistributedExperiment
from environment.parallel_env import ParallelEnvs
from environment.env_runner import EnvRunner
from lib.adapter import DefaultObservationAdapter as ObservationAdapter
from lib.adapter import DefaultActionAdapter as ActionAdapter
from algorithm.a2c import A2C
from lib.config import Config
from experiments.exp_bc import network_creator


# a2c train
class A2CExperiment(DistributedExperiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="MoveToBeacon")
        parser.add_argument("--env_num", type=int, default=32)
        parser.add_argument("--epoch", type=int, default=8096),
        parser.add_argument("--td_step", type=int, default=16, help='td(n)')
        parser.add_argument("--logdir", type=str, default='log/a2c')
        parser.add_argument("--v_coef", type=float, default=0.25)
        parser.add_argument("--ent_coef", type=float, default=1e-3)
        parser.add_argument("--lr", type=float, default=7e-4)
        parser.add_argument("--mode", type=str, default='multi_thread')
        parser.add_argument("--visualize", type=ast.literal_eval, default=False)
        args, _ = parser.parse_known_args()
        self._local_args = args

    def run(self, global_args):
        local_args = self._local_args
        config = Config()
        env_args = [{'map_name': local_args.map_name}
                    for _ in range(local_args.env_num)]
        env_args[0]['visualize'] = local_args.visualize
        with tf.device(self.tf_device(global_args)):
            agent = A2C(
                network_creator=network_creator(config),
                lr=local_args.lr, td_step=local_args.td_step,
                ent_coef=local_args.ent_coef, v_coef=local_args.v_coef)
        with agent.create_session(**self.tf_sess_opts(global_args)):
            env = ParallelEnvs.new(
                mode=local_args.mode,
                env_num=local_args.env_num,
                env_args=env_args)
            obs_adapter = ObservationAdapter(config)
            act_adapter = ActionAdapter(config)
            env_runner = EnvRunner(
                agent=agent, env=env,
                train=global_args.train,
                observation_adapter=obs_adapter,
                action_adapter=act_adapter,
                epoch_n=local_args.epoch,
                step_n=local_args.td_step,
                logdir=local_args.logdir)
            env_runner.run()


Experiment.register(A2CExperiment, "A2C training")

if __name__ == '__main__':
    Experiment.main()
