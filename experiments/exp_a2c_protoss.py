# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/12.
#

import sys
import argparse
import ast

sys.path.append('.')
import tensorflow as tf

from experiments.experiment import DistributedExperiment
from experiments.experiment import Experiment
from experiments.fcn import FCNNetwork2
from lib.config import Config
from lib.base import Network
from lib.protoss_macro import PROTOSS_MACROS
from environment.parallel_env import ParallelEnvs
from environment.macro_env import default_macro_env_maker
from environment.env_runner import EnvRunner
from algorithm.a2c import A2C
from lib.protoss_adapter import ProtossObservationAdapter as ObservationAdapter
from lib.protoss_adapter import ProtossMacroAdapter as MacroAdapter
from lib.protoss_adapter import ProtossRewardAdapter as RewardAdapter


def network_creator(config):
    def network_creator():
        def _network_creator():
            with tf.variable_scope('network'):
                value, policies, actions, states = FCNNetwork2.build_fcn(config)
            inputs = {
                'state': states,
                'reward': tf.placeholder(tf.float32, [None]),
                'done': tf.placeholder(tf.float32, [None]),
                'action': [tf.placeholder(tf.int32, [None]) for _ in range(len(actions))],
                'value': tf.placeholder(tf.float32, [None])}
            nets = {
                'policies': policies,
                'value': value,
                'actions': actions}
            return inputs, nets

        return Network(_network_creator, var_scope='network')

    return network_creator


class A2CProtossExperiment(DistributedExperiment):
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
        parser.add_argument("--visualize", type=ast.literal_eval, default=False)
        parser.add_argument("--difficulty", type=int, default=1)
        parser.add_argument("--mode", type=str, default='multi_thread')
        args, _ = parser.parse_known_args()
        self._local_args = args

    def run(self, global_args):
        local_args = self._local_args
        config = Config(
            screen_features=[],
            minimap_features=[],
            available_actions=PROTOSS_MACROS._macro_list,
            non_spatial_features=[
                'timestep_information',
                'available_actions'],
            non_spatial_feature_dims=dict(
                timestep_information=(19,),
                available_actions=(len(PROTOSS_MACROS),)))
        env_args = [{'map_name': "Simple64",
                     'difficulty': local_args.difficulty}
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
                env_makers=default_macro_env_maker,
                env_num=local_args.env_num,
                env_args=env_args)
            obs_adapter = ObservationAdapter(config)
            act_adapter = MacroAdapter(config)
            rwd_adapter = RewardAdapter(config, 8)
            env_runner = EnvRunner(
                agent=agent, env=env,
                train=global_args.train,
                observation_adapter=obs_adapter,
                action_adapter=act_adapter, epoch_n=local_args.epoch,
                reward_adapter=rwd_adapter,
                step_n=local_args.td_step,
                logdir=local_args.logdir)
            env_runner.run()


Experiment.register(A2CProtossExperiment, "A2C training protoss")

if __name__ == '__main__':
    Experiment.main()
