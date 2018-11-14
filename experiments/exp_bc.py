# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


import sys
import ast

sys.path.append('.')

import argparse
import tensorflow as tf
from experiments.experiment import Experiment
from experiments.experiment import DistributedExperiment
from algorithm.bc import BehaviorClone
from environment.parallel_env import ParallelEnvs
from algorithm.script.parallel_agent import ParallelAgent
from environment.env_runner import EnvRunner
from lib.config import Config
from lib.adapter import DefaultActionAdapter as ActionAdapter
from lib.adapter import DefaultObservationAdapter as ObservationAdapter
from experiments.fcn import FCNNetwork

from lib.base import Network
from algorithm.script.parallel_agent import script_agent_maker


def network_creator(config):
    def network_creator():
        def _network_creator():
            with tf.variable_scope('network'):
                value, policies, actions, states = FCNNetwork.build_fcn(config)
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


# behavior clone train (imitation)
class BCExperiment(DistributedExperiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="MoveToBeacon")
        parser.add_argument("--env_num", type=int, default=32)
        parser.add_argument("--epoch", type=int, default=8096),
        parser.add_argument("--td_step", type=int, default=16)
        parser.add_argument("--logdir", type=str, default='log/bc')
        parser.add_argument("--v_coef", type=float, default=0.0)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--visualize", type=ast.literal_eval, default=False)

        args, _ = parser.parse_known_args()
        self._local_args = args

    def run(self, global_args):
        local_args = self._local_args
        config = Config()
        script_agents = ParallelAgent(
            agent_num=local_args.env_num,
            agent_makers=script_agent_maker(local_args.map_name))
        env_args = [{'map_name': local_args.map_name}
                    for _ in range(local_args.env_num)]
        env_args[0]['visualize'] = local_args.visualize
        with tf.device(self.tf_device(global_args)):
            agent = BehaviorClone(
                network_creator=network_creator(config),
                td_step=local_args.td_step, lr=local_args.lr,
                v_coef=local_args.v_coef, script_agents=script_agents)
        with agent.create_session(**self.tf_sess_opts(global_args)):
            env = ParallelEnvs(
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


Experiment.register(BCExperiment, 'Behavior clone training')

if __name__ == '__main__':
    Experiment.main()
