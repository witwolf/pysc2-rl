# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


import argparse
from experiments.experiment import Experiment
from algorithm.bc import BehaviorClone
from environment.parallel_env import ParallelEnvs
from algorithm.script.parallel_agent import ParallelAgent
from environment.env_runner import EnvRunner
from lib.config import Config
from lib.adapter import DefaultActionAdapter as ActionAdapter
from lib.adapter import DefaultObservationAdapter as ObservationAdapter
from experiments.fcn import FCNNetwork
import tensorflow as tf
from lib.base import Network
from algorithm.script.parallel_agent import script_agent_maker


def network_creator(config):
    def network_creator():
        def _network_creator():
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
class BCExperiment(Experiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="MoveToBeacon")
        parser.add_argument("--env_num", type=int, default=32, help='env parallel run')
        parser.add_argument("--epoch", type=int, default=1024),
        parser.add_argument("--batch", type=int, default=256),
        parser.add_argument("--td_step", type=int, default=16, help='td(n)')

        args, _ = parser.parse_known_args()
        print(args)
        config = Config()
        script_agents = ParallelAgent(agent_num=args.env_num,
                                      agent_makers=script_agent_maker(args.map_name))

        sess = tf.Session()
        agent = BehaviorClone(network_creator=network_creator(config),
                              script_agents=script_agents, sess=sess)
        env = ParallelEnvs(
            env_num=args.env_num,
            env_args={'map_name': args.map_name})
        obs_adapter = ObservationAdapter(config)
        act_adapter = ActionAdapter(config)

        self._env_runner = EnvRunner(
            agent=agent,
            env=env,
            observation_adapter=obs_adapter,
            action_adapter=act_adapter,
            epoch_n=args.epoch,
            batch_n=args.batch,
            step_n=args.td_step,
            test_after_epoch=True)

    def run(self):
        self._env_runner.run()


Experiment.register(BCExperiment, 'Behavior clone training')

if __name__ == '__main__':
    Experiment.main()
