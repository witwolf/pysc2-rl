# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import argparse
from experiments.experiment import Experiment
from environment.parallel_env import ParallelEnvs
from environment.env_runner import EnvRunner
from lib.adapter import DefaultObservationAdapter as ObservationAdapter
from lib.adapter import DefaultActionAdapter as ActionAdapter
from algorithm.a2c import A2C
from lib.config import Config

from experiments.exp_bc import network_creator


# a2c train (imitation)
class A2CExperiment(Experiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="CollectMineralShards")
        parser.add_argument("--env_num", type=int, default=32, help='env parallel run')
        parser.add_argument("--epoch", type=int, default=1024),
        parser.add_argument("--batch", type=int, default=256),

        args, _ = parser.parse_known_args()
        print(args)

        config = Config()
        agent = A2C()
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


Experiment.register(A2CExperiment, "A2C training")

if __name__ == '__main__':
    Experiment.main()
