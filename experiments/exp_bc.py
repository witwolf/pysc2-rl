# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


import argparse
from experiments.experiment import Experiment


# behavior clone train (imitation)
class BCExperiment(Experiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--env_num", default=32, help='env parallel run')
        parser.add_argument("--td_step", default=16, help='td(n)')

        args, _ = parser.parse_known_args()
        print(args)

    def run(self):
        pass


Experiment.register(BCExperiment, 'Behavior clone training')

if __name__ == '__main__':
    Experiment.main()
