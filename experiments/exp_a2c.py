# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import argparse
from experiments.experiment import Experiment


# a2c train (imitation)
class A2CExperiment(Experiment):

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument("--env_num", default=32, help='env parallel run')
        parser.add_argument("--epoch", default=0)
        parser.add_argument("--batch", default=256, help="batches for every epoch")
        parser.add_argument("--td_step", default=16, help='td(n)')

        args, _ = parser.parse_known_args()
        print(args)

    def run(self):
        pass


Experiment.register(A2CExperiment, "A2C training")

if __name__ == '__main__':
    Experiment.main()
