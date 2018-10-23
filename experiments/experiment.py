# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import argparse


class Experiment(object):
    experiments = {}

    def __init__(self):
        pass

    @staticmethod
    def register(subclass, desc=None):
        e = subclass
        if desc is not None:
            e.desc = desc
        name = subclass.__name__
        if name in Experiment.experiments:
            raise KeyError('experiment :%s already exists' % name)
        Experiment.experiments[name] = e

    @staticmethod
    def list():
        msg = ""
        for k in Experiment.experiments:
            v = Experiment.experiments[k]
            msg += "{:20}".format(str(k)) + ":\t" + str(v.desc) + "\n"
        return msg

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("operation", default="list", choices=['list', 'run'])
        parser.add_argument("--name")
        args, _ = parser.parse_known_args()
        if args.operation == "list":
            print(Experiment.list())
        elif args.operation == "run":
            Experiment.start(args.name)

    @staticmethod
    def start(name):
        experiment_class = Experiment.experiments[name]
        logging.warning("Experiment %s running ...", name)
        experiment_class().run()
        logging.warning("Experiment %s end.", name)

    def run(self):
        raise NotImplementedError()
