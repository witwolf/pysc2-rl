# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/19.
#

import logging
import argparse
import tensorflow as tf
import ast


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
        parser.add_argument("--job_name", default='worker')
        parser.add_argument("--ps_hosts", type=str, default=None)
        parser.add_argument("--worker_hosts", type=str, default=None)
        parser.add_argument("--task_index", type=int, default=0)
        parser.add_argument("--train", type=ast.literal_eval, default=True)
        parser.add_argument("--logdir", type=str, default='.')
        parser.add_argument("--save_step", type=int, default=4096)
        parser.add_argument("--restore", type=ast.literal_eval, default=False)

        args, _ = parser.parse_known_args()
        if args.job_name == 'ps':
            Experiment.start_ps(args)
        if args.operation == "list":
            print(Experiment.list())
        elif args.operation == "run":
            Experiment.start(args.name, args)

    @staticmethod
    def start(name, args):
        import sys
        from absl import flags
        flags.FLAGS(sys.argv[:1])
        experiment_class = Experiment.experiments[name]
        logging.warning("Experiment %s running ...", name)
        experiment_class().run(args)
        logging.warning("Experiment %s end.", name)

    @staticmethod
    def start_ps(args):
        if not args.ps_hosts and not args.worker_hosts:
            exit(-1)
        logging.info("starting ps server...")
        cluster = {
            'ps': args.ps_hosts.split(','),
            'worker': args.worker_hosts.split(',')
        }
        logging.warning("Cluster:%s" % cluster)
        cluster_spec = tf.train.ClusterSpec(cluster)
        config = tf.ConfigProto(
            log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        server = tf.train.Server(
            cluster_spec, job_name=args.job_name,
            task_index=args.task_index, config=config)
        server.join()

    def run(self, global_args):
        raise NotImplementedError()


class DistributedExperiment(Experiment):
    def __init__(self):
        super(DistributedExperiment, self).__init__()

    def tf_sess_opts(self, args, config=None):
        if not config:
            config = tf.ConfigProto(
                # log_device_placement=True,
                allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        session_options = {
            'config': config, 'save_dir': args.logdir,
            'save_step': args.save_step, 'restore': args.restore}
        if args.ps_hosts and args.worker_hosts:
            cluster = self.get_cluster_info(args.ps_hosts, args.worker_hosts)
            if len(cluster['worker']) > 1:
                cluster_spec = tf.train.ClusterSpec(cluster)
                server = tf.train.Server(
                    cluster_spec,
                    job_name=args.job_name,
                    task_index=args.task_index,
                    config=config)
                session_options.update({
                    'master': server.target, 'worker_index': args.task_index})
        return session_options

    def tf_device(self, args):
        device = None
        if args.ps_hosts and args.worker_hosts:
            cluster = self.get_cluster_info(args.ps_hosts, args.worker_hosts)
            if len(cluster['worker']) > 1:
                device = tf.train.replica_device_setter(
                    worker_device="/job:worker/task:{}".format(args.task_index),
                    cluster=cluster)
        return device

    def get_cluster_info(self, ps_hosts, worker_hosts):
        return {
            'ps': ps_hosts.split(','),
            'worker': worker_hosts.split(',')
        }
