# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import os
import logging
import time
import tensorflow as tf


class Network(object):
    def __init__(self, network_creator, var_scope,
                 name_scope=None, reuse=False):
        self._var_scope = var_scope
        self._name_scope = name_scope
        self._network_creator = network_creator
        with tf.variable_scope(var_scope, reuse=reuse):
            if name_scope:
                with tf.name_scope(name_scope):
                    inputs, nets = network_creator()
            else:
                inputs, nets = network_creator()
        assert isinstance(nets, dict)
        self._inputs = inputs
        self._net = nets

    def __getitem__(self, item):
        return self._net.get(item)

    def __call__(self, inputs):
        pass

    @property
    def inputs(self):
        return self._inputs

    @property
    def var_scope(self):
        return self._var_scope

    @property
    def name_scope(self):
        return self._name_scope


class NetworkUpdater(object):
    def __init__(self):
        super(NetworkUpdater, self).__init__()
        self.name = None

    def declare_update(self):
        raise NotImplementedError()

    def update(self, sess, *args, **kwargs):
        raise NotImplementedError()


class BaseDeepAgent(object):
    def __init__(self, **kwargs):
        self._sess = None
        self._network = self.init_network(**kwargs)
        self._updater = self.init_updater(**kwargs)

    def init_network(self, **kwargs):
        raise NotImplementedError()

    def init_updater(self, **kwargs):
        raise NotImplementedError()

    def create_session(self, config=None,
                       master='', task_index=0,
                       save_dir=None,
                       save_step=128,
                       restore=False,
                       restore_var_list=None):
        sess = MonitoredTrainingSession(
            master,
            is_chief=(task_index == 0),
            checkpoint_dir=save_dir,
            save_checkpoint_steps=save_step,
            restore=restore,
            restore_var_list=restore_var_list,
            config=config)
        self._sess = sess
        return sess

    def step(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def set_sess(self, sess):
        self._sess = sess

    @property
    def network(self):
        return self._network

    @property
    def updater(self):
        return self._updater


def MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        restore=False,
        restore_var_list=None,
        save_checkpoint_steps=None,
        config=None):
    if not is_chief:
        session_creator = tf.train.WorkerSessionCreator(
            master=master, config=config)
        return tf.train.MonitoredSession(
            session_creator=session_creator)

    scaffold = tf.train.Scaffold()
    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    hooks = [
        tf.train.StopAtStepHook(last_step=1e8)]

    if is_chief and save_checkpoint_steps and save_checkpoint_steps > 0:
        hooks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir, scaffold=scaffold,
            save_steps=save_checkpoint_steps))

    session_creator = tf.train.ChiefSessionCreator(
        checkpoint_dir=checkpoint_dir if restore else None,
        scaffold=scaffold,
        master=master,
        config=config)

    sess = tf.train.MonitoredSession(
        session_creator=session_creator, hooks=hooks)
    logging.warning("session created")
    return sess
