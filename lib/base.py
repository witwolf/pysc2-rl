# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import os
import logging
import time
import tensorflow as tf


class Network(object):
    def __init__(self, network_creator, var_scope, name_scope=None, reuse=False):
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
        self._global_step = tf.train.get_or_create_global_step()

    def init_network(self, **kwargs):
        raise NotImplementedError()

    def init_updater(self, **kwargs):
        raise NotImplementedError()

    def create_session(self, config=None,
                       master='', worker_index=0,
                       save_dir=None,
                       save_step=128,
                       restore=False,
                       restore_var_list=None):
        sess = MonitoredTrainingSession(
            master, is_chief=(worker_index == 0),
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
        step = self._sess.run(self._global_step)
        logging.info("Step:%d", step)

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


class RestoreVariablesHook(tf.train.SessionRunHook):

    def __init__(self,
                 checkpoint_dir,
                 var_list=None):
        super(RestoreVariablesHook, self).__init__()
        self._dir, self._var_list = (checkpoint_dir, var_list)

        self._saver = tf.train.Saver(var_list=self._var_list)

    def after_create_session(self, session, coord):
        super(RestoreVariablesHook, self).after_create_session(session, coord)
        ckpt = tf.train.get_checkpoint_state(self._dir)
        if not (ckpt and ckpt.model_checkpoint_path):
            logging.warning("No checkpoint in %s" % self._dir)
            return
        logging.info("Restore checkpoint: %s", ckpt.model_checkpoint_path)
        self._saver.restore(session, ckpt.model_checkpoint_path)
        self._saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)


def MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=None,
        restore=False,
        restore_var_list=None,
        save_checkpoint_steps=None,
        config=None, stop_grace_period_secs=120):
    scaffold = scaffold or tf.train.Scaffold()

    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    if not is_chief:
        session_creator = tf.train.WorkerSessionCreator(
            scaffold=scaffold, master=master, config=config)
        return tf.train.MonitoredSession(
            session_creator=session_creator, hooks=[],
            stop_grace_period_secs=stop_grace_period_secs)

    all_hooks = []

    if restore:
        all_hooks.append(RestoreVariablesHook(
            checkpoint_dir, var_list=restore_var_list))
        all_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES)
        if restore_var_list:
            missing_vars = [v for v in all_vars if not (v in restore_var_list)]
            logging.warning("Not restore missing vars, count:%d", len(missing_vars))

    if save_checkpoint_steps and save_checkpoint_steps > 0:
        all_hooks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir, save_steps=save_checkpoint_steps,
            scaffold=scaffold))

    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=None,
        master=master,
        config=config)

    return tf.train.MonitoredSession(
        session_creator=session_creator, hooks=all_hooks,
        stop_grace_period_secs=stop_grace_period_secs)
