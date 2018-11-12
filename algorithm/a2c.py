# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#

import tensorflow as tf
from lib.base import BaseDeepAgent
from pysc2.agents.base_agent import BaseAgent as BaseSC2Agent
from tensorflow.contrib import layers
from utils.network import Utils


class A2C(BaseDeepAgent, BaseSC2Agent):

    def __init__(self,
                 network=None,
                 network_creator=None,
                 td_step=16,
                 lr=1e-4,
                 v_coef=0.25,
                 ent_coef=1e-3,
                 discount=0.99,
                 **kwargs):
        if bool(network) == bool(network_creator):
            raise Exception()

        self._network = network
        self._network_creator = network_creator
        self._td_step = td_step
        self._discount = discount
        self._lr = lr
        self._v_coef = v_coef
        self._ent_coef = ent_coef

        super().__init__(**kwargs)

    def init_network(self, **kwargs):
        if not self._network:
            self._network = self._network_creator(**kwargs)
        inputs = self._network.inputs
        self._state_input = inputs['state']
        self._action_input = inputs['action']
        self._reward_input = inputs['reward']
        self._done_input = inputs['done']
        self._value_input = inputs['value']
        self._value = self._network['value']
        self._policies = self._network['policies']
        self._actions = self._network['actions']

    def init_updater(self, **kwargs):
        # advantage
        target_value = Utils.tf_td_value(
            self._reward_input, self._done_input,
            self._value_input, self._discount)
        target_value = tf.stop_gradient(target_value)
        advantage = tf.stop_gradient(target_value - self._value)

        # log pi
        log_pi = 0.0
        for a, p in zip(self._action_input, self._policies):
            r = tf.range(tf.shape(p)[0])
            indices = tf.stack([r, a], axis=1)
            # select index 0 in unused arg_policy
            selected_p = tf.gather_nd(p, indices)
            log_pi += tf.log(tf.clip_by_value(selected_p, 1e-12, 1.0))

        # entropy
        entropy = 0.0
        for p in self._policies:
            entropy += -tf.reduce_sum(
                p * tf.log(tf.clip_by_value(p, 1e-12, 1.0)), axis=-1)
        entropy = tf.reduce_mean(entropy)

        # loss
        policy_loss = -tf.reduce_mean(log_pi * advantage)
        entropy_loss = -self._ent_coef * entropy
        value_loss = self._v_coef * tf.reduce_mean(
            tf.square(target_value - self._value))
        loss = policy_loss + entropy_loss + value_loss
        summary = tf.summary.merge([
            tf.summary.scalar('a2c/advantage', tf.reduce_mean(advantage)),
            tf.summary.scalar('a2c/log_pi', tf.reduce_mean(log_pi)),
            tf.summary.scalar('a2c/value', tf.reduce_mean(self._value)),
            tf.summary.scalar('a2c/target_value', tf.reduce_mean(target_value)),
            tf.summary.scalar('a2c/policy_loss', policy_loss),
            tf.summary.scalar('a2c/entropy', entropy),
            tf.summary.scalar('a2c/entropy_loss', entropy_loss),
            tf.summary.scalar('a2c/value_loss', value_loss),
            tf.summary.scalar('a2c/loss', loss)])
        step = tf.train.get_or_create_global_step()
        opt = tf.train.RMSPropOptimizer(
            learning_rate=self._lr,
            decay=0.99, epsilon=1e-5)
        train_op = layers.optimize_loss(
            loss=loss, optimizer=opt,
            learning_rate=None, global_step=step,
            clip_gradients=1.0)
        self._train_op, self._step, self._summary = (
            train_op, step, summary)
        return None

    def step(self, state=None, **kwargs):
        feed_dict = dict(zip(self._state_input, state))
        return self._sess.run(self._actions, feed_dict=feed_dict)

    def update(self, states, actions,
               next_states, rewards, dones, *args):
        super().update()
        begin = -len(dones) // self._td_step
        last_state = [e[begin:] for e in next_states]
        value = self._sess.run(self._value, feed_dict=dict(
            zip(self._state_input, last_state)))
        input_vars = self._state_input + self._action_input + [
            self._reward_input, self._done_input, self._value_input]
        input_vals = states + actions + [rewards, dones, value]
        feed_dict = dict(zip(input_vars, input_vals))
        _, step, summary = self._sess.run(
            [self._train_op, self._step, self._summary],
            feed_dict=feed_dict)

        return summary, step
