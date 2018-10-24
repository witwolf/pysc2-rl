# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.agents.base_agent import BaseAgent
from lib.base import BaseDeepAgent
from utils.network import Utils


class BehaviorClone(BaseDeepAgent, BaseAgent):

    def __init__(self,
                 sess=None,
                 network=None,
                 network_creator=None,
                 script_agents=None,
                 lr=1e-4,
                 discount=0.99,
                 **kwargs):

        if bool(network) == bool(network_creator):
            raise Exception()

        self._sess = sess
        self._network = network
        self._network_creator = network_creator
        self._script_agents = script_agents
        self._lr = lr
        self._discount = discount

        super().__init__(sess, **kwargs)

        self._sess.run(tf.global_variables_initializer())

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
        policy_loss = 0.
        for label, y in zip(self._action_input, self._policies):
            correction = 1e-12
            cross_entropy = -tf.reduce_sum(
                tf.to_float(label) * tf.log(y + correction))
            policy_loss += cross_entropy

        values = Utils.td_value(
            self._reward_input, self._done_input,
            self._value_input, self._discount)
        values = tf.stop_gradient(values)

        value_loss = tf.reduce_mean(tf.square(values - self._value))
        loss = policy_loss + 0.25 * value_loss

        summary = tf.summary.merge([
            tf.summary.scalar('bc/value', tf.reduce_mean(self._value)),
            tf.summary.scalar('bc/policy_loss', policy_loss),
            tf.summary.scalar('bc/value_loss', value_loss),
            tf.summary.scalar('bc/loss', loss)])

        opt = tf.train.RMSPropOptimizer(
            learning_rate=self._lr, decay=0.99, epsilon=1e-5)
        step = tf.Variable(0, trainable=False)
        train_op = layers.optimize_loss(
            loss=policy_loss, optimizer=opt,
            learning_rate=None, global_step=step, clip_gradients=1.0)

        # todo
        self._train_op, self._step, self._summary = train_op, step, summary
        return None

    def step(self, obs=None, state=None, evaluate=False):
        if not evaluate:
            return self._script_agents.step(obs)
        actions = self._sess.run(
            self._actions,
            feed_dict=dict(zip(self._state_input, state)))
        return actions

    def update(self, states, actions,
               next_states, rewards, dones, *args):

        value = self._sess.run(self._value, feed_dict=dict(
            zip(self._state_input, next_states)))

        input_vars = self._state_input + self._action_input + [
            self._reward_input, self._done_input, self._value_input]
        input_vals = states + actions + [rewards, dones, value]
        feed_dict = dict(zip(input_vars, input_vals))

        _, step, summary = self._sess.run(
            [self._train_op, self._step, self._summary],
            feed_dict=feed_dict)
