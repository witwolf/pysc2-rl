# -*- coding: utf-8 -*-

import tensorflow as tf
from _refactor.network import build_fcn
from _refactor.network import sample, select, clip_log
from tensorflow.contrib import layers


class Model(object):
    def __init__(self, config):
        self.config = config
        args = self.config.args
        self.lr = args.lr
        self.logdir = args.logdir + '/' + args.map
        self.snapshot = args.snapshot + '/' + args.map
        self.build_model()

    def build_model(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Graph().as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    # log_device_placement=True,
                                                    allow_soft_placement=True))
            self.sess = sess
            self.value, self.policies, self.state_inputs = build_fcn(self.config)
            self.action_inputs, self.action_label_inputs, self.return_inputs = self.input_phs()
            self.action = [sample(policy) for policy in self.policies]

            self.imitation_train_op, self.imiation_step, self.imitation_sumarry = self._imitation_train()
            self.a2c_train_op, self.a2c_step, self.a2c_summary = self._a2c_train()

            self.global_step = tf.Variable(0, trainable=False)

            self.summary_writer = tf.summary.FileWriter(self.logdir, graph=None)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            if self.config.args.restore:
                checkpoint = self.snapshot
                print('restore checkpoint:', checkpoint)
                self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))

    def input_phs(self):
        action_inputs = [
            tf.placeholder(dtype=tf.int32, shape=p.shape)
            for p in self.policies
        ]

        action_label_inputs = [
            tf.placeholder(dtype=tf.int32, shape=[None])
            for _ in self.policies
        ]

        return_inputs = tf.placeholder(tf.float32, [None])
        return action_inputs, action_label_inputs, return_inputs

    def _imitation_train(self):
        policy_loss = 0.
        for label, y in zip(self.action_inputs, self.policies):
            # mask unavaliable actions
            # todo with filter or other means
            correction = 1e-12
            cross_entropy = -tf.reduce_sum(tf.to_float(label) * tf.log(y + correction))
            policy_loss += cross_entropy

        returns = self.return_inputs
        value_loss = tf.reduce_mean(tf.square(returns - self.value))

        loss = policy_loss + 0.25 * value_loss

        summary = tf.summary.merge([
            tf.summary.scalar('imitation/value', tf.reduce_mean(self.value)),
            tf.summary.scalar('imitation/policy_loss', policy_loss),
            tf.summary.scalar('imitation/value_loss', value_loss),
            tf.summary.scalar('imitation/loss', loss)])

        opt = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-5)
        step = tf.Variable(0, trainable=False)
        train_op = layers.optimize_loss(loss=policy_loss, optimizer=opt, learning_rate=None,
                                        global_step=step,
                                        clip_gradients=1.0)
        return train_op, step, summary

    def _a2c_train(self):
        returns = self.return_inputs
        advance = tf.stop_gradient(returns - self.value)
        logli = sum([clip_log(select(a, p)) for a, p in zip(self.action_label_inputs, self.policies)])
        entropy = sum([-tf.reduce_sum(p * clip_log(p), axis=-1) for p in self.policies])

        policy_loss = -tf.reduce_mean(logli * advance)
        entropy_loss = -1e-3 * tf.reduce_mean(entropy)
        value_loss = 0.25 * tf.reduce_mean(tf.square(returns - self.value))
        loss = policy_loss + entropy_loss + value_loss
        summary = tf.summary.merge([
            tf.summary.scalar('reinforce/value', tf.reduce_mean(self.value)),
            tf.summary.scalar('reinforce/policy_loss', policy_loss),
            tf.summary.scalar('reinforce/entropy_loss', entropy_loss),
            tf.summary.scalar('reinforce/value_loss', value_loss),
            tf.summary.scalar('reinforce/loss', loss)
        ])
        opt = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-5)
        step = tf.Variable(0, trainable=False)
        train_op = layers.optimize_loss(loss=loss, optimizer=opt, learning_rate=None,
                                        global_step=step,
                                        clip_gradients=1.0)

        return train_op, step, summary

    def imitation_train(self, states, actions, next_states, rewards, dones):
        returns = rewards + 0.99 * (1 - dones) * self.get_value(next_states)
        feed_dict = dict(
            zip(self.state_inputs + self.action_inputs + [self.return_inputs],
                states + actions + [returns]))
        _, summary, step, _ = self.sess.run([self.imitation_train_op,
                                             self.imitation_sumarry,
                                             self.imiation_step,
                                             self.global_step],
                                            feed_dict)

        self.summary_writer.add_summary(summary, step)

    def get_value(self, states):
        value = self.sess.run(self.value, feed_dict=dict(zip(self.state_inputs, states)))
        return value

    def a2c_train(self, states, actions, returns, *args):
        feed_dict = dict(zip(self.state_inputs + self.action_label_inputs + [self.return_inputs],
                             states + actions + [returns]))
        _, summary, step, _ = self.sess.run(
            [self.a2c_train_op,
             self.a2c_summary,
             self.a2c_step,
             self.global_step], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, step)

    def step(self, obs):
        feed_dict = dict(zip(self.state_inputs, obs))
        actions = self.sess.run(self.action, feed_dict=feed_dict)
        return actions

    def summarize(self, step, **kwargs):
        summary = tf.Summary()
        for k, v in kwargs.items():
            summary.value.add(tag=k, simple_value=v)
        self.summary_writer.add_summary(summary, step)

    def save(self):
        import os
        if not os.path.isdir(self.snapshot):
            os.makedirs(self.snapshot)

        self.saver.save(self.sess, self.snapshot + '/model', global_step=self.global_step)
