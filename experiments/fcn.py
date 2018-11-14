# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/23.
#

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


class FCNNetwork():
    @staticmethod
    def build_fcn(config):
        size = config._size
        with tf.variable_scope('screen'):
            screen_input, screen = FCNNetwork._conv_channels(
                size, size, config.screen_dims, 'screen')
        with tf.variable_scope('minimap'):
            minimap_input, minimap = FCNNetwork._conv_channels(
                size, size, config.minimap_dims, 'minimap')
        with tf.variable_scope('nonspatial'):
            non_spatial_input, non_spatial = FCNNetwork._non_spatial(
                size, config.non_spatial_dims,
                config._non_spatial_index_table)

        state = tf.concat([screen, minimap, non_spatial], axis=3)
        fc = layers.fully_connected(
            layers.flatten(state), num_outputs=256, scope='fc')
        value = layers.fully_connected(
            fc, num_outputs=1, activation_fn=None, scope='value')
        value = tf.squeeze(value, axis=1)

        policy_dims = config.policy_dims
        policies = []
        act_dim, _ = policy_dims[0]
        act_policy = layers.fully_connected(
            fc, num_outputs=act_dim, scope='act_policy',
            activation_fn=tf.nn.softmax)
        if 'available_actions' in config._non_spatial_index_table:
            available_actions = non_spatial_input[
                config._non_spatial_index_table['available_actions']]
            act_policy = FCNNetwork.mask_probs(act_policy, available_actions)
        policies.append(act_policy)

        for arg_dim, is_spatial in policy_dims[1:]:
            if is_spatial:
                arg_policy = layers.conv2d(
                    state, num_outputs=1, kernel_size=1,
                    activation_fn=None, data_format="NHWC")
                arg_policy = tf.nn.softmax(layers.flatten(arg_policy))
            else:
                arg_policy = layers.fully_connected(
                    fc, num_outputs=arg_dim, activation_fn=tf.nn.softmax)
            policies.append(arg_policy)

        inputs = [screen_input, minimap_input] + non_spatial_input
        actions = [FCNNetwork._sample(p) for p in policies]
        return value, policies, actions, inputs

    @staticmethod
    def _conv_channels(h, w, cs, var_name=''):
        input_ph = tf.placeholder(tf.float32, [None, h, w, len(cs)])
        channel_arr = tf.split(input_ph, input_ph.shape[-1], axis=-1)
        for i, c in enumerate(cs):
            if c == 1:
                channel_arr[i] = tf.log(channel_arr[i] + 1.0)
            else:
                channel_arr[i] = FCNNetwork._embedding(
                    channel_arr[i], c, int(round(np.log2(c))),
                    var_name + '_' + str(i))
        channels = tf.concat(channel_arr, axis=3)
        conv1 = layers.conv2d(
            channels, num_outputs=16, kernel_size=5, data_format='NHWC')
        conv2 = layers.conv2d(
            conv1, num_outputs=32, kernel_size=3, data_format='NHWC')
        return input_ph, conv2

    @staticmethod
    def _non_spatial(size, dims, idx):
        input_phs = [tf.placeholder(
            tf.float32, [None, *dim]) for dim in dims]
        # todo ,more information
        channel = FCNNetwork._broadcast(
            tf.log(input_phs[idx['player']] + 1.0), size)
        return input_phs, channel

    @staticmethod
    def _embedding(input, label_size, embed_layer_nums, var_name):
        shape = input.shape
        h, w = shape[1], shape[2]
        labels = tf.to_int32(layers.flatten(input))
        embedding_layers = []
        for i in range(embed_layer_nums):
            params_variable = tf.get_variable(
                name=var_name + '_' + str(i), shape=[label_size],
                dtype=tf.float32)
            embedding_layer = tf.nn.embedding_lookup(
                params_variable, labels)
            embedding_layer = tf.reshape(
                embedding_layer, shape=(-1, h, w))
            embedding_layer = tf.expand_dims(embedding_layer, 3)
            embedding_layers.append(embedding_layer)
        return tf.concat(embedding_layers, axis=3)

    @staticmethod
    def _broadcast(input, size):
        t0 = tf.expand_dims(input, 1)
        t1 = tf.expand_dims(t0, 1)
        return tf.tile(t1, [1, size, size, 1])

    @staticmethod
    def _sample(probs):
        u = tf.random_uniform(tf.shape(probs))
        clipped_u = tf.clip_by_value(u, 1e-12, 1.0)
        return tf.argmax(tf.log(clipped_u) / probs, axis=1)

    @staticmethod
    def mask_probs(probs, mask):
        masked = probs * mask
        masked_sum = tf.reduce_sum(masked, axis=1, keep_dims=True)
        return masked / tf.clip_by_value(masked_sum, 1e-12, 1.0)
