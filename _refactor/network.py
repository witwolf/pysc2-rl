# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


def build_fcn(config):
    sz = config._size
    screen_dims = config.screen_dims
    minimap_dims = config.minimap_dims
    non_spatial_dims = config.non_spatial_dims

    screen, screen_input_ph = cnn_block(sz, sz, screen_dims, 'screen')
    minimap, minimap_input_ph = cnn_block(sz, sz, minimap_dims, 'minimap')
    non_spatial, non_spatial_input_phs = non_spatial_block(sz, non_spatial_dims, config.non_spatial_idx)
    state = tf.concat([screen, minimap, non_spatial], axis=3)
    fc = layers.fully_connected(layers.flatten(state), num_outputs=256)

    value = tf.squeeze(layers.fully_connected(fc, num_outputs=1, activation_fn=None), axis=1)

    policies = []
    policy_dims = config.policy_dims()

    act_dim, _ = policy_dims[0]
    act_policy = layers.fully_connected(fc, num_outputs=act_dim, activation_fn=tf.nn.softmax)
    available_actions_ph = non_spatial_input_phs[config.non_spatial_idx['available_actions']]
    act_policy = mask_probs(act_policy, available_actions_ph)
    policies.append(act_policy)

    act_index = tf.stop_gradient(tf.to_float(tf.argmax(act_policy, axis=1)))
    act_index = tf.reshape(act_index, shape=(-1, 1))
    act_channel = broadcast(tf.log(act_index + 1.0), sz)
    act_arg_input = tf.concat([state, act_channel], axis=3)
    act_arg_fc = layers.fully_connected(layers.flatten(act_arg_input), num_outputs=256)

    for arg_dim, is_spatial in policy_dims[1:]:
        if is_spatial:
            arg_policy = layers.conv2d(act_arg_input, num_outputs=1, kernel_size=1, activation_fn=None,
                                       data_format='NHWC')
            arg_policy = tf.nn.softmax(layers.flatten(arg_policy))
        else:
            arg_policy = layers.fully_connected(act_arg_fc, num_outputs=arg_dim, activation_fn=tf.nn.softmax)
        policies.append(arg_policy)

    return value, policies, [screen_input_ph, minimap_input_ph] + non_spatial_input_phs


def cnn_block(height, weight, encoding_sizes, scope=''):
    input_ph = tf.placeholder(tf.float32, [None, height, weight, len(encoding_sizes)])
    blocks = tf.split(input_ph, len(encoding_sizes), axis=-1)
    for i, encoding_size in enumerate(encoding_sizes):
        if encoding_size == 1:
            blocks[i] = tf.log(blocks[i] + 1.0)
        else:
            embed_layer_nums = int(round(np.log2(encoding_size)))
            # blocks[i] = label_encoding_onehot_embedding(blocks[i], encoding_size, embed_layer_nums)
            var_name = '%s-%d' % (scope, i)
            blocks[i] = label_encoding_embedding_lookup(blocks[i], encoding_size, embed_layer_nums, var_name)
    block = tf.concat(blocks, axis=3)
    conv1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format='NHWC')
    conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=3, data_format='NHWC')
    return conv2, input_ph


def non_spatial_block(sz, dims, idx):
    input_phs = [tf.placeholder(tf.float32, [None, *dim]) for dim in dims]
    block = broadcast(tf.log(input_phs[idx['player']] + 1.0), sz)
    # TODO
    # ignore other non spatial information
    return block, input_phs


def label_encoding_onehot_embedding(input, encoding_size, embed_layer_nums):
    labels = tf.to_int32(tf.squeeze(input, axis=3))
    one_hot = tf.one_hot(labels, encoding_size, axis=3)
    return layers.conv2d(one_hot, num_outputs=embed_layer_nums, kernel_size=1, data_format="NHWC")


def label_encoding_embedding_lookup(input, label_size, embed_layer_nums, name):
    input_shape = input.shape
    assert len(input_shape) == 4 and input_shape[-1] == 1
    input_height, input_weight = input_shape[1], input_shape[2]
    labels = tf.to_int32(layers.flatten(input))
    embedding_layers = []
    for i in range(embed_layer_nums):
        var_name = '%s-%d' % (name, i)
        embedding_params = tf.get_variable(var_name, shape=[label_size])
        embedding_layer = tf.nn.embedding_lookup(embedding_params, labels)
        embedding_layer = tf.reshape(embedding_layer, shape=(-1, input_height, input_weight))
        embedding_layer = tf.expand_dims(embedding_layer, 3)
        embedding_layers.append(embedding_layer)
    return tf.concat(embedding_layers, axis=3)


def broadcast(input, size):
    t0 = tf.expand_dims(input, 1)
    t1 = tf.expand_dims(t0, 1)
    return tf.tile(t1, [1, size, size, 1])


def mask_probs(probs, mask):
    # for tests
    avaliable_actions = np.zeros(shape=(549), dtype=np.float32)
    avaliable_actions[0] = avaliable_actions[2] = avaliable_actions[331] = 1

    mask = mask * avaliable_actions
    masked = probs * mask
    correction = tf.cast(tf.reduce_sum(masked, axis=-1, keep_dims=True) < 1e-3, dtype=tf.float32) \
                 * (1.0 / (tf.reduce_sum(mask, axis=-1, keep_dims=True) + 1e-12)) \
                 * mask
    masked += correction
    return masked / tf.clip_by_value(tf.reduce_sum(masked, axis=1, keep_dims=True), 1e-12, 1.0)


def sample(probs):
    u = tf.random_uniform(tf.shape(probs))
    return tf.argmax(tf.log(u) / probs, axis=1)


def select(acts, policy):
    r = tf.range(tf.shape(policy)[0])
    return tf.gather_nd(policy, tf.stack([r, acts], axis=1))


def clip_log(probs):
    return tf.log(tf.clip_by_value(probs, 1e-12, 1.0))
