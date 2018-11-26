# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/26.
#


from algorithm.a2c import A2C


class MLSH(A2C):

    def __init__(self,
                 sub_policies=None,
                 K=1,
                 network=None,
                 network_creator=None,
                 td_step=16,
                 lr=1e-4,
                 v_coef=0.25,
                 ent_coef=1e-3,
                 discount=0.99,
                 **kwargs):
        self._K = K
        self._sub_policies = sub_policies
        super().__init__(
            network,
            network_creator,
            td_step,
            lr, v_coef,
            ent_coef,
            discount,
            **kwargs)

    def step(self, state=None, **kwargs):
        # TODO
        return super().step(state, **kwargs)

    def update(self, states, actions, next_states, rewards, dones, *args):
        # TODO

        return super().update(states, actions, next_states, rewards, dones, *args)
