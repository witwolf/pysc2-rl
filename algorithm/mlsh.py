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
                 summary_family='mlsh',
                 **kwargs):
        self._K = K
        self._sub_policies = sub_policies
        self._agents = [super()] + sub_policies
        super().__init__(
            network,
            network_creator,
            td_step,
            lr, v_coef,
            ent_coef,
            discount,
            summary_family,
            **kwargs)

    def step(self, state=None, policy_index=0, **kwargs):
        return self._agents[policy_index].step(state)

    def update(self, states, actions,
               next_states, rewards, dones, policy_index=0, *args):
        return self._agents[policy_index].update(
            states, actions, next_states, rewards, dones, *args)
