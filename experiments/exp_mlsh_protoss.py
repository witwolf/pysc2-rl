# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/26.
#


import sys
import argparse
import ast

sys.path.append('.')
import tensorflow as tf

from experiments.experiment import DistributedExperiment
from experiments.experiment import Experiment
from experiments.exp_a2c_protoss import network_creator
from lib.config import Config
from lib.protoss_macro import PROTOSS_MACROS
from environment.parallel_env import ParallelEnvs
from environment.macro_env import default_macro_env_maker
from environment.env_runner import EnvRunner2
from algorithm.a2c import A2C
from algorithm.mlsh import MLSH
from lib.protoss_adapter import ProtossObservationAdapter as ObservationAdapter
from lib.protoss_adapter import ProtossMacroAdapter as MacroAdapter
from utils.utility import Utility


class MLSHProtossExperiment(DistributedExperiment):
    def __init__(self):
        super().__init__()
        # TODO
        parser = argparse.ArgumentParser()
        parser.add_argument("--map_name", type=str, default="MoveToBeacon")
        parser.add_argument("--env_num", type=int, default=32)
        parser.add_argument("--epoch", type=int, default=8096),
        parser.add_argument("--td_step", type=int, default=16, help='td(n)')
        parser.add_argument("--logdir", type=str, default='log/a2c')
        parser.add_argument("--v_coef", type=float, default=0.25)
        parser.add_argument("--ent_coef", type=float, default=1e-3)
        parser.add_argument("--lr", type=float, default=7e-4)
        parser.add_argument("--visualize", type=ast.literal_eval, default=False)
        parser.add_argument("--difficulty", type=int, default=1)
        parser.add_argument("--K", type=int, default=8)
        parser.add_argument("--mode", type=str, default='multi_thread')
        args, _ = parser.parse_known_args()
        self._local_args = args

    def run(self, global_args):
        local_args = self._local_args
        non_spatial_features = [
            'timestep_information',
            'available_actions']

        operate_config = Config(
            screen_features=[],
            minimap_features=[],
            available_actions=Utility.select(PROTOSS_MACROS._macro_list, [
                0, 1, 2, 3, 4, 5, 6, 7, 8]),
            non_spatial_features=non_spatial_features,
            non_spatial_feature_dims=dict(
                timestep_information=(19,),
                available_actions=(9,)))
        combat_config = Config(
            screen_features=[],
            minimap_features=[],
            available_actions=Utility.select(PROTOSS_MACROS._macro_list, [9, 10]),
            non_spatial_features=non_spatial_features,
            non_spatial_feature_dims=dict(
                timestep_information=(19,),
                available_actions=(2,)))
        config = Config(
            screen_features=[],
            minimap_features=[],
            available_actions=[0, 1],
            non_spatial_features=[
                'timestep_information'],
            non_spatial_feature_dims=dict(
                timestep_information=(19,)))
        env_args = [{'map_name': "Simple64",
                     'difficulty': local_args.difficulty}
                    for _ in range(local_args.env_num)]
        env_args[0]['visualize'] = local_args.visualize

        with tf.device(self.tf_device(global_args)):
            operate = A2C(
                network_creator=network_creator(operate_config, 'operate'),
                lr=local_args.lr,
                td_step=local_args.K,
                ent_coef=local_args.ent_coef,
                v_coef=local_args.v_coef, summary_family='operate')
            combat = A2C(
                network_creator=network_creator(combat_config, 'combat'),
                lr=local_args.lr,
                td_step=local_args.K,
                ent_coef=local_args.ent_coef,
                v_coef=local_args.v_coef, summary_family='combat')
            agent = MLSH(
                sub_policies=[operate, combat],
                network_creator=network_creator(config, 'main_policy'),
                lr=local_args.lr,
                td_step=local_args.td_step,
                ent_coef=local_args.ent_coef,
                v_coef=local_args.v_coef, summary_family='main')
        with agent.create_session(**self.tf_sess_opts(global_args)) as sess:
            operate.set_sess(sess)
            combat.set_sess(sess)
            env = ParallelEnvs.new(
                mode=local_args.mode,
                env_makers=default_macro_env_maker,
                env_num=local_args.env_num,
                env_args=env_args)

            main_policy_obs_adpt = ObservationAdapter(config)
            operate_obs_adapter = ObservationAdapter(operate_config)
            combat_obs_adapter = ObservationAdapter(combat_config)

            operate_act_adapter = MacroAdapter(operate_config)
            combat_act_adapter = MacroAdapter(combat_config)

            env_runner = EnvRunner2(
                agent=agent, env=env,
                K=local_args.K,
                main_policy_obs_adpt=main_policy_obs_adpt,
                sub_policy_obs_adpts=[
                    operate_obs_adapter, combat_obs_adapter],
                sub_policy_act_adpts=[
                    operate_act_adapter, combat_act_adapter],
                train=global_args.train,
                epoch_n=local_args.epoch,
                step_n=local_args.td_step,
                logdir=local_args.logdir)
            env_runner.run()


Experiment.register(MLSHProtossExperiment, "MLSH training protoss")

if __name__ == '__main__':
    Experiment.main()
