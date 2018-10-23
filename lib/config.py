# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/10/18.
#


from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS, TYPES

NON_SPATIAL_FEATURES = dict(
    player=(11,),
    game_loop=(1,),
    score_cumulative=(13,),
    available_actions=(len(FUNCTIONS),),
    single_select=(1, 7),
    multi_select=(0, 7),
    cargo=(0, 7),
    cargo_slots_available=(1,),
    build_queue=(0, 7),
    control_groups=(10, 2), )

DEFAULT_NON_SPATIAL_FEATURES = [
    'player',
    'available_actions']

SPATIAL_ARG_TYPES = {'screen', 'screen2', 'minimap'}


class Config:
    def __init__(self, screen_size=64,
                 minimap_size=64,
                 screen_features=features.SCREEN_FEATURES._fields,
                 minimap_features=features.MINIMAP_FEATURES._fields,
                 non_spatial_features=DEFAULT_NON_SPATIAL_FEATURES,
                 available_actions=FUNCTIONS._func_list,
                 ):
        assert screen_size == minimap_size
        self._size = screen_size

        # screen feature
        self._screen_features = screen_features
        self._screen_dims = self._dims(self._features(
            features.SCREEN_FEATURES, screen_features))
        self._screen_feature_indexes = self._indexes(
            screen_features, features.SCREEN_FEATURES._fields)

        # minimap feature
        self._minimap_features = minimap_features
        self._minimap_dims = self._dims(self._features(
            features.MINIMAP_FEATURES, minimap_features))
        self._minimap_feature_indexes = self._indexes(
            minimap_features, features.MINIMAP_FEATURES._fields)

        # non spatial feature
        self._non_spatial_features = non_spatial_features
        self._non_spatial_dims = [
            NON_SPATIAL_FEATURES[f] for f in self._non_spatial_features]
        self._non_spatial_index_table = self._index_table(non_spatial_features)

        # available actions
        # todo , replace it with macros
        self._actions = available_actions
        self._action_indexes = [
            int(action.id) for action in available_actions]
        self._action_index_table = self._index_table(self._action_indexes)
        action_args = set()
        for action in available_actions:
            for action_arg in action.args:
                action_args.add(action_arg.name)
        self._action_args = action_args
        self._action_args_index_table = self._index_table(action_args)

        # policy dims
        policy_dims = [(len(available_actions), False)]
        for act_arg in action_args:
            is_spatial = False
            dim = getattr(TYPES, act_arg).sizes[0]
            if act_arg in SPATIAL_ARG_TYPES:
                dim = screen_size * screen_size
                is_spatial = True
            policy_dims.append((dim, is_spatial))
        self._policy_dims = policy_dims

    def _features(self, feature_collection, keys):
        return [getattr(feature_collection, key) for key in keys]

    def _indexes(self, fields, named_tuple):
        return [named_tuple.index(key) for key in fields]

    def _index_table(self, values):
        return {value: i for i, value in enumerate(values)}

    def _dims(self, _features):
        dims = []
        for f in _features:
            dim = 1
            if f.type == features.FeatureType.CATEGORICAL:
                dim = f.scale
            dims.append(dim)
        return dims

    @property
    def screen_dims(self):
        return self._screen_dims

    @property
    def minimap_dims(self):
        return self._minimap_dims

    @property
    def non_spatial_dims(self):
        return self._non_spatial_dims

    @property
    def policy_dims(self):
        return self._policy_dims


if __name__ == '__main__':
    c = Config()
