# Copyright (c) 2018 horizon robotics. All rights reserved.
# Created by yingxiang.hong@horizon.ai on 2018/11/27.
#


class Utility(object):

    @staticmethod
    def select(_list, _indices):
        return [_list[i] for i in _indices]
