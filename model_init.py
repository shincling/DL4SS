# -*- coding: utf-8 -*-
"""
    Initialize the parameters of model
"""
import config
import prepare_data
__author__ = 'jacoxu'


class ModelInit(object):
    def __init__(self, _log_file):
        # pre-process data
        print("Start to preprare data set")
        self.train_gen = prepare_data.get_feature(config.TRAIN_LIST, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                                                  batch_size=config.BATCH_SIZE)
        self.valid_gen = prepare_data.get_feature(config.VALID_LIST, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                                                  batch_size=config.BATCH_SIZE)
        # inp_shape (None, feature_dim), 其中，batch_size 被省略, None是MAX_STEP, Layer层会添加
        # out_shape (max_steps, feature_dim), 其中 batch_size被忽略
        self.inp_fea_shape, self.inp_spec_shape, self.out_shape = prepare_data.get_dims(self.train_gen)
