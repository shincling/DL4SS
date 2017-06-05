# -*- coding: utf8 -*-

import time
import config
from keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, merge
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from model_init import ModelInit
import predict

__author__ = 'jacoxu'


class NNet(ModelInit):
    def __init__(self, _log_file):
        # 模型初始化时加载数据
        super(NNet, self).__init__(_log_file)
        self.log_file = _log_file
        self.optimizer_sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
        self.networks = self.build_network()

    def build_network(self, weights_path=None):
        # weights_path 如果给出weights_path的话，可以接着训练
        mix_fea_inp = Input(shape=self.inp_fea_shape, name='input_mix_feature')
        mix_spec_inp = Input(shape=self.inp_spec_shape, name='input_mix_spectrum')

        mix_fea_layer = mix_fea_inp
        mix_spec_layer = mix_spec_inp
        # 设置了两层 双向 LSTM
        for _layer in range(config.NUM_LAYERS):
            # 开始累加 LSTM, SIZE_RLAYERS LSTM隐层和输出神经元大小，
            mix_fea_layer = \
                Bidirectional(LSTM(config.HIDDEN_UNITS, return_sequences=True),
                              input_shape=self.inp_fea_shape)(mix_fea_layer)
        # 全连接层
        output_mask_layer = TimeDistributed(Dense(self.out_shape[-1], activation='sigmoid'))(mix_fea_layer)
        # output_mask_layer_2 = merge([output_mask_layer,-1], mode='mul', name='target_clean_spectrum')
        # output_mask_layer_2 = merge([output_mask_layer_2,1], mode='sum', name='target_clean_spectrum')
        output_mask_layer_2 = TimeDistributed(Dense(self.out_shape[-1], activation='sigmoid'))(mix_fea_layer)
        # output_clean = TimeDistributed(Dense(self.out_shape[-1]), name='target_clean_spectrum')(mix_fea_layer)
        output_clean = merge([output_mask_layer, mix_spec_layer], mode='mul', name='target_clean_spectrum')
        output_clean_2 = merge([output_mask_layer_2,mix_spec_layer], mode='mul', name='target_clean_spectrum_2')
        # 注意，可以有多个输入，多个输出
        auditory_model = Model(input=[mix_fea_inp, mix_spec_inp], output=[output_clean,output_clean_2])

        # 如果保存过模型的话，可以加载之前的权重继续跑
        if weights_path:
            auditory_model.load_weights(weights_path)

        print 'Compiling...'
        time_start = time.time()
        # 如果采用交叉熵(categorical_crossentropy)的话，输出一定要是0-1之间的概率，那么只能用logistic或softmax做输出
        # 如非概率输出的话，可以考虑最小均方误差(mse)等loss, 后面改用概率输出的话，可换成交叉熵
        auditory_model.compile(loss=['mse','mse'], optimizer=self.optimizer_sgd)
        time_end = time.time()
        print 'Compiled, cost time: %f second' % (time_end - time_start)

        return auditory_model

    def train(self):
        # checkpoint，用于每轮迭代过程后的参数保存
        filepath = "weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=True)

        callbacks_list = [checkpoint]
        for epoch_num in range(config.MAX_EPOCH):
            self.networks.fit_generator(self.train_gen, samples_per_epoch=config.EPOCH_SIZE,
                                        validation_data=self.train_gen, nb_val_samples=config.BATCH_SIZE,
                                        nb_epoch=1, callbacks=callbacks_list)

            if epoch_num % 5 == 0:
                # 学习率进行衰减
                if K.get_value(self.optimizer_sgd.lr) > 0.005:
                    K.set_value(self.optimizer_sgd.lr, 0.8 * K.get_value(self.optimizer_sgd.lr))
                predict.eval_separation(self.networks, config.TEST_LIST, epoch_num=epoch_num, log_file=self.log_file)
