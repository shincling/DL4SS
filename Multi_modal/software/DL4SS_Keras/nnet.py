# -*- coding: utf8 -*-

import time
import config
from keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense, merge, Reshape, Embedding, Masking , Convolution2D, MaxPooling2D, Activation,Flatten
from keras.models import Model
from keras.optimizers import SGD, Nadam
import keras.backend as K
from extend_layers import Attention, MeanPool, SpkLifeLongMemory, SelectSpkMemory, update_memory, MaskingGt
from model_init import ModelInit
import predict
import numpy as np

__author__ = 'jacoxu'


class NNet(ModelInit):
    def __init__(self, _log_file, weights_path):
        # 模型初始化时加载数据
        super(NNet, self).__init__(_log_file)
        self.log_file = _log_file
        # self.optimizer = SGD(lr=0.05, decay=0, momentum=0.9, nesterov=True)
        self.optimizer = Nadam(clipnorm=200)
        print("Start to build models")
        self.auditory_model, self.spk_memory_model = self.build_models(weights_path)
        print("Finished models building")

    def build_models(self, weights_path=None):
        # weights_path 如果给出weights_path的话，可以接着训练
        # inp_mix_fea_shape (MaxLen(time), feature_dim)
        mix_fea_inp = Input(shape=(self.inp_fea_len, self.inp_fea_dim), name='input_mix_feature')
        # inp_mix_spec_shape (MaxLen(time), spectrum_dim)，固定time_steps
        mix_spec_inp = Input(shape=(self.inp_fea_len, self.inp_spec_dim), name='input_mix_spectrum')
        # bg_mask_inp = Input(shape=(self.inp_fea_len, self.inp_spec_dim), name='input_bg_mask')
        # inp_target_spk_shape (1)
        target_spk_inp = Input(shape=(self.inp_spk_len, ), name='input_target_spk') #这个应该就是说话人的id的数字
        # inp_clean_fea_shape (MaxLen(time), feature_dim)，不固定time_steps
        # clean_fea_inp = Input(shape=(None, self.inp_fea_dim), name='input_clean_feature') #这个是目标说话人的原始语音，在evaluate的时候应该是不用的
        clean_fea_inp = Input(shape=config.ImageSize, name='input_clean_feature') #输入的单个目标图片

        mix_fea_layer = mix_fea_inp
        mix_spec_layer = mix_spec_inp
        target_spk_layer = target_spk_inp
        clean_fea_layer = clean_fea_inp

        '''由于图像不存在序列问题，不需要以下的一些mask'''
        # if config.IS_LOG_SPECTRAL:
        #     clean_fea_layer = MaskingGt(mask_value=np.log(np.spacing(1) * 2))(clean_fea_inp)
        # else:
        #     clean_fea_layer = Masking(mask_value=0.)(clean_fea_inp)

        '''混合语音抽取的部分保持不变'''
        # clean_fea_layer = clean_fea_inp
        # 设置了两层 双向 LSTM, 抽取混叠语音的特征
        # (None(batch), MaxLen(time), feature_dim) -> (None(batch), None(time), hidden_dim)
        for _layer in range(config.NUM_LAYERS):
            # 开始累加 LSTM, SIZE_RLAYERS LSTM隐层和输出神经元大小，
            mix_fea_layer = \
                Bidirectional(LSTM(config.HIDDEN_UNITS, return_sequences=True),
                              merge_mode='concat')(mix_fea_layer)
        # 利用全连接层, 转换到语谱图(t,f)的嵌入维度
        # (None(batch), MaxLen(time), hidden_dim) -> (None(batch), MaxLen(time), spec_dim * embed_dim)
        mix_embedding_layer = TimeDistributed(Dense(self.inp_spec_dim*config.EMBEDDING_SIZE
                                                    , activation='tanh'))(mix_fea_layer)

        # (None(batch), MaxLen(time), spec_dim * embed_dim) -> (None(batch), MaxLen(time), spec_dim, embed_dim)
        mix_embedding_layer = Reshape((self.inp_fea_len, self.inp_spec_dim, config.EMBEDDING_SIZE))(mix_embedding_layer)

        # 抽取目标说话人纯净语音的特征, 测试阶段为全0
        # (Batch,imagesize[0],imagesize[1]) -> (Batch,imageEmbedding)
        # TODO: 这块用来设计抽取图像特征所采用的网络
        # spk_vector_layer_forImage=image_net(clear_fea_layer)
        # spk_vector_layer1=Convolution2D(4, 5, 5, border_mode='valid', input_shape=(1,config.ImageSize[0],config.ImageSize[1]))(clean_fea_layer)
        clean_fea_layer=Reshape((1,config.ImageSize[0],config.ImageSize[1]))(clean_fea_layer)
        spk_vector_layer1=Convolution2D(4, 5, 5, border_mode='valid')(clean_fea_layer)
        spk_vector_layer1=Activation('relu')(spk_vector_layer1)
        spk_vector_layer1=MaxPooling2D(pool_size=(2,2))(spk_vector_layer1)

        spk_vector_layer2=Convolution2D(8, 3, 3, border_mode='valid')(spk_vector_layer1)
        spk_vector_layer2=Activation('relu')(spk_vector_layer2)
        spk_vector_layer2=MaxPooling2D(pool_size=(2,2))(spk_vector_layer2)

        spk_vector_layer3=Convolution2D(16, 3, 3, border_mode='valid')(spk_vector_layer2)
        spk_vector_layer3=Activation('relu')(spk_vector_layer3)
        spk_vector_layer3=MaxPooling2D(pool_size=(2,2))(spk_vector_layer3)

        spk_vector_flatten=Flatten()(spk_vector_layer3)
        spk_vector_layer_image=Dense(config.EMBEDDING_SIZE,init='normal')(spk_vector_flatten)


        # 加载并更新到当前的 长时记忆单元中
        # [((None(batch), 1), ((None(batch), embed_dim))] -> (None(batch), spk_size, embed_dim)
        spk_life_long_memory_layer = SpkLifeLongMemory(self.spk_size, config.EMBEDDING_SIZE, unk_spk=config.UNK_SPK,
                                                       name='SpkLifeLongMemory')([target_spk_layer, spk_vector_layer_image])

        # 抽取当前Batch下的记忆单元
        # (None(batch), embed_dim)
        spk_memory_layer = SelectSpkMemory(name='SelectSpkMemory')([target_spk_layer, spk_life_long_memory_layer])

        # 全连接层
        # (None(batch), MaxLen(time), hidden_dim) -> (None(batch), MaxLen(time), spec_dim * embed_dim)
        # (None(batch), embed_dim) -> (None(batch), embed_dim)
        # memory_layer = Dense(config.EMBEDDING_SIZE, activation='tanh')(spk_memory_layer)
        # 进行Attention(Masking)计算
        # (None(batch), MaxLen(time), spec_dim)
        output_mask_layer = Attention(self.inp_fea_len, self.inp_spec_dim, config.EMBEDDING_SIZE,
                                      mode='align', name='Attention')([mix_embedding_layer # 这是那个三维的mix语音
                                                                       , spk_memory_layer # 这个是memory里得到的目标说话人的声纹
                                                                       # , memory_layer
                                                                       # , bg_mask_layer])
                                                                       ])
        # 进行masking
        # (None(batch), MaxLen(time), spec_dim)
        output_clean = merge([output_mask_layer, mix_spec_layer], mode='mul', name='target_clean_spectrum')
        # 注意，可以有多个输入，多个输出
        auditory_model = Model(input=[mix_fea_inp, mix_spec_inp, target_spk_inp, clean_fea_inp],
                               output=[output_clean], name='auditory_model')

        # 输出Memory的结果, 用于外部长时记忆单元的更新
        spk_memory_model = Model(input=auditory_model.input,
                                 output=auditory_model.get_layer('SelectSpkMemory').output,
                                 name='spk_vec_model')
        # 如果保存过模型的话，可以加载之前的权重继续跑
        if weights_path:
            print 'Load the trained weights of ', weights_path
            self.log_file.write('Load the trained weights of %s\n' % weights_path)
            auditory_model.load_weights(weights_path)
        print 'Compiling...'
        time_start = time.time()
        # 如果采用交叉熵(categorical_crossentropy)的话，输出一定要是0-1之间的概率，那么只能用logistic或softmax做输出
        # 如非概率输出的话，可以考虑最小均方误差(mse)等loss, 后面改用概率输出的话，可换成交叉熵
        auditory_model.compile(loss='mse', optimizer=self.optimizer)
        time_end = time.time()
        print 'Compiled, cost time: %f second' % (time_end - time_start)

        return auditory_model, spk_memory_model

    def train(self):
        start_ealy_stop = config.START_EALY_STOP
        lowest_dev_loss = float('inf')
        lowest_dev_epoch = start_ealy_stop
        for epoch_num in range(config.MAX_EPOCH):
            # if (epoch_num != 0) and (epoch_num < 60) and (epoch_num % 10 == 0):
            #     # 学习率进行衰减, 如果采用SGD的话才使用学习率衰减
            #     K.set_value(self.optimizer.lr, 0.5 * K.get_value(self.optimizer.lr))
            time_start = time.time()
            loss = 0.0
            if 0 and epoch_num % 1 == 0:
                # 评估验证集结果,不应该一开始就有,只是为了测试模型是不是work
                dev_loss = predict.eval_loss_forImages(self.auditory_model, config.VALID_LIST, 'valid', epoch_num=epoch_num,
                                             log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                             batch_size=config.BATCH_SIZE_EVAL, unk_spk=config.UNK_SPK)

            for batch_size in range(config.EPOCH_SIZE):
                inp, out = next(self.train_gen)
                loss += self.auditory_model.train_on_batch(inp, out)
                # 获取当前batch的语音向量，并更新到长时memory中
                spk_memory = self.spk_memory_model.predict(inp)
                target_spk = inp['input_target_spk']
                update_memory(self.auditory_model, target_spk, spk_memory)
                time_end = time.time()
                if batch_size != config.EPOCH_SIZE-1:
                    print '\rCurrent batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) + \
                        ', epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) + \
                        ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start)),
                else:
                    print '\rCurrent batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) + \
                        ', epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) + \
                        ' and loss: %.4f, cost time: %.4f sec.' % (loss, (time_end - time_start))
                    self.log_file.write('Current batch:' + str(batch_size+1) + '/' + str(config.EPOCH_SIZE) +
                                        ' and epoch:' + str(epoch_num+1) + '/' + str(config.MAX_EPOCH) +
                                        ' and loss: %.4f, cost time: %.4f sec.\n' % (loss, (time_end - time_start)))

            if epoch_num % 1 == 0:
                # 评估验证集结果
                dev_loss = predict.eval_loss_forImages(self.auditory_model, config.VALID_LIST, 'valid', epoch_num=epoch_num,
                                             log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                             batch_size=config.BATCH_SIZE_EVAL, unk_spk=config.UNK_SPK)
                # # 评估测试集结果
                # predict.eval_loss(self.auditory_model, config.TEST_LIST, 'test', epoch_num=epoch_num,
                #                   log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                #                   batch_size=config.BATCH_SIZE_EVAL, unk_spk=config.UNK_SPK)
                # 保存模型参数
                if epoch_num >= start_ealy_stop:
                    if dev_loss < lowest_dev_loss:
                        lowest_dev_loss = dev_loss
                        lowest_dev_epoch = epoch_num
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/"+config.DATASET+"_weight_%05d_forImages.h5" \
                                                                                        % (epoch_num+1)
                        self.auditory_model.save_weights(tmp_weight_path)
                    if (epoch_num - lowest_dev_epoch >= 10) or (epoch_num == config.MAX_EPOCH-1):
                        tmp_weight_path = config.TMP_WEIGHT_FOLDER + "/"+config.DATASET+"_weight_%05d_forImages.h5" \
                                                                                        % (lowest_dev_epoch+1)
                        self.auditory_model.load_weights(tmp_weight_path)
                        print ('Early stop at epoch: %05d' % (lowest_dev_epoch+1))
                        self.log_file.write(('Early stop at epoch: %05d\n' % (lowest_dev_epoch+1)))
                        return

    def predict(self, file_list, spk_num=3, unk_spk=False, supp_time=1, add_bgd_noise=False):
        # 评估验证集结果
        predict.eval_separation_forImages(self.auditory_model, file_list, 'pred', epoch_num=0,
                                log_file=self.log_file, spk_to_idx=self.spk_to_idx,
                                batch_size=1, spk_num=spk_num, unk_spk=unk_spk, supp_time=supp_time
                                , add_bgd_noise=add_bgd_noise)
