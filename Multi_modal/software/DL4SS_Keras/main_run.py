# <-*- encoding:utf-8 -*->
"""
    The main function of DL4SS
    The python implementation of ICASSP2014 - Deep Learning for Monaural Speech Separation.
"""
import matlab
import matlab.engine
import numpy as np
import config
import time
import nnet


# 为了保证实验可以被重复
np.random.seed(1)
__author__ = '[jacoxu](https://github.com/jacoxu)'


def main():
    # ############## Test,测试区 ############

    # ############## Test,测试区 ############
    config.init_config()
    print('go to model')
    print '*' * 80
    _log_file = open(config.LOG_FILE_PRE + time.strftime("_%Y_%m_%d_%H%M%S", time.localtime()), 'w')
    # 记录参数
    config.log_config(_log_file)
    # 初始化网络 initialize model
    weights_path = './_tmp_weights/WSJ0_weight_00021.h5'
    # weights_path = None
    dl4ss_model = nnet.NNet(_log_file, weights_path)
    if config.MODE == 1:
        print 'still train'
        dl4ss_model.train()

    elif (config.MODE == 2) and (weights_path is not None):
        # 在训练好的模型上测试TEST
        # print 'valid spk number: 2'
        # _log_file.write('valid spk number: 2\n')
        # dl4ss_model.predict(config.VALID_LIST, spk_num=2)
        print 'test spk number: 2'
        _log_file.write('test spk number: 2\n')
        dl4ss_model.predict(config.TEST_LIST, spk_num=2)
        # print 'tes spk number: 3'
        # _log_file.write('test spk number: 3\n')
        # dl4ss_model.predict(config.TEST_LIST, spk_num=3)
        print 'test spk number: 2 with bg noise'
        _log_file.write('test spk number: 2 with bg noise\n')
        dl4ss_model.predict(config.TEST_LIST, spk_num=2, add_bgd_noise=True)

        # 在训练好的模型上测试UNK
        # for supp_time in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
        # for supp_time in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
        #     print 'unk spk and supplemental wav span: %02d' % supp_time
        #     _log_file.write('unk spk and supplemental wav span: %02d\n' % supp_time)
        #     dl4ss_model.predict(config.UNK_LIST, spk_num=2, unk_spk=True, supp_time=supp_time)
        # else:
        #     print 'Wrong mode: %s' % config.MODE
        #     _log_file.write('Wrong mode: %s\n' % config.MODE)
    _log_file.close()

if __name__ == "__main__":
    print '[Test-jacoxu] - 001'
    main()

# TODO 要做的事情
    # 1. 后期加入RNN的Mask, 参考COLING2016的MaskingLayer()
    # 2. 考虑Input的batch_size和step维度设置为None
    # 3. output是否NormL1，参考ICASSP2014的Matlab代码outputL1
    # 4. batch_size 目前是有问题的，后面要进行修正，考虑每个epoch采集多少数据
    # 5. 最大采样点数需要考虑小于实际采样点数时的裁剪问题
    # 6. 目前采用说话人数目的编码不够酷，如果可以直接通过一段纯净语音就抽取出说话人特征的话会比较酷
    # 7. 第6条，可能需要判断是否为training和testing阶段，那么需要利用in_train_phase，参考Dropout参数如何设置的
    # 8. 考虑Memory的初始化和更新方式
    # 9. 代码最后一定要review一遍横纵坐标是否正确，以防出现转置导致的错误
    # 10. 考虑特征层是否添加Masking?
#
