# <-*- encoding:utf-8 -*->
"""
    The main function of DL4SS
    The python implementation of ICASSP2014 - Deep Learning for Monaural Speech Separation.
"""
import numpy
import config
import time
import nnet

# 为了保证实验可以被重复
numpy.random.seed(0)
__author__ = '[jacoxu](https://github.com/jacoxu)'


def main():
    # ############## Test,测试区 ############
    # soundFile = './TIMIT/male_train.wav'
    # sig, rate = sf.read(soundFile)
    # if rate != 8000:
    #     print ("Convert the sample rate of the file" + soundFile + "from " + str(rate) + "Hz to " + str(8000))
    #     # https://github.com/bmcfee/resampy
    #     sig_8k = resampy.resample(sig, 16000, 8000, filter='kaiser_best')
    #     soundFile_8k = './TIMIT/male_train_8k.wav'
    #     sf.write(soundFile_8k, sig_8k, 8000)
    #     sig_8k_new, rate = sf.read(soundFile_8k)
    #     sig_16k = resampy.resample(sig_8k_new, 8000, 16000, filter='kaiser_best')
    #     soundFile_16k = './TIMIT/male_train_16k.wav'
    #     sf.write(soundFile_16k, sig_16k, 16000)
    # ############## Test,测试区 ############
    config.init_config()
    print('go to model')
    print '*' * 80
    _log_file = open(config.LOG_FILE_PRE + time.strftime("_%Y_%m_%d_%H%M%S", time.localtime()), 'w')
    # 记录参数
    config.log_config(_log_file)
    # 初始化网络 initialize model
    dl4ss_model = nnet.NNet(_log_file)
    dl4ss_model.train()
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
#
