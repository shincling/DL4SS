#coding=utf8
import sys
import os
import numpy as np
import time
import random
import config
import re
import soundfile as sf
import resampy
import librosa

def prepare_data():
    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN))
    mix_feas=np.zeros(config.BATCH_SIZE,None,None)
    aim_fea=np.zeros(config.BATCH_SIZE,None,None)
    aim_spkid=0
    query=np.zeros(config.BATCH_SIZE,None,None)

    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=config.aim_path+'/data'
    #语音刺激
    if config.MODE==1:
        pass

    #图像刺激
    elif config.MODE==2:
        pass

    #视频刺激
    elif config.MODE==3:
        if config.DATASET=='AVA':
            pass
        elif config.DATASET=='GRID':
            #开始构建ＧＲＩＤ数据集
            all_spk=os.listdir(data_path)
            spk_samples_list={}
            while True:
                mix_len=0
                mix_k=random.randint(config.MIN_MIX,config.MAX_MIX)
                aim_spk_k=random.sample(all_spk,mix_k)#本次混合的候选人
                for k,spk in enumerate(aim_spk_k):

                    #若是没有出现在整体列表内就注册进去,且第一次的时候读取所有的samples的名字
                    if spk not in spk_samples_list:
                        spk_samples_list[spk]=[]
                        for ss in os.listdir(data_path+'/'+spk+'/'+spk+'_speech'):
                            spk_samples_list[spk].append(ss[:-4]) #去掉.wav后缀

                    #这个时候这个spk已经注册了，所以直接从里面选就好了
                    sample_name=random.sample(spk_samples_list[spk],1)[0]
                    spk_samples_list.pop(sample_name)#取出来一次之后，就把这个人去掉（避免一个batch里某段语音的重复出现）
                    spk_speech_path=data_path+'/'+spk+'/'+spk+'_speech/'+sample_name+'.wav'

                    signal, rate = sf.read(spk_speech_path)  # signal 是采样值，rate 是采样频率
                    if len(signal.shape) > 1:
                        signal = signal[:, 0]
                    if rate != config.FRAME_RATE:
                        # 如果频率不是设定的频率则需要进行转换
                        signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
                    if signal.shape[0] > config.MAX_LEN:  # 根据最大长度裁剪
                        signal = signal[:config.MAX_LEN]
                    # 更新混叠语音长度
                    if signal.shape[0] > mix_len:
                        mix_len = signal.shape[0]

                    signal -= np.mean(signal)  # 语音信号预处理，先减去均值
                    signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

                    # 如果需要augment数据的话，先进行随机shift, 以后考虑固定shift
                    if config.AUGMENT_DATA:
                        random_shift = random.sample(range(len(signal)), 1)[0]
                        signal = signal[random_shift:] + signal[:random_shift]

                    if signal.shape[0] < config.MAX_LEN:  # 根据最大长度用 0 补齐,
                        signal=np.append(signal,np.zeros(config.MAX_LEN - signal.shape[0]))

                    if k==0:#第一个作为目标
                        aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0]) #选定第一个作为目标说话人
                        aim_spk_speech=signal
                        wav_mix=signal
                        aim_spk_vedio_path=data_path+'/'+spk+'/'+spk+'_speech/'+sample_name+'.wav'
                    else:
                        wav_mix = wav_mix + signal  # 混叠后的语音


                aim_spk_k_samples=[os.listdir(data_path+'/'+spk+'/'+spk+'_speech') for spk in aim_spk_k]
                #TODO:这里有个问题是spk是从１开始的貌似，这个后面要统一一下

                for spk in aim_spk_k: #对于此次混合的每一个说话人
                    pass










            print 'hhh'
        else:
            raise ValueError('No such dataset:{} for Video'.format(config.DATASET))

    #概念刺激
    elif config.MODE==4:
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))
