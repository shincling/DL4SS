#coding=utf8
import sys
import os
import numpy as np
import time
import random
import config
import re


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
                mix_k=random.randint(config.MIN_MIX,config.MAX_MIX)
                aim_spk_k=random.sample(all_spk,mix_k)#本次混合的候选人
                for k,spk in enumerate(aim_spk_k):
                    #若是没有出现在整体上就注册进去,且第一次的时候读取所有的samples的名字
                    if spk not in spk_samples_list:
                        spk_samples_list[spk]=[]
                        for ss in os.listdir(data_path+'/'+spk+'/'+spk+'_speech'):
                            spk_samples_list[spk].append(ss[:-4]) #去掉.wav后缀

                    #这个时候这个spk已经注册了，所以直接从里面选就好了
                    if k==0:#第一个作为目标吧
                        aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0]) #选定第一个作为目标说话人

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
