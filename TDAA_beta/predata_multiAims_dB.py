#coding=utf8
import sys
import os
import numpy as np
import time
import random
import config_WSJ0_dB as config
import re
import soundfile as sf
import resampy
import librosa
import shutil
import subprocess

channel_first=config.channel_first
np.random.seed(1)#设定种子
random.seed(1)

def split_forTrainDevTest(spk_list,train_or_test):
    '''为了保证一个统一的训练和测试的划分标准，不得不用通用的一些方法来限定一下,
    这里采用的是用sorted先固定方法的排序，那么不论方法或者seed怎么设置，训练测试的划分标准维持不变，
    也就是数据集会维持一直'''
    length=len(spk_list)
    # spk_list=sorted(spk_list,key=lambda x:(x[1]))#这个意思是按照文件名的第二个字符排序
    # spk_list=sorted(spk_list)#这个意思是按照文件名的第1个字符排序,暂时采用这种
    spk_list=sorted(spk_list,key=lambda x:(x[-1]))#这个意思是按照文件名的最后一个字符排序
    #TODO:暂时用最后一个字符排序，这个容易造成问题，可能第一个比较不一样的，这个需要注意一下
    if train_or_test=='train':
        return spk_list[:int(round(0.7*length))]
    elif train_or_test=='valid':
        return spk_list[(int(round(0.7*length))+1):int(round(0.8*length))]
    elif train_or_test=='test':
        return spk_list[(int(round(0.8*length))+1):]
    else:
        raise ValueError('Wrong input of train_or_test.')

def prepare_datasize(gen):
    data=gen.next()
    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    #暂时输出的是：语音长度、语音频率数量、视频截断之后的长度
    print 'datasize:',data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])
    return data[1].shape[1],data[1].shape[2],data[4].shape[1],data[-1],(data[4].shape[2],data[4].shape[3])

def prepare_data(mode,train_or_test):
    '''
    :param
    mode: type str, 'global' or 'once' ， global用来获取全局的spk_to_idx的字典，所有说话人的列表等等
    train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid，最后20%作为测试
    :return:
    '''
    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN))
    mix_feas=[]#应该是bs,n_frames,n_fre这么多
    mix_phase=[]#应该是bs,n_frames,n_fre这么多
    aim_fea=[]#应该是bs,n_frames,n_fre这么多
    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
    aim_spkname=[] #np.zeros(config.BATCH_SIZE)
    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
    multi_spk_fea_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典
    multi_spk_wav_list=[] #应该是bs个dict，每个dict里是说话人name为key，clean_fea为value的字典

    #目标数据集的总data，底下应该存放分目录的文件夹，每个文件夹应该名字是sX
    data_path=config.aim_path+'/data'
    #语音刺激
    if config.MODE==1:
        if config.DATASET=='WSJ0': #开始构建数据集
            WSJ0_eval_list=['440', '441', '442', '443', '444', '445', '446', '447']
            WSJ0_test_list=['22g', '22h', '050', '051', '052', '053', '420', '421', '422', '423']
            all_spk_train=os.listdir(data_path+'/train')
            all_spk_eval=os.listdir(data_path+'/eval')
            all_spk_test=os.listdir(data_path+'/test')
            all_spk_evaltest=os.listdir(data_path+'/eval_test')
            all_spk = all_spk_train+all_spk_eval+all_spk_test
            spk_samples_list={}
            batch_idx=0
            mix_k=random.randint(config.MIN_MIX,config.MAX_MIX)
            while True:
                mix_len=0
                # mix_k=random.randint(config.MIN_MIX,config.MAX_MIX)
                if train_or_test=='train':
                    aim_spk_k=random.sample(all_spk_train,mix_k)#本次混合的候选人
                elif train_or_test=='eval':
                    aim_spk_k=random.sample(all_spk_eval,mix_k)#本次混合的候选人
                elif train_or_test=='test':
                    aim_spk_k=random.sample(all_spk_test,mix_k)#本次混合的候选人
                elif train_or_test=='eval_test':
                    aim_spk_k=random.sample(all_spk_evaltest,mix_k)#本次混合的候选人

                multi_fea_dict_this_sample={}
                multi_wav_dict_this_sample={}

                channel_act=None
                if 1 and config.dB and config.MIN_MIX==config.MAX_MIX==2:
                    dB_rate=10**(config.dB/20.0*np.random.rand())#e**(0——0.5)
                    if np.random.rand()>0.5: #选择哪个通道来
                        channel_act=1
                    else:
                        channel_act=2
                    print 'channel to change with dB:',channel_act,dB_rate
                for k,spk in enumerate(aim_spk_k):
                    #选择dB的通道～！
                    #若是没有出现在整体列表内就注册进去,且第一次的时候读取所有的samples的名字
                    if spk not in spk_samples_list:
                        spk_samples_list[spk]=[]
                        for ss in os.listdir(data_path+'/'+train_or_test+'/'+spk):
                            spk_samples_list[spk].append(ss[:-4]) #去掉.wav后缀

                        if 0: #这部分是选择独立同分布的
                            #这个函数让spk_sanmples_list[spk]按照设定好的方式选择是train的部分还是test
                            spk_samples_list[spk]=split_forTrainDevTest(spk_samples_list[spk],train_or_test)

                    #这个时候这个spk已经注册了，所以直接从里面选就好了
                    sample_name=random.sample(spk_samples_list[spk],1)[0]
                    spk_samples_list[spk].remove(sample_name)#取出来一次之后，就把这个人去掉（避免一个batch里某段语音的重复出现）
                    spk_speech_path=data_path+'/'+train_or_test+'/'+spk+'/'+sample_name+'.wav'

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
                        aim_spkname.append(aim_spk_k[0])
                        # aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0])-1 #选定第一个作为目标说话人
                        #TODO:这里有个问题是spk是从１开始的貌似，这个后面要统一一下　-->　已经解决，构建了spk和idx的双向索引
                        aim_spk_speech=signal
                        aim_spkid.append(aim_spkname)
                        if channel_act==1:
                            signal=signal*dB_rate
                        wav_mix=signal
                        aim_fea_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                                    config.FRAME_SHIFT)))
                        aim_fea.append(aim_fea_clean)
                        # 把第一个人顺便也注册进去混合dict里
                        multi_fea_dict_this_sample[spk]=aim_fea_clean
                        multi_wav_dict_this_sample[spk]=signal

                        #视频处理部分，为了得到query
                    else:
                        if channel_act==2:
                            signal=signal*dB_rate

                        wav_mix = wav_mix + signal  # 混叠后的语音
                        #　这个说话人的语音
                        some_fea_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                                       config.FRAME_SHIFT,)))
                        multi_fea_dict_this_sample[spk]=some_fea_clean
                        multi_wav_dict_this_sample[spk]=signal

                multi_spk_fea_list.append(multi_fea_dict_this_sample) #把这个sample的dict传进去
                multi_spk_wav_list.append(multi_wav_dict_this_sample) #把这个sample的dict传进去

                # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
                if config.IS_LOG_SPECTRAL:
                    feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                        config.FRAME_SHIFT,
                                                                                        window=config.WINDOWS)))
                                         + np.spacing(1))
                else:
                    feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                     config.FRAME_SHIFT,)))

                mix_speechs[batch_idx,:]=wav_mix
                mix_feas.append(feature_mix)
                mix_phase.append(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                     config.FRAME_SHIFT,)))
                batch_idx+=1
                print 'batch_dix:{}/{},'.format(batch_idx,config.BATCH_SIZE),
                if batch_idx==config.BATCH_SIZE: #填满了一个batch
                    mix_feas=np.array(mix_feas)
                    mix_phase=np.array(mix_phase)
                    aim_fea=np.array(aim_fea)
                    # aim_spkid=np.array(aim_spkid)
                    query=np.array(query)
                    print '\nspk_list_from_this_gen:{}'.format(aim_spkname)
                    print 'aim spk list:', [one.keys() for one in multi_spk_fea_list]
                    # print '\nmix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkname.shape,query.shape,all_spk_num:'
                    # print mix_speechs.shape,mix_feas.shape,aim_fea.shape,len(aim_spkname),query.shape,len(all_spk)
                    if mode=='global':
                        all_spk=sorted(all_spk)
                        all_spk=sorted(all_spk_train)
                        all_spk_eval=sorted(all_spk_eval)
                        all_spk_test=sorted(all_spk_test)
                        dict_spk_to_idx={spk:idx for idx,spk in enumerate(all_spk)}
                        dict_idx_to_spk={idx:spk for idx,spk in enumerate(all_spk)}
                        yield all_spk,dict_spk_to_idx,dict_idx_to_spk,\
                              aim_fea.shape[1],aim_fea.shape[2],32,len(all_spk)
                              #上面的是：语音长度、语音频率、视频分割多少帧 TODO:后面把这个替换了query.shape[1]
                    elif mode=='once':
                        yield {'mix_wav':mix_speechs,
                               'mix_feas':mix_feas,
                               'mix_phase':mix_phase,
                               'aim_fea':aim_fea,
                               'aim_spkname':aim_spkname,
                               'query':query,
                               'num_all_spk':len(all_spk),
                               'multi_spk_fea_list':multi_spk_fea_list,
                               'multi_spk_wav_list':multi_spk_wav_list
                               }

                    batch_idx=0
                    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN))
                    mix_feas=[]#应该是bs,n_frames,n_fre这么多
                    mix_phase=[]
                    aim_fea=[]#应该是bs,n_frames,n_fre这么多
                    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
                    aim_spkname=[]
                    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把
                    multi_spk_fea_list=[]
                    multi_spk_wav_list=[]

        else:
            raise ValueError('No such dataset:{} for Speech.'.format(config.DATASET))
        pass

    #图像刺激
    elif config.MODE==2:
        pass

    #视频刺激
    elif config.MODE==3:
        raise ValueError('No such dataset:{} for Video'.format(config.DATASET))
    #概念刺激
    elif config.MODE==4:
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))
