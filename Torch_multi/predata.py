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
import shutil
import subprocess
import Image

def split_forTrainDevTest(spk_list,train_or_test):
    '''为了保证一个统一的训练和测试的划分标准，不得不用通用的一些方法来限定一下,
    这里采用的是用sorted先固定方法的排序，那么不论方法或者seed怎么设置，训练测试的划分标准维持不变，
    也就是数据集会维持一直'''
    length=len(spk_list)
    # spk_list=sorted(spk_list,key=lambda x:(x[1]))#这个意思是按照文件名的第二个字符排序
    # spk_list=sorted(spk_list)#这个意思是按照文件名的第1个字符排序,暂时采用这种
    spk_list=sorted(spk_list,key=lambda x:(x[-1]))#这个意思是按照文件名的最后一个字符排序
    #TODO:暂时用第一个字符排序，这个容易造成问题，可能第一个比较不一样的，这个需要注意胰腺癌
    if train_or_test=='train':
        return spk_list[:int(round(0.7*length))]
    elif train_or_test=='valid':
        return spk_list[(int(round(0.7*length))+1):int(round(0.8*length))]
    elif train_or_test=='test':
        return spk_list[(int(round(0.8*length))+1):]
    else:
        raise ValueError('Wrong input of train_or_test.')


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        video_id = video.split("/")[-1].split(".")[0]
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   '-y',  # (optional) overwrite output file if it exists
                                   '-i', video,  # input file
                                   '-vf', "scale={}:{}".format(config.VideoSize[0],config.VideoSize[1]),  # input file
                                   '-r', str(config.VIDEO_RATE),  # samplling rate of the Video
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%03d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

def prepare_data(train_or_test):
    '''
    :param train_or_test:type str, 'train','valid' or 'test'
     其中把每个文件夹每个人的按文件名的排序的前70%作为训练，70-80%作为valid，最后20%作为测试
    :return:
    '''
    mix_speechs=np.zeros((config.BATCH_SIZE,config.MAX_LEN))
    mix_feas=[]#应该是bs,n_frames,n_fre这么多
    aim_fea=[]#应该是bs,n_frames,n_fre这么多
    aim_spkid=[] #np.zeros(config.BATCH_SIZE)
    query=[]#应该是BATCH_SIZE，shape(query)的形式，用list再转换把

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
            batch_idx=0
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

                    #这个函数让spk_sanmples_list[spk]按照设定好的方式选择是train的部分还是test
                    spk_samples_list[spk]=split_forTrainDevTest(spk_samples_list[spk],train_or_test)

                    #这个时候这个spk已经注册了，所以直接从里面选就好了
                    sample_name=random.sample(spk_samples_list[spk],1)[0]
                    spk_samples_list[spk].remove(sample_name)#取出来一次之后，就把这个人去掉（避免一个batch里某段语音的重复出现）
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
                        aim_spk=eval(re.findall('\d+',aim_spk_k[0])[0])-1 #选定第一个作为目标说话人
                        #TODO:这里有个问题是spk是从１开始的貌似，这个后面要统一一下
                        aim_spk_speech=signal
                        aim_spkid.append(aim_spk)
                        wav_mix=signal
                        aim_fea_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))
                        aim_fea.append(aim_fea_clean)
                        aim_spk_vedio_path=data_path+'/'+spk+'/'+spk+'_video/'+sample_name+'.mpg'
                        extract_frames(aim_spk_vedio_path,sample_name)
                        image_list = sorted(os.listdir(sample_name))
                        for img in image_list:
                            im=Image.open(img)

                        shutil.rmtree(sample_name)



                    else:
                        wav_mix = wav_mix + signal  # 混叠后的语音

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

                if batch_idx==config.BATCH_SIZE-1: #填满了一个batch
                    mix_feas=np.array(mix_feas)
                    aim_fea=np.array(aim_fea)
                    aim_spkid=np.array(aim_spkid)
                    query=np.array(query)
                    break

                batch_idx+=1

            print 'hhh'
            return (mix_speechs,mix_feas,aim_fea,aim_spkid,query)
        else:
            raise ValueError('No such dataset:{} for Video'.format(config.DATASET))

    #概念刺激
    elif config.MODE==4:
        pass

    else:
        raise ValueError('No such Model:{}'.format(config.MODE))
