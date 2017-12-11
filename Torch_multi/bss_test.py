import numpy as np
import os
import soundfile as sf
from separation import bss_eval_sources

path='server_out/'
mix_number=len(set([l.split('_')[0] for l in os.listdir(path) if l[-3:]=='wav']))
print 'num of mixed :',mix_number


for idx in range(mix_number):
    pre_speech_channel=[]
    aim_speech_channel=[]
    mix_speech=[]
    aim_list=[l for l in os.listdir(path) if l[-3:]=='wav' and l.split('_')[0]==str(idx)]
    for l in sorted(os.listdir(path)):
        if l[-3:]!='wav':
            continue
        if l.split('_')[0]==str(idx):
            if 'True_mix' in l:
                mix_speech.append(sf.read(path+l)[0])
            if 'genTrue' in l:
                aim_speech_channel.append(sf.read(path+l)[0])
            if 'pre' in l:
                pre_speech_channel.append(sf.read(path+l)[0])

    assert len(aim_speech_channel)==len(pre_speech_channel)
    aim_speech_channel=np.array(aim_speech_channel)
    pre_speech_channel=np.array(pre_speech_channel)
    # print aim_speech_channel.shape
    # print pre_speech_channel.shape
    print bss_eval_sources(aim_speech_channel,pre_speech_channel)



