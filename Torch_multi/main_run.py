#coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import config
from predata import prepare_data,prepare_datasize,prepare_data_fake

np.random.seed(1)#设定种子
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE

class VIDEO_QUERY(nn.Module):
    def __init__(self,total_frames,video_size):
        super(VIDEO_QUERY,self).__init__()
        self.total_frames=total_frames
        self.video_size=video_size

    pass

class MIX_SPEECH(nn.Module):
    def __init__(self,input_fre,mix_speech_len):
        super(MIX_SPEECH,self).__init__()
        self.input_fre=input_fre
        self.mix_speech_len=mix_speech_len
        self.layer=nn.LSTM(
            input_size=input_fre,
            hidden_size=config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.Linear=nn.Linear(2*config.HIDDEN_UNITS,self.input_fre*config.EMBEDDING_SIZE)

    def forward(self,x):
        x,hidden=self.layer(x)
        x=x.contiguous()
        x=x.view(config.BATCH_SIZE*self.mix_speech_len,-1)
        out=F.tanh(self.Linear(x))
        out=out.view(config.BATCH_SIZE,self.mix_speech_len,self.input_fre,-1)
        print 'Mix speech output shape:',out.size()
        return out


class MULTI_MODAL(object):
    def __init__(self,datasize,gen):
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.mix_speech_len,self.speech_fre,self.total_frames,self.spk_num_total=datasize
        self.gen=gen

    def build(self):
        mix_hidden_layer_3d=MIX_SPEECH(self.speech_fre,self.mix_speech_len)
        output=mix_hidden_layer_3d(Variable(torch.from_numpy(self.gen.next()[1])))


def main():
    print('go to model')
    print '*' * 80

    # data_generator=prepare_data('train')
    data_generator=prepare_data_fake('train') #写一个假的数据生成，可以用来写模型先

    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    datasize=prepare_datasize(data_generator)
    mix_speech_len,speech_fre,total_frames,spk_num_total,video_size=datasize
    print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'

    # This part is to build the 3D mix speech embedding maps.
    mix_hidden_layer_3d=MIX_SPEECH(speech_fre,mix_speech_len)
    print mix_hidden_layer_3d
    mix_speech_output=mix_hidden_layer_3d(Variable(torch.from_numpy(data_generator.next()[1])))

    # This part is to conduct the video inputs.




if __name__ == "__main__":
    main()