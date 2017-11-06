#coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import time
import config
from predata import prepare_data,prepare_datasize,prepare_data_fake

np.random.seed(1)#设定种子
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE

class MULTI_MODAL(object):
    def __init__(self,datasize):
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.mix_speech_len,self.speech_fre,self.total_frames,self.spk_num_total=datasize

def main():
    print('go to model')
    print '*' * 80

    # data_generator=prepare_data('train')
    data_generator=prepare_data_fake('train') #写一个假的数据生成，可以用来写模型先

    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    datasize=prepare_datasize(data_generator)
    multi_model=MULTI_MODAL(datasize)





if __name__ == "__main__":
    main()