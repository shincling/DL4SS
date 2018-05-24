#coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import time
import config_WSJ0_dB as config
from predata_fromList_cRM_123 import prepare_data,prepare_datasize
from test_multi_labels_speech import multi_label_vector
import os
import shutil
import librosa
import soundfile as sf
import bss_test
# import matlab
# import matlab.engine
# from separation import bss_eval_sources
# import bss_test
import lrs

np.random.seed(1)#设定种子
torch.manual_seed(1)
random.seed(1)
torch.cuda.set_device(0)
test_all_outputchannel=0

def bss_eval(predict_multi_map,y_multi_map,y_map_gtruth,dict_idx2spk,train_data):
    #评测和结果输出部分
    if config.Out_Sep_Result:
        dst='batch_output'
        if os.path.exists(dst):
            print " \ncleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    for sample_idx,each_sample in enumerate(train_data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk=each_spk
            wav_genTrue=each_sample[this_spk]
            min_len = 39936
            sf.write('batch_output/{}_{}_realTrue.wav'.format(sample_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)

    # 对于每个sample
    sample_idx=0 #代表一个batch里的依次第几个
    for each_y,each_pre,each_trueVector,spk_name in zip(y_multi_map,predict_multi_map,y_map_gtruth,train_data['aim_spkname']):
        _mix_spec=train_data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx,one_cha in enumerate(each_trueVector):
            if one_cha: #　如果此刻这个候选人通道是开启的
                this_spk=dict_idx2spk[one_cha]
                y_true_map=each_y[idx].data.cpu().numpy()
                y_pre_map=each_pre[idx].data.cpu().numpy()
                _pred_spec = y_pre_map * np.exp(1j * phase_mix)
                _genture_spec = y_true_map * np.exp(1j * phase_mix)
                wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
                wav_genTrue=librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT,)
                min_len = np.min((len(train_data['multi_spk_wav_list'][sample_idx][this_spk]), len(wav_pre)))
                if test_all_outputchannel:
                    min_len =  len(wav_pre)
                sf.write('batch_output/{}_{}_pre.wav'.format(sample_idx,this_spk),wav_pre[:min_len],config.FRAME_RATE,)
                sf.write('batch_output/{}_{}_genTrue.wav'.format(sample_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx),train_data['mix_wav'][sample_idx][:min_len],config.FRAME_RATE,)
        sample_idx+=1

def print_memory_state(memory):
    print '\n memory states:'
    for one in memory:
        print '\nspk:{},video:{},age:{}'.format(one[0],one[1].cpu().numpy()[2*config.EMBEDDING_SIZE-3:2*config.EMBEDDING_SIZE+5],one[2])

class MEMORY(object):
    def __init__(self,total_size,hidden_size,):
        '''memory的设计很关键
        目前想的是一个list,每个条目包括:id(str),voide_vector,image_vector,video_vector(这三个合成一个向量）,
        age_vector（长度为3,整型，呼应voice\image\ video的样本个数）'''
        self.total_size=total_size
        self.hidden_size=hidden_size
        self.init_memory()

    def init_memory(self):
        # self.memory=[['Unknown_id',np.zeros(3*self.hidden_size),[0,0,0]] for i in range(self.total_size)] #最外层是一个list
        self.memory=[['Unknown_id',torch.zeros(3*self.hidden_size).cuda(),[0,0,0]] for i in range(self.total_size)] #最外层是一个list
        # self.memory=[['Unknown_id',torch.range(1,3*self.hidden_size),[0,0,0]] for i in range(self.total_size)] #最外层是一个list
        # self.memory=[['Unknown_id',Variable(torch.zeros(3*self.hidden_size),requires_grad=True),[0,0,0]] for i in range(self.total_size)] #最外层是一个list

    def register_spklist(self,spk_list):
        num=len(spk_list)
        sample_list=random.sample(range(len(self.memory)),num)
        for idx,sdx in enumerate(sample_list):
            assert self.memory[sdx][0]=='Unknown_id'
            self.memory[sdx][0]=spk_list[idx]

    def get_all_spkid(self):
        l=[]
        for spk in self.memory:
            l.append(spk[0])
        return set(l)

    def get_speech_vector(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][1][:self.hidden_size]
    def get_image_vector(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][1][self.hidden_size:2*self.hidden_size]
    def get_video_vector(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][1][2*self.hidden_size:3*self.hidden_size]
    def get_speech_num(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][2][0]
    def get_image_num(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][2][1]
    def get_video_num(self,spk_id,idx=None):
        if not idx:
            idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        return self.memory[idx][2][2]

    def find_spk(self,spk_id):
        for idx,spk in enumerate(self.memory):
            if spk_id==spk[0]:
                break
        else:
            raise KeyError('The spk_id:{} is not in the memory list.'.format(spk_id))
        return idx

    #注意这几个new_vector可能会是Variable变量，所以得想好这个怎么运算
    def updata_vector(self,old,new,old_num):
        '''这里定义如何更新旧的记忆里的memory和新的memory,
        必须是new(Variable)+常量的形式,返回一个可以继续计算梯度的东西
        '''
        # return (old+new)/2 #最简单的，靠近最新的样本
        if isinstance(old,Variable):
            pass
        else:
            old=Variable(old,requires_grad=False)
        tmp=(old+new) #最简单的，靠近最新的样本
        final=tmp/tmp.data.norm(2)
        return final

    def add_speech(self,spk_id,new_vector,return_final=True):
        idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        old=self.get_speech_vector()
        old_num=self.get_speech_num()
        final=self.updata_vector(old,new_vector,old_num)
        self.memory[idx][1][:self.hidden_size]=final.data #这里是FloatTensor的加法
        self.memory[idx][2][0]=self.memory[idx][2][0]+1

        if return_final:
            return final

    def add_image(self,spk_id,new_vector,return_final=True):
        idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        old=self.get_speech_vector()
        old_num=self.get_speech_num()
        final=self.updata_vector(old,new_vector,old_num)
        self.memory[idx][1][self.hidden_size:2*self.hidden_size]=final.data #这里是FloatTensor的加法
        self.memory[idx][2][1]=self.memory[idx][2][1]+1
        if return_final:
            return final

    def add_video(self,spk_id,new_vector,return_final=True):
        idx=self.find_spk(spk_id)#先找到spk_id对应的说话人的索引
        old=self.get_speech_vector(spk_id)
        old_num=self.get_speech_num(spk_id)
        final=self.updata_vector(old,new_vector,old_num)
        self.memory[idx][1][2*self.hidden_size:3*self.hidden_size]=final.data #这里是FloatTensor的加法
        self.memory[idx][2][2]=self.memory[idx][2][2]+1
        if return_final:
            return final

    def find_idx_fromQueryVector(self,form,query_vecotr):
        #todo:这个重点考虑一下如何设计
        assert form in ['speech','image','video']
        if form=='speech':
            for idx,spk in self.memory:
                if spk[2][0]:
                    similarity=None
                else:
                    continue


class ATTENTION(nn.Module):
    def __init__(self,hidden_size,mode='dot'):
        super(ATTENTION,self).__init__()
        # self.mix_emb_size=config.EMBEDDING_SIZE
        self.hidden_size=hidden_size
        self.align_hidden_size=hidden_size #align模式下的隐层大小，暂时取跟原来一致的
        self.mode=mode
        self.Linear_1=nn.Linear(self.hidden_size,self.align_hidden_size,bias=False)
        self.Linear_2=nn.Linear(hidden_size,self.align_hidden_size,bias=False)
        self.Linear_3=nn.Linear(self.align_hidden_size,1,bias=False)

    def forward(self,mix_hidden,query):
        #todo:这个要弄好，其实也可以直接抛弃memory来进行attention | DONE
        BATCH_SIZE=mix_hidden.size()[0]
        # assert query.size()==(BATCH_SIZE,self.hidden_size)
        # assert mix_hidden.size()[-1]==self.hidden_size
        #mix_hidden：bs,max_len,fre,hidden_size  query:bs,hidden_size
        if self.mode=='dot':
            if not config.is_ComlexMask:
                # mix_hidden=mix_hidden.view(-1,1,self.hidden_size)
                mix_shape=mix_hidden.size()
                mix_hidden=mix_hidden.view(BATCH_SIZE,-1,self.hidden_size)
                query=query.view(-1,self.hidden_size,1)
                # print '\n\n',mix_hidden.requires_grad,query.requires_grad,'\n\n'
                dot=torch.baddbmm(Variable(torch.zeros(1,1).cuda()),mix_hidden,query)
                energy=dot.view(BATCH_SIZE,mix_shape[1],mix_shape[2])
                # TODO: 这里可以想想是不是能换成Relu之类的
                mask=F.sigmoid(energy)
                return mask
            else:
                query_1=query[:,:,:config.EMBEDDING_SIZE].contiguous()
                query_2=query[:,:,config.EMBEDDING_SIZE:].contiguous()
                mix_shape=mix_hidden.size()
                mix_hidden=mix_hidden.view(BATCH_SIZE,-1,self.hidden_size)
                masks=[]
                for idx,query in enumerate([query_1,query_2]):
                    query=query.view(-1,self.hidden_size,1)
                    dot=torch.baddbmm(Variable(torch.zeros(1,1).cuda()),mix_hidden,query)
                    energy=dot.view(BATCH_SIZE,mix_shape[1],mix_shape[2],1)
                    masks.append(F.tanh(energy))
                mask=torch.cat((masks[0],masks[1]),3)
                return mask

        elif self.mode=='align':
            if not config.is_ComlexMask:
                # mix_hidden=Variable(mix_hidden)
                # query=Variable(query)
                mix_shape=mix_hidden.size()
                mix_hidden=mix_hidden.view(-1,self.hidden_size)
                mix_hidden=self.Linear_1(mix_hidden).view(BATCH_SIZE,-1,self.align_hidden_size)
                query=self.Linear_2(query).view(-1,1,self.align_hidden_size) #bs,1,hidden
                sum=F.tanh(mix_hidden+query)
                #TODO:从这里开始做起
                energy=self.Linear_3(sum.view(-1,self.align_hidden_size)).view(BATCH_SIZE,mix_shape[1],mix_shape[2])
                mask=F.sigmoid(energy)
                return mask
            else:
                query_1=query[:,:,:config.EMBEDDING_SIZE]
                query_2=query[:,:,config.EMBEDDING_SIZE:]
                mix_shape=mix_hidden.size()
                mix_hidden=mix_hidden.view(-1,self.hidden_size)
                mix_hidden=self.Linear_1(mix_hidden).view(BATCH_SIZE,-1,self.align_hidden_size)
                masks=[]
                for idx,query in enumerate([query_1,query_2]):
                    query=self.Linear_2(query).view(-1,1,self.align_hidden_size) #bs,1,hidden
                    sum=F.tanh(mix_hidden+query)
                    #TODO:从这里开始做起
                    energy=self.Linear_3(sum.view(-1,self.align_hidden_size)).view(BATCH_SIZE,mix_shape[1],mix_shape[2])
                    mask=F.tanh(energy)
                mask=masks[0].reshape(BATCH_SIZE,mix_shape[1],mix_shape[2],1).expand(BATCH_SIZE,mix_shape[1],mix_shape[2],2)
                mask[:,:,:,1]=masks[1]
                return mask

        else:
            print 'NO this attention methods.'
            raise IndexError


class VIDEO_QUERY(nn.Module):
    def __init__(self,total_frames,video_size,spk_total_num):
        super(VIDEO_QUERY,self).__init__()
        self.total_frames=total_frames
        self.video_size=video_size
        self.spk_total_num=spk_total_num
        self.images_net=myNet.inception_v3(pretrained=True)#注意这个输出[2]才是最后的隐层状态
        for para in self.images_net.parameters():
            para.requires_grad=False
        self.size_hidden_image=2048 #抽取的图像的隐层向量的长度,Inception_v3对应的是2048
        self.lstm_layer=nn.LSTM(
            input_size=self.size_hidden_image,
            hidden_size=config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.dense=nn.Linear(2*config.HIDDEN_UNITS,config.EMBEDDING_SIZE) #把输出的东西映射到embding_size的维度上
        self.Linear=nn.Linear(config.EMBEDDING_SIZE,self.spk_total_num)

    def forward(self, x):
        assert x.size()[2]==3#判断是否不同通道在第三个维度
        x=x.contiguous()
        x=x.view(-1,3,self.video_size[0],self.video_size[1])
        x_hidden_images=self.images_net(x)[2]
        x_hidden_images=x_hidden_images.view(-1,self.total_frames,self.size_hidden_image)
        x_lstm,hidden_lstm=self.lstm_layer(x_hidden_images)
        last_hidden=self.dense(x_lstm[:,-1])
        # out=F.softmax(self.Linear(last_hidden)) #出处类别的概率,为什么很多都不加softmax的。。。
        out=self.Linear(last_hidden) #出处类别的概率,为什么很多都不加softmax的。。。
        return out,last_hidden

class MIX_SPEECH(nn.Module):
    def __init__(self,input_fre,mix_speech_len):
        super(MIX_SPEECH,self).__init__()
        self.input_fre=input_fre
        self.mix_speech_len=mix_speech_len
        self.layer=nn.GRU(
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
        xx=x
        x=x.view(config.BATCH_SIZE*self.mix_speech_len,-1)
        # out=F.tanh(self.Linear(x))
        out=self.Linear(x)
        out=F.tanh(out)
        # out=F.relu(out)
        out=out.view(config.BATCH_SIZE,self.mix_speech_len,self.input_fre,-1)
        # print 'Mix speech output shape:',out.size()
        return out,xx

class MIX_SPEECH_classifier(nn.Module):
    def __init__(self,input_fre,mix_speech_len,num_labels):
        super(MIX_SPEECH_classifier,self).__init__()
        self.input_fre=input_fre
        self.mix_speech_len=mix_speech_len
        self.layer=nn.LSTM(
            input_size=input_fre,
            hidden_size=2*config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.Linear=nn.Linear(2*2*config.HIDDEN_UNITS,num_labels)

    def forward(self,x):
        x,hidden=self.layer(x)
        x=x.contiguous() #bs*len*600
        # x=x.view(config.BATCH_SIZE*self.mix_speech_len,-1)
        x=torch.mean(x,1)
        out=F.sigmoid(self.Linear(x))
        # out=self.Linear(x)
        return out

class SPEECH_EMBEDDING(nn.Module):
    def __init__(self,num_labels,embedding_size,max_num_channel):
        super(SPEECH_EMBEDDING,self).__init__()
        self.num_all=num_labels
        self.emb_size=embedding_size
        self.max_num_out=max_num_channel
        # self.layer=nn.Embedding(num_labels,embedding_size,padding_idx=-1)
        if not config.is_ComlexMask:
            self.layer=nn.Embedding(num_labels,embedding_size)
        else:#　如果是cRM,就得双倍说话人emb了，获得两个通道的attention大小
            self.layer=nn.Embedding(num_labels,2*embedding_size)

    def forward(self, input,mask_idx):
        aim_matrix=torch.from_numpy(np.array(mask_idx))
        all=self.layer(Variable(aim_matrix).cuda()) # bs*num_labels（最多混合人个数）×Embedding的大小
        out=all
        return out

class ADDJUST(nn.Module):
    # 这个模块是负责处理目标人的对应扰动的，进行一些偏移的调整
    def __init__(self,hidden_units,embedding_size):
        super(ADDJUST,self).__init__()
        self.hidden_units=hidden_units
        self.emb_size=embedding_size
        self.layer=nn.Linear(hidden_units+embedding_size,embedding_size,bias=False).cuda()

    def forward(self,input_hidden,prob_emb):
        top_k_num=prob_emb.size()[1]
        x=torch.mean(input_hidden,1).view(config.BATCH_SIZE,1,self.hidden_units).expand(config.BATCH_SIZE,top_k_num,self.hidden_units)
        can=torch.cat([x,prob_emb],dim=2)
        all=self.layer(can) # bs*num_labels（最多混合人个数）×Embedding的大小
        out=all
        return out

class MULTI_MODAL(object):
    def __init__(self,datasize,gen):
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.mix_speech_len,self.speech_fre,self.total_frames,self.spk_num_total=datasize
        self.gen=gen

    def build(self):
        mix_hidden_layer_3d=MIX_SPEECH(self.speech_fre,self.mix_speech_len)
        output=mix_hidden_layer_3d(Variable(torch.from_numpy(self.gen.next()[1])))


def top_k_mask(batch_pro,alpha,top_k):
    'batch_pro是 bs*n的概率分布，例如2×3的，每一行是一个概率分布\
    alpha是阈值，大于它的才可以取，可以跟Multi-label语音分离的ACC的alpha对应;\
    top_k是最多输出几个候选目标\
    输出是与bs*n的一个mask，float型的'
    size=batch_pro.size()
    final=torch.zeros(size)
    sort_result,sort_index=torch.sort(batch_pro,1,True) #先排个序
    sort_index=sort_index[:,:top_k] #选出每行的top_k的id
    sort_result=torch.sum(sort_result>alpha,1)
    for line_idx in range(size[0]):
        line_top_k=sort_index[line_idx][:int(sort_result[line_idx].data.cpu().numpy())]
        line_top_k=line_top_k.data.cpu().numpy()
        for i in line_top_k:
            final[line_idx,i]=1
    return final

def eval_bss(mix_hidden_layer_3d,adjust_layer,mix_speech_classifier,mix_speech_multiEmbedding,att_speech_layer,
             loss_multi_func,dict_spk2idx,dict_idx2spk,num_labels,mix_speech_len,speech_fre):
    for i in [mix_speech_multiEmbedding,adjust_layer,mix_speech_classifier,mix_hidden_layer_3d,att_speech_layer]:
        i.training=False
    print '#' * 40
    eval_data_gen=prepare_data('once','valid')
    SDR_SUM=np.array([])
    while True:
        print '\n\n'
        eval_data=eval_data_gen.next()
        if eval_data==False:
            break #如果这个epoch的生成器没有数据了，直接进入下一个epoch
        '''混合语音len,fre,Emb 3D表示层'''
        mix_speech_hidden,mix_tmp_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(eval_data['mix_feas'])).cuda())
        # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

        '''Speech self Sepration　语音自分离部分'''
        mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(eval_data['mix_feas'])).cuda())

        y_spk_list= eval_data['multi_spk_fea_list']
        y_spk_gtruth,y_map_gtruth=multi_label_vector(y_spk_list,dict_spk2idx)
        # 如果训练阶段使用Ground truth的分离结果作为判别
        if config.Ground_truth:
            mix_speech_output=Variable(torch.from_numpy(y_map_gtruth)).cuda()
            if test_all_outputchannel: #把输入的mask改成全１，可以用来测试输出所有的channel
                mix_speech_output=Variable(torch.ones(config.BATCH_SIZE,num_labels,))
                y_map_gtruth=np.ones([config.BATCH_SIZE,num_labels])

        top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=0.5,top_k=num_labels) #torch.Float型的
        top_k_mask_idx=[np.where(line==1)[0] for line in top_k_mask_mixspeech.numpy()]
        mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech,top_k_mask_idx) # bs*num_labels（最多混合人个数）×Embedding的大小
        mix_adjust=adjust_layer(mix_tmp_hidden,mix_speech_multiEmbs)
        mix_speech_multiEmbs=mix_adjust+mix_speech_multiEmbs

        assert len(top_k_mask_idx[0])==len(top_k_mask_idx[-1])
        top_k_num=len(top_k_mask_idx[0])

        #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
        #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
        mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
        mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
        mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
        # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
        # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'dot').cuda()
        att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
        # print att_multi_speech.size()
        att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
        # print att_multi_speech.size()
        multi_mask=att_multi_speech
        # top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,top_k_num,1,1).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
        # multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

        x_input_map=Variable(torch.from_numpy(eval_data['mix_feas'])).cuda()
        # print x_input_map.size()
        x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
        # predict_multi_map=multi_mask*x_input_map_multi
        predict_multi_map=multi_mask*x_input_map_multi

        y_multi_map=np.zeros([config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre],dtype=np.float32)
        batch_spk_multi_dict=eval_data['multi_spk_fea_list']
        for idx,sample in enumerate(batch_spk_multi_dict):
            y_idx=sorted([dict_spk2idx[spk] for spk in sample.keys()])
            assert y_idx==list(top_k_mask_idx[idx])
            for jdx,oo in enumerate(y_idx):
                y_multi_map[idx,jdx]=sample[dict_idx2spk[oo]]
        y_multi_map= Variable(torch.from_numpy(y_multi_map)).cuda()

        loss_multi_speech=loss_multi_func(predict_multi_map,y_multi_map)

        #各通道和为１的loss部分,应该可以更多的带来差异
        y_sum_map=Variable(torch.ones(config.BATCH_SIZE,mix_speech_len,speech_fre)).cuda()
        predict_sum_map=torch.sum(multi_mask,1)
        loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
        # loss_multi_speech=loss_multi_speech #todo:以后可以研究下这个和为１的效果对比一下，暂时直接MSE效果已经很不错了。
        print 'loss 1 eval, losssum eval : ',loss_multi_speech.data.cpu().numpy(),loss_multi_sum_speech.data.cpu().numpy()
        lrs.send('loss mask eval:',loss_multi_speech.data.cpu()[0])
        lrs.send('loss sum eval:',loss_multi_sum_speech.data.cpu()[0])
        loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
        print 'evaling multi-abs norm this eval batch:',torch.abs(y_multi_map-predict_multi_map).norm().data.cpu().numpy()
        print 'loss:',loss_multi_speech.data.cpu().numpy()
        bss_eval(predict_multi_map,y_multi_map,top_k_mask_idx,dict_idx2spk,eval_data)
        SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_output/', 2))
        print 'SDR_aver_now:',SDR_SUM.mean()

    SDR_aver=SDR_SUM.mean()
    print 'SDR_SUM (len:{}) for epoch eval : '.format(SDR_SUM.shape)
    lrs.send('SDR eval aver',SDR_aver)
    print '#'*40

def main():
    print('go to model')
    print '*' * 80

    spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
    global_para=spk_global_gen.next()
    print global_para
    spk_all_list,dict_spk2idx,dict_idx2spk,mix_speech_len,speech_fre,total_frames,spk_num_total,batch_total=global_para
    del spk_global_gen
    num_labels=len(spk_all_list)

    print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'

    # This part is to build the 3D mix speech embedding maps.
    mix_hidden_layer_3d=MIX_SPEECH(speech_fre,mix_speech_len).cuda()
    mix_speech_classifier=MIX_SPEECH_classifier(speech_fre,mix_speech_len,num_labels).cuda()
    mix_speech_multiEmbedding=SPEECH_EMBEDDING(num_labels,config.EMBEDDING_SIZE,spk_num_total+config.UNK_SPK_SUPP).cuda()
    print mix_hidden_layer_3d
    print mix_speech_classifier
    print mix_speech_multiEmbedding
    att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'dot').cuda()
    adjust_layer=ADDJUST(2*config.HIDDEN_UNITS,config.EMBEDDING_SIZE)
    print att_speech_layer
    print att_speech_layer.mode
    print adjust_layer
    lr_data=0.0002
    optimizer = torch.optim.Adam([{'params':mix_hidden_layer_3d.parameters()},
                                 {'params':mix_speech_multiEmbedding.parameters()},
                                 {'params':mix_speech_classifier.parameters()},
                                 {'params':adjust_layer.parameters()},
                                 {'params':att_speech_layer.parameters()},
                                 ], lr=lr_data)
    if 0 and config.Load_param:
        mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix101_WSJ0_hidden3d_180'))
        mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix101_WSJ0_emblayer_180'))
        att_speech_layer.load_state_dict(torch.load('params/param_mix101_WSJ0_attlayer_180'))
        adjust_layer.load_state_dict(torch.load('params/param_mix101_WSJ0_attlayer_180'))
    loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    # loss_multi_func = torch.nn.L1Loss()  # the target label is NOT an one-hotted
    loss_query_class=torch.nn.CrossEntropyLoss()

    lrs.send({
        'title': 'TDAA classifier',
        'batch_size':config.BATCH_SIZE,
        'batch_total':batch_total,
        'epoch_size':config.EPOCH_SIZE,
        'loss func':loss_func.__str__(),
        'initial lr':lr_data
    })

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx%10==0:
            for ee in optimizer.param_groups:
                if ee['lr']>=1e-7:
                    ee['lr']/=2
                lr_data=ee['lr']
        lrs.send('lr',lr_data)
        if epoch_idx>0:
            print 'SDR_SUM (len:{}) for epoch {} : '.format(SDR_SUM.shape,epoch_idx-1,SDR_SUM.mean())
        SDR_SUM=np.array([])
        train_data_gen=prepare_data('once','train')
        # train_data_gen=prepare_data('once','test')
        while 1 and True:
            train_data=train_data_gen.next()
            if train_data==False:
                break #如果这个epoch的生成器没有数据了，直接进入下一个epoch
            '''混合语音len,fre,Emb 3D表示层'''
            mix_speech_hidden,mix_tmp_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            # mix_tmp_hidden:[bs*T*hidden_units]
            # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

            '''Speech self Sepration　语音自分离部分'''
            mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            #从数据里得到ground truth的说话人名字和vector
            # y_spk_list=[one.keys() for one in train_data['multi_spk_fea_list']]
            y_spk_list= train_data['multi_spk_fea_list']
            y_spk_gtruth,y_map_gtruth=multi_label_vector(y_spk_list,dict_spk2idx)
            # 如果训练阶段使用Ground truth的分离结果作为判别
            if config.Ground_truth:
                mix_speech_output=Variable(torch.from_numpy(y_map_gtruth)).cuda()
                if test_all_outputchannel: #把输入的mask改成全１，可以用来测试输出所有的channel
                    mix_speech_output=Variable(torch.ones(config.BATCH_SIZE,num_labels,))
                    y_map_gtruth=np.ones([config.BATCH_SIZE,num_labels])

            top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=0.5,top_k=num_labels) #torch.Float型的
            top_k_mask_idx=[np.where(line==1)[0] for line in top_k_mask_mixspeech.numpy()]
            mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech,top_k_mask_idx) # bs*num_labels（最多混合人个数）×Embedding的大小
            if 0 and config.is_SelfTune:
                mix_adjust=adjust_layer(mix_tmp_hidden,mix_speech_multiEmbs)
                mix_speech_multiEmbs=mix_adjust+mix_speech_multiEmbs

            assert len(top_k_mask_idx[0])==len(top_k_mask_idx[-1])
            top_k_num=len(top_k_mask_idx[0])

            #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
            #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
            mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
            mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            if not config.is_ComlexMask:
                mix_speech_multiEmbedding=mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE)
            else:
                mix_speech_multiEmbedding=mix_speech_multiEmbs.view(-1,2*config.EMBEDDING_SIZE)
            att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs)
            print att_multi_speech.size()

            if not config.is_ComlexMask:
                att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
            else:
                att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre,2) # bs,num_labels,len,fre这个东西

            multi_mask=att_multi_speech
            # top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,top_k_num,1,1).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
            # multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

            x_input_map=Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            # print x_input_map.size()
            x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
            # predict_multi_map=multi_mask*x_input_map_multi
            predict_multi_map=multi_mask*x_input_map_multi

            y_multi_map=np.zeros([config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre],dtype=np.float32)
            batch_spk_multi_dict=train_data['multi_spk_fea_list']
            for idx,sample in enumerate(batch_spk_multi_dict):
                y_idx=sorted([dict_spk2idx[spk] for spk in sample.keys()])
                assert y_idx==list(top_k_mask_idx[idx])
                for jdx,oo in enumerate(y_idx):
                    y_multi_map[idx,jdx]=sample[dict_idx2spk[oo]]
            y_multi_map= Variable(torch.from_numpy(y_multi_map)).cuda()

            loss_multi_speech=loss_multi_func(predict_multi_map,y_multi_map)

            #各通道和为１的loss部分,应该可以更多的带来差异
            y_sum_map=Variable(torch.ones(config.BATCH_SIZE,mix_speech_len,speech_fre)).cuda()
            predict_sum_map=torch.sum(multi_mask,1)
            loss_multi_sum_speech=loss_multi_func(predict_sum_map,y_sum_map)
            # loss_multi_speech=loss_multi_speech #todo:以后可以研究下这个和为１的效果对比一下，暂时直接MSE效果已经很不错了。
            print 'loss 1, losssum : ',loss_multi_speech.data.cpu().numpy(),loss_multi_sum_speech.data.cpu().numpy()
            lrs.send('loss mask:',loss_multi_speech.data.cpu()[0])
            lrs.send('loss sum:',loss_multi_sum_speech.data.cpu()[0])
            loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
            print 'training multi-abs norm this batch:',torch.abs(y_multi_map-predict_multi_map).norm().data.cpu().numpy()
            print 'loss:',loss_multi_speech.data.cpu().numpy()


            optimizer.zero_grad()   # clear gradients for next train
            loss_multi_speech.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        if 1 and epoch_idx >= 10 and epoch_idx % 5 == 0:
            # torch.save(mix_speech_multiEmbedding.state_dict(),'params/param_mixalignag_{}_emblayer_{}'.format(config.DATASET,epoch_idx))
            # torch.save(mix_hidden_layer_3d.state_dict(),'params/param_mixalignag_{}_hidden3d_{}'.format(config.DATASET,epoch_idx))
            # torch.save(att_speech_layer.state_dict(),'params/param_mixalignag_{}_attlayer_{}'.format(config.DATASET,epoch_idx))
            torch.save(mix_speech_multiEmbedding.state_dict(),
                       'params/param_mix{}ag_{}_emblayer_{}'.format(att_speech_layer.mode, config.DATASET, epoch_idx))
            torch.save(mix_hidden_layer_3d.state_dict(),
                       'params/param_mix{}ag_{}_hidden3d_{}'.format(att_speech_layer.mode, config.DATASET, epoch_idx))
            torch.save(att_speech_layer.state_dict(),
                       'params/param_mix{}ag_{}_attlayer_{}'.format(att_speech_layer.mode, config.DATASET, epoch_idx))
            torch.save(adjust_layer.state_dict(),
                       'params/param_mix{}ag_{}_attlayer_{}'.format(att_speech_layer.mode, config.DATASET, epoch_idx))
        if 1 and epoch_idx % 3 == 0:
            eval_bss(mix_hidden_layer_3d,adjust_layer, mix_speech_classifier, mix_speech_multiEmbedding, att_speech_layer,
                     loss_multi_func, dict_spk2idx, dict_idx2spk, num_labels, mix_speech_len, speech_fre)

if __name__ == "__main__":
    main()
