#coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import time
import config
from predata_multiAims import prepare_data,prepare_datasize,prepare_data_fake
import myNet
from test_multi_labels_speech import multi_label_vector
import os
import shutil
import librosa
import soundfile as sf
# import matlab
# import matlab.engine
# from separation import bss_eval_sources
import bss_test
from sklearn.decomposition.pca import PCA
from sklearn.metrics.pairwise import paired_cosine_distances
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(1)#设定种子
torch.manual_seed(1)
random.seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE
test_all_outputchannel=0

def bss_eval_fromGenMap(predict_multi_map,top_k_mask_mixspeech,dict_idx2spk,data):
    if config.Out_Sep_Result:
        dst='batch_output'
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    sample_idx=0
    for each_pre,mask in zip(predict_multi_map,top_k_mask_mixspeech):
        _mix_spec=data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx,each_spk in enumerate(mask):
            if each_spk:
                this_spk=idx
                y_pre_map=each_pre[idx].data.cpu().numpy()
                _pred_spec = y_pre_map * np.exp(1j * phase_mix)
                wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
                min_len = len(wav_pre)
                if test_all_outputchannel:
                    min_len =  len(wav_pre)
                sf.write('batch_output/{}_testspk{}_pre.wav'.format(sample_idx,this_spk),wav_pre[:min_len],config.FRAME_RATE,)
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx),data['mix_wav'][sample_idx][:min_len],config.FRAME_RATE,)
        sample_idx+=1

    for sample_idx,each_sample in enumerate(data['multi_spk_fea_list']):
        _mix_spec=data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for each_spk in each_sample.keys():
            this_spk=each_spk
            y_true_map= each_sample[this_spk]
            _genture_spec = y_true_map * np.exp(1j * phase_mix)
            wav_genTrue=librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT,)
            min_len = len(wav_pre)
            sf.write('batch_output/{}_{}_genTrue.wav'.format(sample_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)

def bss_eval(predict_multi_map,y_multi_map,y_map_gtruth,dict_idx2spk,train_data):
    #评测和结果输出部分
    if config.Out_Sep_Result:
        dst='batch_output'
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)

    # 对于每个sample
    sample_idx=0 #代表一个batch里的依次第几个
    for each_y,each_pre,each_trueVector,spk_name in zip(y_multi_map,predict_multi_map,y_map_gtruth,train_data['aim_spkname']):
        _mix_spec=train_data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for idx,one_cha in enumerate(each_trueVector):
            if one_cha: #　如果此刻这个候选人通道是开启的
                this_spk=dict_idx2spk[idx]
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
            # mix_hidden=mix_hidden.view(-1,1,self.hidden_size)
            mix_shape=mix_hidden.size()
            mix_hidden=mix_hidden.view(BATCH_SIZE,-1,self.hidden_size)
            query=query.view(-1,self.hidden_size,1)
            # print '\n\n',mix_hidden.requires_grad,query.requires_grad,'\n\n'
            dot=torch.baddbmm(Variable(torch.zeros(1,1).cuda()),mix_hidden,query)
            energy=dot.view(BATCH_SIZE,mix_shape[1],mix_shape[2])
            mask=F.sigmoid(energy)
            return mask

        elif self.mode=='align':
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
        x=x.view(config.BATCH_SIZE*self.mix_speech_len,-1)
        # out=F.tanh(self.Linear(x))
        out=self.Linear(x)
        out=F.tanh(out)
        # out=F.relu(out)
        out=out.view(config.BATCH_SIZE,self.mix_speech_len,self.input_fre,-1)
        # print 'Mix speech output shape:',out.size()
        return out

class MIX_SPEECH_classifier(nn.Module):
    def __init__(self,input_fre,mix_speech_len,num_labels):
        super(MIX_SPEECH_classifier,self).__init__()
        self.input_fre=input_fre
        self.mix_speech_len=mix_speech_len
        self.layer=nn.LSTM(
            input_size=input_fre,
            hidden_size=2*config.HIDDEN_UNITS,
            num_layers=3,
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
        self.layer=nn.Embedding(num_labels,embedding_size)

    def forward(self, input):
        input_float=input
        size=input.size()
        input=torch.from_numpy(np.int64(input.numpy()))
        order_matrix=np.arange(0,self.num_all).reshape([1,self.num_all]).repeat(config.BATCH_SIZE,0)
        order_matrix=torch.from_numpy(order_matrix)
        aim_matrix=order_matrix*input
        all=self.layer(Variable(aim_matrix).cuda()) # bs*num_labels（最多混合人个数）×Embedding的大小
        input_float=input_float.view(size[0],size[1],1).expand(size[0],size[1],self.emb_size).contiguous()
        input_float=Variable(input_float,requires_grad=False).cuda()
        out=all*input_float
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


def select_the_final(mask,Embs,pro,dict,top_k=2,alpha=0.7):
    # print np.corrcoef(Embs)
    # print np.dot(Embs,np.transpose(Embs))
    ll=len(mask)
    final_matrix=np.zeros([ll,ll])
    for i in range(ll):
        for j in range(ll):
            # dis=np.linalg.(Embs[i]-Embs[j])
            dis=paired_cosine_distances(Embs[i],Embs[j])
            final_matrix[i,j]=dis
    # final_matrix=paired_cosine_distances(Embs,Embs)
    print final_matrix
    print [dict[i] for i in mask]
    print pro
    pro_sortidx=np.flip(np.argsort(pro),0)
    top_idx_list=[]
    for jj,id in enumerate(pro_sortidx):
        if jj==0:
            top_idx_list.append(id)
            continue
        else:
            if np.count_nonzero(final_matrix[id,top_idx_list]<alpha):
                continue
            else:
                top_idx_list.append(id)

        if len(top_idx_list)>=top_k:
            break

    return top_idx_list

    # pca=PCA(2)
    # newData=pca.fit_transform(Embs)
    # print newData.shape
    # f1=plt.figure(1)
    # t1=plt.scatter(newData[0,0],newData[0,1],s=60,c='red')
    # t2=plt.scatter(newData[1,0],newData[1,1],s=60,c='blue')
    # t3=plt.scatter(newData[2,0],newData[2,1],s=60,c='green')
    # t4=plt.scatter(newData[3,0],newData[3,1],s=60,c='black')
    # t5=plt.scatter(newData[4,0],newData[4,1],s=60,c='grey')
    # f1.legend((t1,t2,t3,t4,t5),('0','1','2','3','4'))
    # plt.show()

def main():
    print('go to model')
    print '*' * 80

    spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
    global_para=spk_global_gen.next()
    print global_para
    spk_all_list,dict_spk2idx,dict_idx2spk,mix_speech_len,speech_fre,total_frames,spk_num_total=global_para
    del spk_global_gen
    num_labels=len(spk_all_list)
    print 'print num_labels:',num_labels

    # data_generator=prepare_data('once','train')
    # data_generator=prepare_data_fake(train_or_test='train',num_labels=num_labels) #写一个假的数据生成，可以用来写模型先

    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    # datasize=prepare_datasize(data_generator)
    # mix_speech_len,speech_fre,total_frames,spk_num_total,video_size=datasize
    print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
    # data=data_generator.next()

    # This part is to build the 3D mix speech embedding maps.
    mix_hidden_layer_3d=MIX_SPEECH(speech_fre,mix_speech_len).cuda()
    mix_speech_classifier=MIX_SPEECH_classifier(speech_fre,mix_speech_len,num_labels).cuda()
    mix_speech_multiEmbedding=SPEECH_EMBEDDING(num_labels,config.EMBEDDING_SIZE,spk_num_total+config.UNK_SPK_SUPP).cuda()
    print mix_hidden_layer_3d
    print mix_speech_classifier
    # mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(data[1])).cuda())

    # mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(data[1])).cuda())
    # 技巧：alpha0的时候，就是选出top_k，top_k很大的时候，就是选出来大于alpha的
    # top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=config.ALPHA,top_k=config.MAX_MIX)
    # top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=config.ALPHA,top_k=3)
    # print top_k_mask_mixspeech
    # mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech) # bs*num_labels（最多混合人个数）×Embedding的大小
    # mix_speech_multiEmbs=mix_speech_multiEmbedding(Variable(torch.from_numpy(top_k_mask_mixspeech),requires_grad=False).cuda()) # bs*num_labels（最多混合人个数）×Embedding的大小

    # 需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
    # 把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
    # mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
    # mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
    # mix_speech_hidden_5d=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
    # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
    # att_multi_speech=att_speech_layer(mix_speech_hidden_5d,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
    # print att_multi_speech.size()
    # att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre,-1)
    # print att_multi_speech.size()


    # This part is to conduct the video inputs.
    # query_video_layer=VIDEO_QUERY(total_frames,config.VideoSize,spk_num_total).cuda()
    query_video_layer=None
    # print query_video_layer
    # query_video_output,xx=query_video_layer(Variable(torch.from_numpy(data[4])))

    # This part is to conduct the memory.
    # hidden_size=(config.HIDDEN_UNITS)
    hidden_size=(config.EMBEDDING_SIZE)
    memory=MEMORY(spk_num_total+config.UNK_SPK_SUPP,hidden_size)
    memory.register_spklist(spk_all_list) #把spk_list注册进空的memory里面去

    # Memory function test.
    print 'memory all spkid:',memory.get_all_spkid()
    # print memory.get_image_num('Unknown_id')
    # print memory.get_video_vector('Unknown_id')
    # print memory.add_video('Unknown_id',Variable(torch.ones(300)))

    # This part is to test the ATTENTION methond from query(~) to mix_speech
    # x=torch.arange(0,24).view(2,3,4)
    # y=torch.ones([2,4])
    att_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
    att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
    # att=ATTENTION(4,'align')
    # mask=att(x,y)#bs*max_len

    # del data_generator
    # del data

    optimizer = torch.optim.Adam([{'params':mix_hidden_layer_3d.parameters()},
                                 {'params':mix_speech_multiEmbedding.parameters()},
                                 {'params':mix_speech_classifier.parameters()},
                                 # {'params':query_video_layer.lstm_layer.parameters()},
                                 # {'params':query_video_layer.dense.parameters()},
                                 # {'params':query_video_layer.Linear.parameters()},
                                 {'params':att_layer.parameters()},
                                 {'params':att_speech_layer.parameters()},
                                 # ], lr=0.02,momentum=0.9)
                                 ], lr=0.0002)
    if 1 and config.Load_param:
        # query_video_layer.load_state_dict(torch.load('param_video_layer_19'))
        # mix_speech_classifier.load_state_dict(torch.load('params/param_speech_multilabel_epoch249'))
        # mix_speech_classifier.load_state_dict(torch.load('params/param_speech_WSJ0_multilabel_epoch249'))
        mix_speech_classifier.load_state_dict(torch.load('params/param_speech_123onezeroag3_WSJ0_multilabel_epoch40'))
        mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix101_WSJ0_hidden3d_180'))
        mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix101_WSJ0_emblayer_180'))
        att_speech_layer.load_state_dict(torch.load('params/param_mix101_WSJ0_attlayer_180'))
    loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    # loss_multi_func = torch.nn.L1Loss()  # the target label is NOT an one-hotted
    loss_query_class=torch.nn.CrossEntropyLoss()

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx>0:
            print 'SDR_SUM (len:{}) for epoch {} : '.format(SDR_SUM.shape,epoch_idx-1,SDR_SUM.mean())
        SDR_SUM=np.array([])
        # print_memory_state(memory.memory)
        print 'SDR_SUM for epoch {}:{}'.format(epoch_idx - 1, SDR_SUM.mean())
        for batch_idx in range(config.EPOCH_SIZE):
            print '*' * 40,epoch_idx,batch_idx,'*'*40
            # train_data_gen=prepare_data('once','train')
            train_data_gen=prepare_data('once','test')
            # train_data_gen=prepare_data('once','eval_test')
            train_data=train_data_gen.next()
            # test_data_gen=prepare_data('once','test')
            # test_data=train_data_gen.next()

            '''混合语音len,fre,Emb 3D表示层'''
            mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

            '''Speech self Sepration　语音自分离部分'''
            mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            #从数据里得到ground truth的说话人名字和vector
            # y_spk_list=[one.keys() for one in train_data['multi_spk_fea_list']]
            # y_spk_gtruth,y_map_gtruth=multi_label_vector(y_spk_list,dict_spk2idx)
            # 如果训练阶段使用Ground truth的分离结果作为判别
            if 0 and config.Ground_truth:
                mix_speech_output=Variable(torch.from_numpy(y_map_gtruth)).cuda()
                if test_all_outputchannel: #把输入的mask改成全１，可以用来测试输出所有的channel
                    mix_speech_output=Variable(torch.ones(config.BATCH_SIZE,num_labels,))
                    y_map_gtruth=np.ones([config.BATCH_SIZE,num_labels])

            # top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=0.5,top_k=num_labels) #torch.Float型的
            max_num_labels=5
            top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=-1,top_k=max_num_labels) #torch.Float型的
            mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech) # bs*num_labels（最多混合人个数）×Embedding的大小

            '''这里应该有个大致的去重复的手段'''
            top_k_mask_idx=[np.where(line==1) for line in top_k_mask_mixspeech.numpy()]
            for l_idx in range(config.BATCH_SIZE):
                mask_idx_thisline=top_k_mask_idx[l_idx][0]
                mix_speech_multiEmbs_thisline=mix_speech_multiEmbs.data.cpu().numpy()[l_idx,mask_idx_thisline]
                select_the_final(mask_idx_thisline,mix_speech_multiEmbs_thisline,mix_speech_output.data.cpu().numpy()[l_idx,mask_idx_thisline],dict_idx2spk,top_k=3)


            pass
            continue

            #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
            #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
            mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
            mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
            att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'dot').cuda()
            att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
            # print att_multi_speech.size()
            att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
            # print att_multi_speech.size()
            multi_mask=att_multi_speech
            top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,num_labels,1,1).expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre)
            multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

            x_input_map=Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            # print x_input_map.size()
            x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre)
            predict_multi_map=multi_mask*x_input_map_multi
            if batch_idx%100==0:
                print multi_mask
            # print predict_multi_map

            bss_eval_fromGenMap(predict_multi_map,top_k_mask_mixspeech,dict_idx2spk,train_data)
            SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_output/', 2))

            continue

            optimizer.zero_grad()   # clear gradients for next train
            loss_multi_speech.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if 1 and epoch_idx>20 and epoch_idx%10==0 and batch_idx==config.EPOCH_SIZE-1:
                torch.save(mix_speech_multiEmbedding.state_dict(),'params/param_mix_{}_emblayer_{}'.format(config.DATASET,epoch_idx))
                torch.save(mix_hidden_layer_3d.state_dict(),'params/param_mix_{}_hidden3d_{}'.format(config.DATASET,epoch_idx))
                torch.save(att_speech_layer.state_dict(),'params/param_mix_{}_attlayer_{}'.format(config.DATASET,epoch_idx))

            # print 'Parameter history:'
            # for pa_gen in [{'params':mix_hidden_layer_3d.parameters()},
            #                                  {'params':mix_speech_multiEmbedding.parameters()},
            #                                  {'params':mix_hidden_layer_3d.parameters()},
            #                                  {'params':att_speech_layer.parameters()},
            #                                  {'params':att_layer.parameters()},
            #                                  {'params':mix_speech_classifier.parameters()},
            #                                  ]:
            #     print pa_gen['params'].next().data.cpu().numpy()[0]





if __name__ == "__main__":
    main()
