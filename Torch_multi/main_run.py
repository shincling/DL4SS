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
import torchvision.models as models
import myNet

np.random.seed(1)#设定种子
torch.manual_seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE

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
            print '\n\n',mix_hidden.requires_grad,query.requires_grad,'\n\n'
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
        # print 'Mix speech output shape:',out.size()
        return out

class MIX_SPEECH_classifier(nn.Module):
    def __init__(self,input_fre,mix_speech_len,num_labels):
        super(MIX_SPEECH_classifier,self).__init__()
        self.input_fre=input_fre
        self.mix_speech_len=mix_speech_len
        self.layer=nn.LSTM(
            input_size=input_fre,
            hidden_size=config.HIDDEN_UNITS,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )
        self.Linear=nn.Linear(2*config.HIDDEN_UNITS,num_labels)

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
        self.layer=nn.Embedding(num_labels,embedding_size,padding_idx=-1)

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

def main():
    print('go to model')
    print '*' * 80

    spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
    global_para=spk_global_gen.next()
    print global_para
    spk_all_list,dict1,dict2,mix_speech_len,speech_fre,total_frames,spk_num_total=global_para
    del spk_global_gen
    num_labels=len(spk_all_list)

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
    query_video_layer=VIDEO_QUERY(total_frames,config.VideoSize,spk_num_total).cuda()
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
    # att=ATTENTION(4,'align')
    # mask=att(x,y)#bs*max_len

    # del data_generator
    # del data

    optimizer = torch.optim.RMSprop([{'params':mix_hidden_layer_3d.parameters()},
                                 # {'params':query_video_layer.lstm_layer.parameters()},
                                 # {'params':query_video_layer.dense.parameters()},
                                 # {'params':query_video_layer.Linear.parameters()},
                                 {'params':att_layer.parameters()},
                                 # ], lr=0.02,momentum=0.9)
                                 ], lr=0.01)
    if config.Load_param:
        # query_video_layer.load_state_dict(torch.load('param_video_layer_19'))
        mix_speech_classifier.load_state_dict(torch.load('params/param_speech_multilabel_epoch249'))
    loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_query_class=torch.nn.CrossEntropyLoss()

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        print_memory_state(memory.memory)
        for batch_idx in range(config.EPOCH_SIZE):
            print '*' * 40,epoch_idx,batch_idx,'*'*40
            train_data_gen=prepare_data('once','train')
            train_data=train_data_gen.next()

            '''混合语音len,fre,Emb 3D表示层'''
            mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(train_data[1])).cuda())
            # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

            '''Speech self Sepration　语音自分离部分'''
            mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(train_data[1])).cuda())
            top_k_mask_mixspeech=top_k_mask(mix_speech_output,alpha=0.5,top_k=3) #torch.Float型的
            mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech) # bs*num_labels（最多混合人个数）×Embedding的大小

            #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
            #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
            mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
            mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
            att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
            print att_multi_speech.size()
            att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
            print att_multi_speech.size()
            multi_mask=att_multi_speech

            x_input_map=Variable(torch.from_numpy(train_data[1])).cuda()
            print x_input_map.size()
            x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre)
            predict_multi_map=multi_mask*x_input_map_multi

            y_multi_map=np.zeros([config.BATCH_SIZE,num_labels,mix_speech_len,speech_fre],dtype=np.float32)
            batch_spk_multi_dict=train_data[-1]
            for idx,sample in enumerate(batch_spk_multi_dict):
                for spk in sample.keys():
                    print sample[spk].shape
                    try:
                        y_multi_map[idx,dict1[spk]]=sample[spk]
                    except:
                        pass
            y_multi_map= Variable(torch.from_numpy(y_multi_map)).cuda()

            loss_multi_speech=loss_multi_func(predict_multi_map,y_multi_map)
            print loss_multi_speech

            1/0

            '''视频刺激 Sepration　部分'''
            # try:
            #     query_video_output,query_video_hidden=query_video_layer(Variable(torch.from_numpy(train_data[4])).cuda())
            # except RuntimeError:
            #     print 'RuntimeError here.'+'#'*30
            #     continue

            query_video_output,query_video_hidden=query_video_layer(Variable(torch.from_numpy(train_data[4])).cuda())
            if config.Comm_with_Memory:
                #TODO:query更新这里要再检查一遍，最好改成函数，现在有点丑陋。
                aim_idx_FromVideoQuery=torch.max(query_video_output,dim=1)[1] #返回最大的参数
                aim_spk_batch=[dict2[int(idx.data.cpu().numpy())] for idx in aim_idx_FromVideoQuery]
                print 'Query class result:',aim_spk_batch,'p:',query_video_output.data.cpu().numpy()

                for idx,aim_spk in enumerate(aim_spk_batch):
                    batch_vector=torch.stack([memory.get_video_vector(aim_spk)])
                    memory.add_video(aim_spk,query_video_hidden[idx])
                query_video_hidden=query_video_hidden+Variable(batch_vector)
                query_video_hidden=query_video_hidden/torch.sum(query_video_hidden*query_video_hidden,0)
                y_class=Variable(torch.from_numpy(np.array([dict1[spk] for spk in train_data[3]])),requires_grad=False).cuda()
                print y_class
                loss_video_class=loss_query_class(query_video_output,y_class)

            mask=att_layer(mix_speech_hidden,query_video_hidden)#bs*max_len*fre

            predict_map=mask*Variable(torch.from_numpy(train_data[1])).cuda()
            y_map=Variable(torch.from_numpy(train_data[2])).cuda()
            print 'training abs norm this batch:',torch.abs(y_map-predict_map).norm().data.cpu().numpy()
            loss_all=loss_func(predict_map,y_map)
            if 0 and config.Save_param:
                torch.save(query_video_layer.state_dict(),'param_video_layer_19_forS1S5')

            if 0 and epoch_idx<20:
                loss=loss_video_class
                if epoch_idx%1==0 and batch_idx==config.EPOCH_SIZE-1:
                    torch.save(query_video_layer.state_dict(),'param_video_layer_19_forS1S5')
            else:
                # loss=loss_all+0.1*loss_video_class
                loss = loss_all
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward(retain_graph=True)         # backpropagation, compute gradients
            optimizer.step()        # apply gradients


            # Print the Params history , that it proves well.
            # print 'Parameter history:'
            # for pa_gen in [{'params':mix_hidden_layer_3d.parameters()},
            #                                  {'params':query_video_layer.lstm_layer.parameters()},
            #                                  {'params':query_video_layer.dense.parameters()},
            #                                  {'params':query_video_layer.Linear.parameters()},
            #                                  {'params':att_layer.parameters()},
            #                                  ]:
            #     print pa_gen['params'].next()






if __name__ == "__main__":
    main()
