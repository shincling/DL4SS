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
# import torchvision.models as models
import myNet

from sklearn import metrics

np.random.seed(1)#设定种子
torch.manual_seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE

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
        self.memory=[['Unknown_id',torch.ones(3*self.hidden_size),[0,0,0]] for i in range(self.total_size)] #最外层是一个list
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
        assert query.size()==(config.BATCH_SIZE,self.hidden_size)
        assert mix_hidden.size()[-1]==self.hidden_size
        #mix_hidden：bs,max_len,fre,hidden_size  query:bs,hidden_size
        if self.mode=='dot':
            # mix_hidden=mix_hidden.view(-1,1,self.hidden_size)
            mix_shape=mix_hidden.size()
            mix_hidden=mix_hidden.view(config.BATCH_SIZE,-1,self.hidden_size)
            query=query.view(-1,self.hidden_size,1)
            print '\n\n',mix_hidden.requires_grad,query.requires_grad,'\n\n'
            dot=torch.baddbmm(Variable(torch.zeros(1,1)),mix_hidden,query)
            energy=dot.view(config.BATCH_SIZE,mix_shape[1],mix_shape[2])
            mask=F.sigmoid(energy)
            return mask

        elif self.mode=='align':
            # mix_hidden=Variable(mix_hidden)
            # query=Variable(query)
            mix_shape=mix_hidden.size()
            mix_hidden=mix_hidden.view(-1,self.hidden_size)
            mix_hidden=self.Linear_1(mix_hidden).view(config.BATCH_SIZE,-1,self.align_hidden_size)
            query=self.Linear_2(query).view(-1,1,self.align_hidden_size) #bs,1,hidden
            sum=F.tanh(mix_hidden+query)
            #TODO:从这里开始做起
            energy=self.Linear_3(sum.view(-1,self.align_hidden_size)).view(config.BATCH_SIZE,mix_shape[1],mix_shape[2])
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
        out=F.softmax(self.Linear(last_hidden)) #出处类别的概率,为什么很多都不加softmax的。。。
        # out=self.Linear(last_hidden) #出处类别的概率,为什么很多都不加softmax的。。。
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
            hidden_size=config.HIDDEN_UNITS*2,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # self.cnn=nn.Conv2d(1, 33, (5, 3), stride=(2, 1), padding=(4, 2))
        # self.cnn1=nn.Conv2d(33, 33, (5, 3), stride=(2, 1), padding=(4, 2))
        # self.cnn2=nn.Conv2d(33, 5, (5, 3), stride=(2, 1), padding=(4, 2))
        # self.cnn3=nn.Conv2d(5, 5, (5, 3), stride=(2, 1), padding=(4, 2))

        self.Linear=nn.Linear(2*2*config.HIDDEN_UNITS,num_labels)

        # self.Linear_cnn=nn.Linear(16440,num_labels)

    def forward(self,x):
        xx=x
        x,hidden=self.layer(x)
        x=x.contiguous() #bs*len*600
        x=torch.mean(x,1)
        # x=F.dropout(x,0.5)
        out=F.sigmoid(self.Linear(x))

        # print xx.size()
        # y=self.cnn(xx.view(config.BATCH_SIZE,1,xx.size()[-2],xx.size()[-1]))
        # y=self.cnn1(y)
        # y=self.cnn2(y)
        # y=self.cnn3(y)
        # y=y.view(y.size()[0],-1)
        # o=F.sigmoid(self.Linear_cnn(y))
        #
        return out

class MULTI_MODAL(object):
    def __init__(self,datasize,gen):
        print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'
        self.mix_speech_len,self.speech_fre,self.total_frames,self.spk_num_total=datasize
        self.gen=gen

    def build(self):
        mix_hidden_layer_3d=MIX_SPEECH(self.speech_fre,self.mix_speech_len)
        output=mix_hidden_layer_3d(Variable(torch.from_numpy(self.gen.next()[1])))

def multi_label_vector(x,dict_name2idx):
    y_spk,y_aim=[],[]
    length=len(dict_name2idx)
    for sample in x:
        tmp_vector=[0 for _ in range(length)]
        line=[]
        for spk in sample.keys():
            if spk not in dict_name2idx.keys():
                continue
            line.append(dict_name2idx[spk])
            for l in line:
                tmp_vector[l]=1
        y_spk.append(line)
        y_aim.append(tmp_vector)
    y_map=np.array(y_aim,dtype=np.float32)
    return y_spk,y_map

def count_multi_acc(y_out_batch,true_spk,alpha=0.5,top_k_num=3):
    def get_metrics(y, y_pre):
        hamming_loss = metrics.hamming_loss(y, y_pre)
        macro_f1 = metrics.f1_score(y, y_pre, average='macro')
        macro_precision = metrics.precision_score(y, y_pre, average='macro')
        macro_recall = metrics.recall_score(y, y_pre, average='macro')
        micro_f1 = metrics.f1_score(y, y_pre, average='micro')
        micro_precision = metrics.precision_score(y, y_pre, average='micro')
        micro_recall = metrics.recall_score(y, y_pre, average='micro')
        print 'ha,pre,re,f1',hamming_loss,micro_precision,micro_recall,micro_f1
        return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall

    len_vector=y_out_batch.shape[1]
    all_num=y_out_batch.flatten().shape[0]
    all_line=y_out_batch.shape[0]
    right_num=0
    right_line=0
    recall_rate=None
    if top_k_num:
        recall=0.
        top_k_idx=np.flip(np.argsort(y_out_batch),1)[:,:top_k_num]
        print 'top k predicted:',top_k_idx[:5]
        print 'top k real:',true_spk[:5]
        num_all_true=0
        for pre,true in zip(top_k_idx,true_spk):
            num_all_true+=len(true)
            for one_pre in pre:
                if one_pre in true:
                    recall+=1
        recall_rate=recall/num_all_true
        # print 'recall rate:',recall_rate

        y_out_batch=np.zeros_like(y_out_batch)
        for k,yy in enumerate(top_k_idx):
            y_out_batch[k,yy]=1
    else:
        y_out_batch=np.int32(y_out_batch>alpha)
        print 'Aver labels to output:',y_out_batch.sum()/float(config.BATCH_SIZE)
        y_out_batch_idx=[np.where(line==1) for line in y_out_batch]
        for ddd in y_out_batch_idx:
            if len(ddd[0])==2:
                print '+1'
        return None
        recall=0.
        num_all_true=0.
        for pre,true in zip(y_out_batch_idx,true_spk):
            num_all_true+=len(true)
            for one_pre in pre[0]:
                if one_pre in true:
                    recall+=1
        recall_rate=recall/num_all_true

    y_true_batch=[]
    for line_idx,line in enumerate(true_spk):
        out_vector=y_out_batch[line_idx]
        true_vector=np.zeros(len_vector)
        for x in line:
            # out_vector[x]=1
            true_vector[x]=1
        y_true_batch.append(true_vector)
        if (out_vector==true_vector).min()==1: #如果最小的也是true，那么就都是true了
            line_right=1
        else:
            line_right=0

        right_num+=np.count_nonzero((out_vector-true_vector)==0)
        right_line+=line_right

    y_true_batch=np.array(y_true_batch)
    return get_metrics(y_out_batch,y_true_batch)

    allelement_acc=float(right_num)/all_num
    allsample_acc=float(right_line)/all_line
    return allelement_acc,allsample_acc,all_num,all_line,recall_rate


def main():
    print('go to model')
    print '*' * 80

    spk_global_gen=prepare_data(mode='global',train_or_test='train') #写一个假的数据生成，可以用来写模型先
    global_para=spk_global_gen.next()
    print global_para
    spk_all_list,dict_spk2idx,dict_idx2spk,mix_speech_len,speech_fre,total_frames,spk_num_total=global_para
    del spk_global_gen
    num_labels=len(spk_all_list)

    #此处顺序是 mix_speechs.shape,mix_feas.shape,aim_fea.shape,aim_spkid.shape,query.shape
    #一个例子：(5, 17040) (5, 134, 129) (5, 134, 129) (5,) (5, 32, 400, 300, 3)
    print 'Begin to build the maim model for Multi_Modal Cocktail Problem.'

    mix_speech_class=MIX_SPEECH_classifier(speech_fre,mix_speech_len,num_labels).cuda()
    print mix_speech_class

    if 1 and config.Load_param:
        # para_name='param_speech_WSJ0_multilabel_epoch42'
        # para_name='param_speech_WSJ0_multilabel_epoch249'
        # para_name='param_speech_123_WSJ0_multilabel_epoch75'
        # para_name='param_speech_123_WSJ0_multilabel_epoch24'
        para_name='param_speech_123onezero_WSJ0_multilabel_epoch75' #top3 召回率80%
        para_name='param_speech_123onezeroag_WSJ0_multilabel_epoch80'#83.6
        para_name='param_speech_123onezeroag1_WSJ0_multilabel_epoch45'
        para_name='param_speech_123onezeroag2_WSJ0_multilabel_epoch40'
        para_name='param_speech_123onezeroag4_WSJ0_multilabel_epoch75'
        para_name='param_speech_123onezeroag3_WSJ0_multilabel_epoch40'
        para_name='param_speech_123onezeroag4_WSJ0_multilabel_epoch20'
        para_name='param_speech_4lstm_multilabelloss30map_epoch440' #这个是最好的目前

        para_name='param_speech_123onezeroag5dropout_WSJ0_multilabel_epoch20'#这个替代only2
        # mix_speech_class.load_state_dict(torch.load('params/param_speech_multilabel_epoch249'))
        mix_speech_class.load_state_dict(torch.load('params/{}'.format(para_name)))
        print 'Load Success:',para_name

    optimizer = torch.optim.Adam([{'params':mix_speech_class.parameters()},
                                 # {'params':query_video_layer.lstm_layer.parameters()},
                                 # {'params':query_video_layer.dense.parameters()},
                                 # {'params':query_video_layer.Linear.parameters()},
                                 # {'params':att_layer.parameters()},
                                 # ], lr=0.02,momentum=0.9)
                                 ], lr=0.00001)
    # loss_func = torch.nn.KLDivLoss()  # the target label is NOT an one-hotted
    loss_func = torch.nn.MultiLabelSoftMarginLoss()  # the target label is NOT an one-hotted
    # loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    # loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
    # loss_func = torch.nn.MultiLabelMarginLoss()  # the target label is NOT an one-hotted
    # loss_func = torch.nn.L1Loss()  # the target label is NOT an one-hotted

    print '''Begin to calculate.'''
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx%50==0:
            for ee in optimizer.param_groups:
                ee['lr']/=2
        acc_all,acc_line=0,0
        hamming_loss_all,micro_recall_all,micro_precision_all,micro_f1_all=0,0,0,0
        if epoch_idx>0:
            print 'recal_rate this epoch {}: {}'.format(epoch_idx,recall_rate_list.mean())
        recall_rate_list=np.array([])
        for batch_idx in range(config.EPOCH_SIZE):
            print '*' * 40,epoch_idx,batch_idx,'*'*40
            train_data_gen=prepare_data('once','test')
            train_data=train_data_gen.next()
            mix_speech=mix_speech_class(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())

            y_spk,y_map=multi_label_vector(train_data['multi_spk_fea_list'],dict_spk2idx)
            y_map=Variable(torch.from_numpy(y_map)).cuda()
            y_out_batch=mix_speech.data.cpu().numpy()
            # acc1,acc2,all_num_batch,all_line_batch,recall_rate=count_multi_acc(y_out_batch,y_spk,alpha=-0.1,top_k_num=2)
            # acc1,acc2,all_num_batch,all_line_batch,recall_rate=count_multi_acc(y_out_batch,y_spk,alpha=0.5,top_k_num=0)
            count_multi_acc(y_out_batch,y_spk,alpha=0.5,top_k_num=0)
            continue
            hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall=count_multi_acc(y_out_batch,y_spk,alpha=0.5,top_k_num=0)
            hamming_loss_all+=hamming_loss
            micro_f1_all+=micro_f1
            micro_precision_all+=micro_precision
            micro_recall_all+=micro_recall
            print 'Metrics now: hamming loss--{},all sample pre--{},all sample recall--{},all sample f1--{}'.format(hamming_loss_all/(batch_idx+1),micro_precision_all/(batch_idx+1),micro_recall_all/(batch_idx+1),micro_f1_all/(batch_idx+1))
            continue

            acc_all+=acc1
            acc_line+=acc2
            recall_rate_list=np.append(recall_rate_list,recall_rate)

            # print 'training abs norm this batch:',torch.abs(y_map-predict_map).norm().data.cpu().numpy()
            for i in range(config.BATCH_SIZE):
                print 'aim:{}-->{},predict:{}'.format(train_data['multi_spk_fea_list'][i].keys(),y_spk[i],mix_speech.data.cpu().numpy()[i][y_spk[i]])#除了输出目标的几个概率，也输出倒数四个的
                print 'last 4 probility:{}'.format(mix_speech.data.cpu().numpy()[i][-5:])#除了输出目标的几个概率，也输出倒数四个的
            print '\nAcc for this batch: all elements({}) acc--{},all sample({}) acc--{} recall--{}'.format(all_num_batch,acc1,all_line_batch,acc2,recall_rate)
            # if epoch_idx==0 and batch_idx<50:
            #     loss=loss_func(mix_speech,100*y_map)
            # else:
            #     loss=loss_func(mix_speech,y_map)
            # loss=loss_func(mix_speech,30*y_map)
            loss=loss_func(mix_speech,y_map)
            loss_sum=loss_func(mix_speech.sum(1),y_map.sum(1))
            print 'loss this batch:',loss.data.cpu().numpy(),loss_sum.data.cpu().numpy()
            print 'time:',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # continue
            # loss=loss+0.2*loss_sum
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        if 0 and config.Save_param and epoch_idx > 10 and epoch_idx % 5 == 0:
            try:
                torch.save(mix_speech_class.state_dict(), 'params/param_speech_123onezeroag5dropout_{}_multilabel_epoch{}'.format(config.DATASET,epoch_idx))
            except:
                print '\n\nSave paras failed ~! \n\n\n'

            # Print the Params history , that it proves well.
            # print 'Parameter history:'
            # for pa_gen in [{'params':mix_hidden_layer_3d.parameters()},
            #                                  {'params':query_video_layer.lstm_layer.parameters()},
            #                                  {'params':query_video_layer.dense.parameters()},
            #                                  {'params':query_video_layer.Linear.parameters()},
            #                                  {'params':att_layer.parameters()},
            #                                  ]:
            #     print pa_gen['params'].next()

        print 'Acc for this epoch: all elements acc--{},all sample acc--{}'.format(acc_all/config.EPOCH_SIZE,acc_line/config.EPOCH_SIZE)
        print 'Metrics for this epoch: hamming loss--{},all sample pre--{},all sample recall--{},all sample f1--{}'.format(hamming_loss_all/config.EPOCH_SIZE,micro_precision_all/config.EPOCH_SIZE,micro_recall_all/config.EPOCH_SIZE,micro_f1_all/config.EPOCH_SIZE)
        1/0






if __name__ == "__main__":
    main()