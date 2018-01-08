#coding=utf8
import sys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import config_WSJ0_dB as config
from predata_multiAims_dB import prepare_data
import myNet
from test_multi_labels_speech import multi_label_vector
import os
import shutil
import librosa
import soundfile as sf
import bss_test
from collections import OrderedDict


np.random.seed(1)#设定种子
torch.manual_seed(1)
random.seed(1)
# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE
test_all_outputchannel=0
config.BATCH_SIZE=1

def bss_eval_fromGenMap(multi_mask,x_input,top_k_mask_mixspeech,dict_idx2spk,data,batch_idx):
    sample_idx=0
    for each_pre,mask in zip(multi_mask,top_k_mask_mixspeech):
        _mix_spec=data['mix_phase'][sample_idx]
        xxx=x_input[sample_idx].data.cpu().numpy()
        phase_mix = np.angle(_mix_spec)
        for idx,each_spk in enumerate(each_pre):
            this_spk=idx
            y_pre_map=each_pre[idx].data.cpu().numpy()
            #如果第二个通道概率比较大
            # if idx==0 and s_idx[0].data.cpu().numpy()>s_idx[1].data.cpu().numpy():
            #     y_pre_map=1-each_pre[1].data.cpu().numpy()
            # if idx==1 and s_idx[0].data.cpu().numpy()<s_idx[1].data.cpu().numpy():
            #     y_pre_map=1-each_pre[0].data.cpu().numpy()
            y_pre_map=y_pre_map*xxx
            _pred_spec = y_pre_map* np.exp(1j * phase_mix)
            wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len = len(wav_pre)
            if test_all_outputchannel:
                min_len =  len(wav_pre)
            sf.write('batch_output/{}_testspk{}_pre.wav'.format(batch_idx,this_spk),wav_pre[:min_len],config.FRAME_RATE,)
        # sample_idx+=1

def bss_eval_recu(multi_mask,x_input,top_k_mask_mixspeech,spk_name,data,num_step,batch_idx):
    if config.Out_Sep_Result:
        dst='batch_output'

    sample_idx=0
    for each_pre,mask in zip(multi_mask,top_k_mask_mixspeech):
        _mix_spec=data['mix_phase'][sample_idx]
        xxx=x_input[sample_idx].data.cpu().numpy()
        phase_mix = np.angle(_mix_spec)
        for idx,each_spk in enumerate(each_pre):
            this_spk=idx
            y_pre_map=each_pre[idx].data.cpu().numpy()
            y_pre_map=y_pre_map*xxx
            _pred_spec = y_pre_map* np.exp(1j * phase_mix)
            wav_pre=librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT)
            min_len = len(wav_pre)
            if test_all_outputchannel:
                min_len =  len(wav_pre)
            sf.write('batch_output/{}_{}testspk_{}_pre.wav'.format(batch_idx,num_step,spk_name),wav_pre[:min_len],config.FRAME_RATE,)
        sample_idx+=1

def bss_eval_groundtrue(data,batch_idx):
    if 0 and config.Out_Sep_Result:
        dst='batch_output'
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
    for sample_idx,each_wav in enumerate(data['mix_wav']):
        sf.write('batch_output/{}_True_mix.wav'.format(sample_idx),each_wav,config.FRAME_RATE,)

    for sample_idx,each_sample in enumerate(data['multi_spk_wav_list']):
        for each_spk in each_sample.keys():
            this_spk=each_spk
            wav_genTrue=each_sample[this_spk]
            min_len = 39936
            sf.write('batch_output/{}_{}_realTrue.wav'.format(batch_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)

    for sample_idx,each_sample in enumerate(data['multi_spk_fea_list']):
        _mix_spec=data['mix_phase'][sample_idx]
        phase_mix = np.angle(_mix_spec)
        for each_spk in each_sample.keys():
            this_spk=each_spk
            y_true_map= each_sample[this_spk]
            _genture_spec = y_true_map * np.exp(1j * phase_mix)
            wav_genTrue=librosa.core.spectrum.istft(np.transpose(_genture_spec), config.FRAME_SHIFT,)
            min_len = len(each_wav)
            sf.write('batch_output/{}_{}_genTrue.wav'.format(batch_idx,this_spk),wav_genTrue[:min_len],config.FRAME_RATE,)

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

    def forward(self, input,mask_idx):
        aim_matrix=torch.from_numpy(np.array(mask_idx))
        all=self.layer(Variable(aim_matrix).cuda()) # bs*num_labels（最多混合人个数）×Embedding的大小
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
    if not sort_result.data.cpu().numpy()[0]:
        return final,[[]]
    for line_idx in range(size[0]):
        line_top_k=sort_index[line_idx][:int(sort_result[line_idx].data.cpu().numpy())]
        line_top_k=line_top_k.data.cpu().numpy()
        for i in line_top_k:
            final[line_idx,i]=1
    return final,sort_index.data.cpu().numpy()

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

    hidden_size=(config.EMBEDDING_SIZE)
    # x=torch.arange(0,24).view(2,3,4)
    # y=torch.ones([2,4])
    att_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
    att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
    print att_speech_layer

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
        # mix_speech_classifier.load_state_dict(torch.load('params/param_speech_123onezeroag3_WSJ0_multilabel_epoch40'))
        # mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix101_WSJ0_hidden3d_180'))
        # mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix101_WSJ0_emblayer_180'))
        # att_speech_layer.load_state_dict(torch.load('params/param_mix101_WSJ0_attlayer_180'))
        # mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix101_dbag1nosum_WSJ0_hidden3d_250',map_location={'cuda:1':'cuda:0'}))
        # mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix101_dbag1nosum_WSJ0_emblayer_250',map_location={'cuda:1':'cuda:0'}))
        # att_speech_layer.load_state_dict(torch.load('params/param_mix101_dbag1nosum_WSJ0_attlayer_250',map_location={'cuda:1':'cuda:0'}))

        # mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix2or3_db_WSJ0_hidden3d_560',map_location={'cuda:1':'cuda:0'}))
        # mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix2or3_db_WSJ0_emblayer_560',map_location={'cuda:1':'cuda:0'}))
        # att_speech_layer.load_state_dict(torch.load('params/param_mix2or3_db_WSJ0_attlayer_560',map_location={'cuda:1':'cuda:0'}))

        mix_speech_classifier.load_state_dict(torch.load('params/param_speech_4lstm_multilabelloss30map_epoch440'))
        mix_hidden_layer_3d.load_state_dict(torch.load('params/param_mix101_dbag2sum_WSJ0_hidden3d_460',map_location={'cuda:1':'cuda:0'}))
        mix_speech_multiEmbedding.load_state_dict(torch.load('params/param_mix101_dbag2sum_WSJ0_emblayer_460',map_location={'cuda:1':'cuda:0'}))
        att_speech_layer.load_state_dict(torch.load('params/param_mix101_dbag2sum_WSJ0_attlayer_460',map_location={'cuda:1':'cuda:0'}))
    loss_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    loss_multi_func = torch.nn.MSELoss()  # the target label is NOT an one-hotted
    # loss_multi_func = torch.nn.L1Loss()  # the target label is NOT an one-hotted
    loss_query_class=torch.nn.CrossEntropyLoss()

    print '''Begin to calculate.'''
    SDR_SUM_total=np.array([])
    for epoch_idx in range(config.MAX_EPOCH):
        if epoch_idx>0:
            print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape,epoch_idx-1,SDR_SUM.mean())
        SDR_SUM=np.array([])
        # print 'SDR_SUM for epoch {}:{}'.format(epoch_idx - 1, SDR_SUM.mean())
        dst='batch_output'
        if os.path.exists(dst):
            print " cleanup: " + dst + "/"
            shutil.rmtree(dst)
        os.makedirs(dst)
        for batch_idx in range(config.EPOCH_SIZE):
            print '*' * 40,epoch_idx,batch_idx,'*'*40
            # train_data_gen=prepare_data('once','train')
            # train_data_gen=prepare_data('once','test')
            train_data_gen=prepare_data('once','eval_test')
            train_data=train_data_gen.next()
            mix_feas=train_data['mix_feas']
            '''混合语音len,fre,Emb 3D表示层'''
            mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            # 暂时关掉video部分,因为s2 s3 s4 的视频数据不全暂时

            '''Speech self Sepration　语音自分离部分'''
            mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            #从数据里得到ground truth的说话人名字和vector
            # y_spk_list=[one.keys() for one in train_data['multi_spk_fea_list']]
            # y_spk_list= train_data['multi_spk_fea_list']
            # y_spk_gtruth,y_map_gtruth=multi_label_vector(y_spk_list,dict_spk2idx)
            # 如果训练阶段使用Ground truth的分离结果作为判别
            if 0 and config.Ground_truth:
                mix_speech_output=Variable(torch.from_numpy(y_map_gtruth)).cuda()
                if test_all_outputchannel: #把输入的mask改成全１，可以用来测试输出所有的channel
                    mix_speech_output=Variable(torch.ones(config.BATCH_SIZE,num_labels,))
                    y_map_gtruth=np.ones([config.BATCH_SIZE,num_labels])
            recu_spk_list=OrderedDict() #每step对应spk以及分离出来的目标语音
            speech_history=[] #将每step剩余speech 频谱的历史记录下来
            bss_eval_groundtrue(train_data,batch_idx)

            now_feas=train_data['mix_feas']
            while True:
                speech_history.append(now_feas)
                max_num_labels=3
                top_k_mask_mixspeech,top_k_sort_index=top_k_mask(mix_speech_output,alpha=-0.3,top_k=max_num_labels) #torch.Float型的
                # top_k_mask_idx=[np.where(line==1)[0] for line in top_k_mask_mixspeech.numpy()]
                top_k_mask_idx=top_k_sort_index
                #过滤一下，把之前见过的spk过滤掉
                print 'predict spk:',top_k_mask_idx[0]
                for k in top_k_mask_idx[0]:
                    if k not in recu_spk_list.keys():
                        top_k_mask_idx=[[k]]
                        break
                print 'flitered spk:',top_k_mask_idx[0]
                # 如果过滤完了之后啥也没有了，那么就结束了
                if len(top_k_mask_idx[0])==0:
                    break
                # elif top_k_mask_idx[0][0] in speech_history

                mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech,top_k_mask_idx) # bs*num_labels（最多混合人个数）×Embedding的大小

                assert len(top_k_mask_idx[0])==len(top_k_mask_idx[-1])
                top_k_num=len(top_k_mask_idx[0])

                #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
                #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
                mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
                mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
                mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
                # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
                att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'dot').cuda()
                att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
                # print att_multi_speech.size()
                att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
                # print att_multi_speech.size()
                multi_mask=att_multi_speech
                # multi_mask=(att_multi_speech>0.5)
                # multi_mask=Variable(torch.from_numpy(np.float32(multi_mask.data.cpu().numpy()))).cuda()
                # top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,top_k_num,1,1).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
                # multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

                x_input_map=Variable(torch.from_numpy(now_feas)).cuda()
                # print x_input_map.size()
                x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
                # predict_multi_map=multi_mask*x_input_map_multi
                predict_multi_map=multi_mask*x_input_map_multi #该说话人预测出来的频谱
                recu_spk_list[top_k_mask_idx[0][0]]=predict_multi_map

                pre_spk=dict_idx2spk[top_k_mask_idx[0][0]]
                num_step=len(recu_spk_list)
                print 'Now output the {} th spk , closest to spk <{}> in train list.'.format(num_step,pre_spk)
                # bss_eval_recu(multi_mask,x_input_map,top_k_mask_mixspeech,pre_spk,train_data,num_step-1,batch_idx)

                if num_step>=2:
                    # bss_eval_recu(multi_mask,x_input_map,top_k_mask_mixspeech,pre_spk,train_data,num_step,batch_idx)
                    break

                now_feas=((1-multi_mask)*x_input_map_multi).data.cpu().numpy().reshape(1,mix_speech_len,speech_fre)
                mix_speech_output=mix_speech_classifier(Variable(torch.from_numpy(now_feas)).cuda())
                mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(now_feas)).cuda())


            cal_spk=recu_spk_list.keys()
            mix_speech_multiEmbs=mix_speech_multiEmbedding(top_k_mask_mixspeech,cal_spk) # bs*num_labels（最多混合人个数）×Embedding的大小

            top_k_num=len(cal_spk)

            #需要计算：mix_speech_hidden[bs,len,fre,emb]和mix_mulEmbedding[bs,num_labels,EMB]的Ａttention
            #把　前者扩充为bs*num_labels,XXXXXXXXX的，后者也是，然后用ＡＴＴ函数计算它们再转回来就好了　
            mix_speech_hidden=mix_hidden_layer_3d(Variable(torch.from_numpy(train_data['mix_feas'])).cuda())
            mix_speech_hidden_5d=mix_speech_hidden.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            mix_speech_hidden_5d=mix_speech_hidden_5d.expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre,config.EMBEDDING_SIZE).contiguous()
            mix_speech_hidden_5d_last=mix_speech_hidden_5d.view(-1,mix_speech_len,speech_fre,config.EMBEDDING_SIZE)
            # att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'align').cuda()
            att_speech_layer=ATTENTION(config.EMBEDDING_SIZE,'dot').cuda()
            att_multi_speech=att_speech_layer(mix_speech_hidden_5d_last,mix_speech_multiEmbs.view(-1,config.EMBEDDING_SIZE))
            # print att_multi_speech.size()
            att_multi_speech=att_multi_speech.view(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre) # bs,num_labels,len,fre这个东西
            # print att_multi_speech.size()
            multi_mask=att_multi_speech
            # multi_mask=(att_multi_speech>0.5)
            # multi_mask=Variable(torch.from_numpy(np.float32(multi_mask.data.cpu().numpy()))).cuda()
            # top_k_mask_mixspeech_multi=top_k_mask_mixspeech.view(config.BATCH_SIZE,top_k_num,1,1).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
            # multi_mask=multi_mask*Variable(top_k_mask_mixspeech_multi).cuda()

            x_input_map=Variable(torch.from_numpy(train_data['mix_feas'])).cuda()
            # print x_input_map.size()
            x_input_map_multi=x_input_map.view(config.BATCH_SIZE,1,mix_speech_len,speech_fre).expand(config.BATCH_SIZE,top_k_num,mix_speech_len,speech_fre)
            bss_eval_fromGenMap(multi_mask,x_input_map,top_k_mask_mixspeech,dict_idx2spk,train_data,batch_idx)



        # SDR_SUM = np.append(SDR_SUM, bss_test.cal('batch_output/', 2))
        # print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM.shape,epoch_idx,SDR_SUM.mean())
        # 1/0
        SDR_SUM_total = np.append(SDR_SUM_total, bss_test.cal('batch_output/', 2))
        print 'SDR_SUM (len:{}) for epoch {} : {}'.format(SDR_SUM_total.shape,epoch_idx,SDR_SUM_total.mean())



if __name__ == "__main__":
    main()
