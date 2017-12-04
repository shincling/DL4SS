#coding=utf8
import os
import shutil
from tqdm import tqdm

'''这个脚本用来处理移动硬盘上的数据spk_all_wav目录下所有的wav，将它们弄到本文件目录下并且分出来train/eva/test
'''
all_wav_root='/media/sw/Elements/多说话人语音数据 - WSJ0/spk_all_wav'
WSJ0_eval_list=['440', '441', '442', '443', '444', '445', '446', '447']
WSJ0_test_list=['22g', '22h', '050', '051', '052', '053', '420', '421', '422', '423']

out_root='WSJ0/data'
for one_wav_name in tqdm(os.listdir(all_wav_root)):
    spk_name=one_wav_name[:3]
    # wav_name=one_wav_name[3:-3]
    if spk_name in WSJ0_eval_list:
        spk_dir=out_root+'/eval/'+spk_name+'/'
        if not os.path.exists(spk_dir):
            os.makedirs(spk_dir)
        shutil.copy(all_wav_root+'/'+one_wav_name,spk_dir+one_wav_name)
    elif spk_name in WSJ0_test_list:
        spk_dir=out_root+'/test/'+spk_name+'/'
        if not os.path.exists(spk_dir):
            os.makedirs(spk_dir)
        shutil.copy(all_wav_root+'/'+one_wav_name,spk_dir+one_wav_name)
    elif one_wav_name[-3:]=='wav':
        spk_dir=out_root+'/train/'+spk_name+'/'
        if not os.path.exists(spk_dir):
            os.makedirs(spk_dir)
        shutil.copy(all_wav_root+'/'+one_wav_name,spk_dir+one_wav_name)
    else:
        print 'Not wav:',one_wav_name






