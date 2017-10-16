# -*- coding: utf8 -*-
import os
import os.path
import copy
import random

__author__ = 'jacoxu'

dataset = 'WSJ0'  # 候选：THCHS-30: 中文语音数据，WSJ0: 英文语音数据

data_folder_str = None
trained_spk = ['spk01', 'spk02', 'spk03', 'spk04', 'spk05', 'spk06', 'spk07', 'spk08', 'spk09', 'spk10']
trained_sound_split = [7, 3, 5]
untrained_spk = ['spk11', 'spk12', 'spk13', 'spk14', 'spk15']
untrained_sound_split = [5, 10]

if dataset == 'THCHS-30':
    data_folder_str = "D:/jacoxu/CASIA/2017_ASA_Platform/dataset/THCHS-30/multi_spk_selected_8kHz/"
elif dataset == 'WSJ0':
    data_folder_str = "D:/jacoxu/CASIA/2017_ASA_Platform/dataset/WSJ0/multi_spk_selected_8kHz/"
else:
    raise Exception('error, wrong dataset:' + dataset)

tar_train_file_str = "./train_wavlist_"+dataset
tar_train_file_fw = open(tar_train_file_str, 'w')
tar_dev_file_str = "./valid_wavlist_"+dataset
tar_dev_file_fw = open(tar_dev_file_str, 'w')
tar_test_file_str = "./test_wavlist_"+dataset
tar_test_file_fw = open(tar_test_file_str, 'w')
tar_unk_file_str = "./unk_wavlist_"+dataset
tar_unk_file_fw = open(tar_unk_file_str, 'w')

pre_data_folder_str = "./../../dataset/"+dataset+"/multi_spk_selected_8kHz/"

train_folder_str = data_folder_str+"train/"
dev_folder_str = data_folder_str+"dev/"
test_folder_str = data_folder_str+"test/"
unk_test_folder_str = data_folder_str + "unk/test/"
unk_sound_folder_str = data_folder_str + "unk/sounds/"

# 处理训练语音的file list
# 格式：说话人语音 说话人
# 例如：./male_train.wav spk1
for parent, dir_names, file_names in os.walk(train_folder_str):
    for dir_name in dir_names:
        for tmp_parent, tmp_dir_names, tmp_file_names in os.walk(train_folder_str + dir_name):
            for tmp_file_name in tmp_file_names:
                tmp_spk = dir_name[:5]
                if tmp_spk not in trained_spk:
                    raise Exception('Wrong spk number:' + tmp_spk)
                tmp_tar_file_str = pre_data_folder_str + "train/" + dir_name + "/" + tmp_file_name
                tar_train_file_fw.write(tmp_tar_file_str+" "+tmp_spk+"\n")
tar_train_file_fw.close()

# 处理开发集语音的file list
# 格式：目标说话人语音 干扰说话人语音 目标说话人
# 例如：./male_dev.wav ./female_dev.wav spk1
for parent, dir_names, file_names in os.walk(dev_folder_str):
    for dir_name in dir_names:
        for tmp_parent, tmp_dir_names, tmp_file_names in os.walk(dev_folder_str + dir_name):
            for tmp_file_name in tmp_file_names:
                tmp_spk = dir_name[:5]
                if tmp_spk not in trained_spk:
                    raise Exception('Wrong spk number:' + tmp_spk)
                tmp_tar_file_str = pre_data_folder_str + "dev/" + dir_name + "/" + tmp_file_name
                for dir_name_bg in dir_names:
                    if dir_name_bg == dir_name:
                        continue
                    for tmp_parent_bg, tmp_dir_names_bg, tmp_file_names_bg in os.walk(dev_folder_str + dir_name_bg):
                        for tmp_file_name_bg in tmp_file_names_bg:
                            tmp_bg_file_str = pre_data_folder_str + "dev/" + dir_name_bg + "/" + tmp_file_name_bg
                            tar_dev_file_fw.write(tmp_tar_file_str+" "+tmp_bg_file_str+" "+tmp_spk+"\n")
tar_dev_file_fw.close()

# 处理测试集语音的file list
# 格式：目标说话人语音 干扰说话人语音 目标说话人
# 例如：./male_test.wav ./female_test.wav spk1
for parent, dir_names, file_names in os.walk(test_folder_str):
    for dir_name in dir_names:
        for tmp_parent, tmp_dir_names, tmp_file_names in os.walk(test_folder_str + dir_name):
            for tmp_file_name in tmp_file_names:
                tmp_spk = dir_name[:5]
                if tmp_spk not in trained_spk:
                    raise Exception('Wrong spk number:' + tmp_spk)
                tmp_tar_file_str = pre_data_folder_str + "test/" + dir_name + "/" + tmp_file_name
                for dir_name_bg in dir_names:
                    if dir_name_bg == dir_name:
                        continue
                    bg_dir_names = copy.deepcopy(dir_names)
                    bg_dir_names.pop(bg_dir_names.index(dir_name))
                    bg_dir_names.pop(bg_dir_names.index(dir_name_bg))
                    for tmp_parent_bg, tmp_dir_names_bg, tmp_file_names_bg in os.walk(test_folder_str + dir_name_bg):
                        for tmp_file_name_bg in tmp_file_names_bg:
                            tmp_bg_file_str = pre_data_folder_str + "test/" + dir_name_bg + "/" + tmp_file_name_bg
                            for tmp_dir_name_bg in random.sample(bg_dir_names, 8):
                                for tmp_tmp_parent_bg, tmp_tmp_dir_names_bg, tmp_tmp_file_names_bg in os.walk(test_folder_str + tmp_dir_name_bg):
                                    tmp_tmp_file_name_bg = random.sample(tmp_tmp_file_names_bg, 1)[0]
                                    tmp_tmp_bg_file_str = pre_data_folder_str + "test/" + tmp_dir_name_bg + "/" + tmp_tmp_file_name_bg
                                    tmp_bg_file_str = tmp_bg_file_str+','+tmp_tmp_bg_file_str
                            tar_test_file_fw.write(tmp_tar_file_str+" "+tmp_bg_file_str+" "+tmp_spk+"\n")
tar_test_file_fw.close()

# 处理UNK集语音的file list
# 格式：目标说话人语音 干扰说话人语音 目标说话人其他语音List
# 例如：./male_unk0.wav ./female_unk0.wav ./male_unk1.wav,./male_unk2.wav,./male_unk3.wav
for parent, dir_names, file_names in os.walk(unk_test_folder_str):
    for dir_name in dir_names:
        for tmp_parent, tmp_dir_names, tmp_file_names in os.walk(unk_test_folder_str + dir_name):
            for tmp_file_name in tmp_file_names:
                tmp_spk = dir_name[:5]
                if tmp_spk not in untrained_spk:
                    raise Exception('Wrong spk number:' + tmp_spk)
                tmp_tar_file_str = pre_data_folder_str + "unk/test/" + dir_name + "/" + tmp_file_name
                tmp_sds_file_str = ""
                for parent_sds, dir_names_sds, file_names_sds in os.walk(unk_sound_folder_str+dir_name):
                        for tmp_file_name_sd in file_names_sds:
                            tmp_sds_file_str += ","+pre_data_folder_str + "unk/sounds/" + dir_name + "/" + \
                                                tmp_file_name_sd
                tmp_sds_file_str = tmp_sds_file_str[1:]
                # TODO 找背景噪音
                for dir_name_bg in dir_names:
                    if dir_name_bg == dir_name:
                        continue
                    for tmp_parent_bg, tmp_dir_names_bg, tmp_file_names_bg in os.walk(unk_test_folder_str + dir_name_bg):
                        for tmp_file_name_bg in tmp_file_names_bg:
                            tmp_bg_file_str = pre_data_folder_str + "unk/test/" + dir_name_bg + "/" + tmp_file_name_bg
                            tar_unk_file_fw.write(tmp_tar_file_str+" "+tmp_bg_file_str+" unk "+tmp_sds_file_str+"\n")
tar_unk_file_fw.close()
