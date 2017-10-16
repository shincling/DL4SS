# <-*- encoding: utf-8 -*->
"""
    pre-process data
"""
import numpy as np
import random
import config
import soundfile as sf
import resampy
import librosa
import sys
sys.path.append('./../../dataset/Mnist_data')
import Multi_modal.dataset.Mnist_data.input_data as mnist_input_data

mnist=mnist_input_data.read_data_sets('./../../dataset/Mnist_data')
# train_mnist=[[] for i in range(10)]
# for idx in range(mnist.train.num_examples):
# for idx in range(50):
#     train_mnist[mnist.train.labels[idx]].append(mnist.train.images[idx])
# del mnist,mnist_input_data
# print 'ok'
# train_mnist[0]=np.float32(train_mnist[0])
# for num,idx in enumerate(train_mnist):
#     print 'number of {}'.format(idx),len(num)


def get_idx(train_list, valid_list=None, test_list=None):
    # 用于搜集spk的set
    spk_set = set()
    audio_path_list = []
    if train_list is not None:
        audio_path_list.append(train_list)
    else:
        raise Exception("Error, train_list should not be None.")
    if valid_list is not None:
        audio_path_list.append(valid_list)
    if test_list is not None:
        audio_path_list.append(test_list)

    for audio_list in audio_path_list:
        file_list = open(audio_list)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 2:
                print 'Wrong audio list file record in the line:', line
                continue
            spk = line[-1]
            # 把spk放到spk_set中
            spk_set.add(spk)
        file_list.close()
    spk_to_idx = {}
    # 从第1个人开始 1, 2, ...
    for spk in spk_set:
        # int('spk01'[-2:]), 留意固定的命名规则
        spk_to_idx[spk] = int(spk[-2:])
    idx_to_spk = {}
    for spk, idx in spk_to_idx.iteritems():
        idx_to_spk[idx] = spk
    return spk_to_idx, idx_to_spk


def get_dims(generator):
    inp, out = next(generator)
    inp_fea_len = inp['input_mix_feature'].shape[1]
    inp_fea_dim = inp['input_mix_feature'].shape[-1]
    inp_spec_dim = inp['input_mix_spectrum'].shape[-1]
    inp_spk_len = inp['input_target_spk'].shape[-1]
    out_spec_dim = out['target_clean_spectrum'].shape[-1]
    return inp_fea_len, inp_fea_dim, inp_spec_dim, inp_spk_len, out_spec_dim


def get_feature(audio_list, spk_to_idx, min_mix=2, max_mix=2, batch_size=1):
    """
    :param audio_list: 语音文件列表，如下
        path/to/1st.wav spk1
        path/to/2nd.wav spk2
        path/to/3rd.wav spk1
    :param spk_to_idx: 说话人映射到idx的dict, spk1:0, spk2:1, ...
    :param min_mix: 设置最小混叠说话人数
    :param max_mix: 设置最大混叠说话人数
    :param batch_size: 批处理的数据大小
    :return:
    """
    speaker_audios = {}
    batch_input_mix_fea = []
    batch_input_mix_spec = []
    # batch_input_silence_mask = []
    batch_input_spk = []
    batch_input_clean_fea = []
    batch_target_spec = []
    batch_input_len = []  # 后面根据len进行输出语音裁剪, 训练阶段无用
    batch_count = 0
    while True:  # 无限制地生成训练样例，直到当前batch_size的list满了
        # 随机选择一个说话人混叠数
        mix_k = np.random.randint(min_mix, max_mix+1)
        # 如果要参与混叠的说话人数 大于 当前语音集合中的说话人数，则重新加载语音
        # 否则，跳过直接进行混叠
        if mix_k > len(speaker_audios):
            # 读取说话人语音进行混叠
            speaker_audios = {}
            file_list = open(audio_list)
            for line in file_list:
                line = line.strip().split()
                if len(line) != 2:
                    print 'Wrong audio list file record in the line:', line
                    continue
                file_str, spk = line
                if spk not in speaker_audios:
                    speaker_audios[spk] = []
                speaker_audios[spk].append(file_str)
            file_list.close()
            # 将每个说话人的语音顺序打乱
            for spk in speaker_audios:
                random.shuffle(speaker_audios[spk])

        # 开始混叠语音
        wav_mix = None
        # 记录目标说话人
        target_spk = None
        # 记录混叠语音长度，用于测试阶段输出语音的裁剪
        mix_len = 0
        # 记录纯净目标语音
        target_sig = None
        # 开始随机选择说话人的语音进行混叠，加载到batch数据中
        # 先注释掉随机选择说话人，因为目前的算法有排列问题，后期加入说话人注意之后改为随机选择说话人语音
        # for spk in speaker_audios.keys():
        for spk in random.sample(speaker_audios.keys(), mix_k):
            file_str = speaker_audios[spk].pop()
            if not speaker_audios[spk]:
                del(speaker_audios[spk])  # 删除掉空的说话人
            signal, rate = sf.read(file_str)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            # 转成List
            signal = list(signal)
            if len(signal) > config.MAX_LEN:  # 根据最大长度裁剪
                signal = signal[:config.MAX_LEN]
            # 更新混叠语音长度
            if len(signal) > mix_len:
                mix_len = len(signal)

            # 转回Array, 进行信号预处理
            signal = np.array(signal)
            signal -= np.mean(signal)  # 语音信号预处理，先减去均值
            signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化

            # 转成List
            signal = list(signal)
            # 如果需要augment数据的话，先进行随机shift, 以后考虑固定shift
            if config.AUGMENT_DATA:
                random_shift = random.sample(range(len(signal)), 1)[0]
                signal = signal[random_shift:] + signal[:random_shift]

            if len(signal) < config.MAX_LEN:  # 根据最大长度用 0 补齐,
                signal.extend(np.zeros(config.MAX_LEN - len(signal)))
            # 转回Array
            signal = np.array(signal)

            if wav_mix is None:
                wav_mix = signal
                target_sig = signal  # 混叠前的纯净语音，只保存第一个说话人语音就好
                target_spk = spk_to_idx[spk]  # 目标说话人
            else:
                # 注意：这里一定不可以改成 wav_mix += signal的形式，target_sig会被修改（地址传递）
                wav_mix = wav_mix + signal  # 混叠后的语音

        # 这里采用log 以后可以考虑采用MFCC或GFCC特征做为输入
        if config.IS_LOG_SPECTRAL:
            feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
        else:
            feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))
        # 混叠的语谱图，用于masking输出
        spec_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                  config.FRAME_SHIFT, window=config.WINDOWS)))

        #　这里直接采用单个说话人图像的原始特征作为输入,从Ｍnist的训练数据中随机抽取一个。
        idxs_for_spk=np.where(mnist.train.labels==(target_spk-1))[0]
        feature_inp_clean=mnist.train.images[random.choice(idxs_for_spk)].reshape(config.ImageSize)

        # 计算纯净语音的语谱图, STFTs for individual signals
        spec_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))

        # silence_mask = np.ones(spec_mix.shape)
        # if config.DB_THRESHOLD > 0:
        #     log_feature_mix = np.log10(spec_mix)
        #     m_threshold = np.max(log_feature_mix) - config.DB_THRESHOLD/20.  # From dB to log10 power
        #     silence_mask[log_feature_mix < m_threshold] = 0.  # 这里的mask是为了把silence覆盖住，不参与模型的训练过程中
        # 加载到batch缓存中
        batch_input_mix_fea.append(feature_mix)
        batch_input_mix_spec.append(spec_mix)
        # batch_input_silence_mask.append(silence_mask)
        batch_input_spk.append(target_spk)
        batch_input_clean_fea.append(feature_inp_clean)
        batch_target_spec.append(spec_clean)
        batch_input_len.append(mix_len)
        batch_count += 1

        if batch_count == batch_size:
            # 混叠特征，mix_input_fea (batch_size, time_steps, feature_dim)
            mix_input_fea = np.array(batch_input_mix_fea).reshape((batch_size, ) + feature_mix.shape)
            # 混叠语谱，mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec = np.array(batch_input_mix_spec).reshape((batch_size, ) + spec_mix.shape)
            # bg_input_mask = np.array(batch_input_silence_mask).reshape((batch_size, ) + spec_mix.shape)
            # 目标说话人，target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk, dtype=np.int32).reshape((batch_size, 1))
            # 目标特征，clean_input_fea (batch_size, time_steps, feature_dim)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((batch_size, ) + feature_inp_clean.shape)
            # 目标
            # 目标语谱，clean_target_spec (batch_size, time_steps, spectrum_dim)
            clean_target_spec = np.array(batch_target_spec).reshape((batch_size, ) + spec_clean.shape)
            # 挂起生成器，等待调用next
            yield ({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec,
                    # 'input_bg_mask': bg_input_mask,
                    'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea},
                   {'target_clean_spectrum': clean_target_spec})
            batch_input_mix_fea = []
            batch_input_mix_spec = []
            # batch_input_silence_mask = []
            batch_input_spk = []
            batch_input_clean_fea = []
            batch_target_spec = []
            batch_input_len = []
            batch_count = 0

if __name__ == "__main__":
    # 测试函数
    config.init_config()
    spk_to_idx, idx_to_spk = get_idx(config.TRAIN_LIST, config.VALID_LIST, config.TEST_LIST)
    x, y = next(get_feature(config.TRAIN_LIST, spk_to_idx, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                            batch_size=config.BATCH_SIZE))
    print (x['input_mix_feature'].shape)
    print (x['input_mix_spectrum'].shape)
    print (x['input_target_spk'].shape)
    print (x['input_clean_feature'].shape)
    print (y['target_clean_spectrum'].shape)
