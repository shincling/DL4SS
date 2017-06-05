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
__author__ = 'jacoxu'


def get_dims(generator):
    inp, out = next(generator)
    inp_fea_shape = (None, inp['input_mix_feature'].shape[-1])
    inp_spec_shape = (None, inp['input_mix_spectrum'].shape[-1])
    # 这里output的time_step维度后面没有用到，可以改成None，或删掉均可
    out_shape = out['target_clean_spectrum'].shape[1:]
    out_shape = tuple(out_shape)
    return inp_fea_shape, inp_spec_shape, out_shape


def get_feature(audio_list, min_mix=2, max_mix=2, batch_size=1):
    """
    :param audio_list: 语音文件列表，如下
        path/to/1st.wav spk1
        path/to/2nd.wav spk2
        path/to/3rd.wav spk1
    :param min_mix: 设置最小混叠说话人数
    :param max_mix: 设置最大混叠说话人数
    :param batch_size: 批处理的数据大小
    :return:
    """
    speaker_audios = {}
    batch_input_fea = []
    batch_input_spec = []
    batch_target_spec = []
    batch_target_spec_2 = []
    batch_count = 0
    while True:  # 无限制地生成训练样例，直到当前batch_size的list满了

        #shin:这里的写法好想是只针对一条来的，后面需要载tran_wavlist.txt做相应的改动
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
        target_sig = []
        # 开始随机选择说话人的语音进行混叠，加载到batch数据中
        # 先注释掉随机选择说话人，因为目前的算法有排列问题，后期加入说话人注意之后改为随机选择说话人语音
        # for spk in random.sample(speaker_audios.keys(), mix_k):
        for idx,spk in enumerate(speaker_audios.keys()):
            file_str = speaker_audios[spk].pop()
            if not speaker_audios[spk]:
                del(speaker_audios[spk])  # 删除掉空的说话人
            signal, rate = sf.read(file_str)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            if len(signal) < config.MAX_TRAIN_LEN:  # 根据最大长度补齐
                signal = list(signal)
                signal.extend(np.ones(config.MAX_TRAIN_LEN - len(signal)) * np.spacing(1))
                if config.AUGMENT_DATA:
                    random_shift = random.sample(range(config.MAX_TRAIN_LEN), 1)[0]
                    signal = signal[random_shift:] + signal[:random_shift]
                signal = np.array(signal)
            elif len(signal) > config.MAX_TRAIN_LEN:  # 根据最大长度裁剪
                signal = signal[:config.MAX_TRAIN_LEN]
            signal -= np.mean(signal)  # 语音信号预处理，先减去均值
            signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化
            if wav_mix is None:
                wav_mix = signal
                target_sig = signal  # 混叠前的纯净语音，只保存第一个说话人语音就好
            else:
                # 注意：这里一定不可以改成 wav_mix += signal的形式，target_sig会被修改（地址传递）
                wav_mix = wav_mix + signal  # 混叠后的语音
                if idx==1:
                    target_sig_2=signal


        # 以后可以考虑采用MFCC或GFCC特征做为输入
        feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                     config.FRAME_SHIFT, window=config.WINDOWS)))
        # 混叠的语谱图，用于masking输出
        spec_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                  config.FRAME_SHIFT, window=config.WINDOWS)))

        # 计算纯净语音的语谱图, STFTs for individual signals
        spec_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))
        spec_clean_2 = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig_2, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))

        # 加载到batch缓存中
        batch_input_fea.append(feature_mix)
        batch_input_spec.append(spec_mix)
        batch_target_spec.append(spec_clean)
        batch_target_spec_2.append(spec_clean_2)
        batch_count += 1

        if batch_count == batch_size:
            # 混叠特征，mix_input_fea (batch_size, time_steps, feature_size)
            mix_input_fea = np.array(batch_input_fea).reshape((batch_size, ) + feature_mix.shape)
            # 混叠语谱
            mix_input_spec = np.array(batch_input_spec).reshape((batch_size, ) + spec_mix.shape)
            # 目标语谱，target (batch_size, time_steps, spectrum_size)
            clean_target_spec = np.array(batch_target_spec).reshape((batch_size, ) + spec_clean.shape)
            clean_target_spec_2 = np.array(batch_target_spec_2).reshape((batch_size, ) + spec_clean.shape)
            # 挂起生成器，等待调用next
            yield ({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec},
                   {'target_clean_spectrum': clean_target_spec,'target_clean_spectrum_2':clean_target_spec_2})
            batch_input_fea = []
            batch_input_spec = []
            batch_target_spec = []
            batch_target_spec_2 = []
            batch_count = 0

if __name__ == "__main__":
    # 测试函数
    x, y = next(get_feature(config.TRAIN_LIST, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                            batch_size=config.BATCH_SIZE))
    print (x['input_mix_feature'].shape)
    print (y['target_clean_spectrum'].shape)
