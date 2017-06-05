# -*- coding: utf-8 -*-

import soundfile as sf
import numpy as np
import random
import config
import resampy
import librosa
import matlab


def eval_separation(model, audio_list, epoch_num, log_file, min_mix=2, max_mix=2, batch_size=1):
    speaker_audios = {}
    batch_input_fea = []
    batch_input_spec = []
    batch_mix_spec = []
    batch_mix_wav = []
    batch_target_wav = []
    batch_noise_wav = []
    batch_count = 0
    batch_sdr = []
    batch_sir = []
    batch_sar = []
    batch_nsdr = []
    while True:  # 无限制地生成测试样例，直到当前batch满了
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
        # 记录纯净目标语音
        target_sig = None
        # 记录纯净干扰语音
        noise_sig = None

        # 开始随机选择说话人的语音进行混叠，加载到batch数据中
        # 先注释掉随机选择说话人，因为目前的算法有排列问题，后期加入说话人注意之后改为随机选择说话人语音
        # for spk in random.sample(speaker_audios.keys(), mix_k):
        for spk in speaker_audios.keys():
            file_str = speaker_audios[spk].pop()
            if not speaker_audios[spk]:
                del(speaker_audios[spk])  # 删除掉空的说话人
            signal, rate = sf.read(file_str)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            if len(signal) < config.MAX_TEST_LEN:  # 根据最大长度补齐
                signal = list(signal)
                signal.extend(np.ones(config.MAX_TEST_LEN - len(signal)) * np.spacing(1))
                signal = np.array(signal)
            elif len(signal) > config.MAX_TEST_LEN:  # 根据最大长度裁剪
                signal = signal[:config.MAX_TEST_LEN]
            signal -= np.mean(signal)  # 语音信号预处理，先减去均值
            signal /= np.max(np.abs(signal))  # 波形幅值预处理，幅值归一化
            if wav_mix is None:
                wav_mix = signal
                target_sig = signal  # 混叠前的纯净目标语音，只保存第一个说话人语音就好
            else:
                wav_mix = wav_mix + signal  # 混叠后的语音
                if noise_sig is None:
                    noise_sig = signal
                else:
                    noise_sig = noise_sig + signal  # 混叠前的纯净干扰语音，其他说话人语音混叠为干扰语音

        # 以后可以考虑采用MFCC或GFCC特征做为输入
        feature_mix = np.abs(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                     config.FRAME_SHIFT,
                                                                     window=config.WINDOWS)))
        # 混叠的语谱图，用于masking输出 和 iSTFT变换时提取phase
        spec_mix = np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                           config.FRAME_SHIFT,
                                                           window=config.WINDOWS))

        # 加载到batch缓存中, 用于predict 模型预测
        batch_input_fea.append(feature_mix)
        batch_input_spec.append(np.abs(spec_mix))
        # 加载到batch缓存中，用于BSS_EVAL 评估
        batch_mix_spec.append(spec_mix)
        batch_mix_wav.append(wav_mix)
        batch_target_wav.append(target_sig)
        batch_noise_wav.append(noise_sig)

        batch_count += 1

        if batch_count == batch_size:
            # 混叠特征，mix_input_fea (batch_size, time_steps, feature_size)
            mix_input_fea = np.array(batch_input_fea).reshape((batch_size, ) + feature_mix.shape)
            # 混叠语谱
            mix_input_spec = np.array(batch_input_spec).reshape((batch_size, ) + spec_mix.shape)

            # 进行预测
            target_pred = model.predict({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec})
            batch_idx = 0
            # 对预测结果进行评估
            for _pred_output in list(target_pred):
                _mix_spec = batch_mix_spec[batch_idx]
                phase_mix = np.angle(_mix_spec)
                _pred_spec = _pred_output * np.exp(1j * phase_mix)
                _pred_wav = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT,
                                                        window=config.WINDOWS)
                _target_wav = batch_target_wav[batch_idx]
                # 进行长度补齐, 截断
                if len(_target_wav) != len(_pred_wav):
                    min_len = np.minimum(len(_target_wav), len(_pred_wav))
                    _pred_wav = _pred_wav[:min_len]
                    batch_target_wav[batch_idx] = _target_wav[:min_len]
                    batch_noise_wav[batch_idx] = batch_noise_wav[batch_idx][:min_len]
                    batch_mix_wav[batch_idx] = batch_mix_wav[batch_idx][:min_len]

                # 计算SDR, SAR, SIR（前面是true，后面是pred）
                mix_wav = matlab.double(batch_mix_wav[batch_idx].tolist())
                target_wav = matlab.double(batch_target_wav[batch_idx].tolist())
                noise_wav = matlab.double(batch_noise_wav[batch_idx].tolist())
                pred_wav = matlab.double(_pred_wav.tolist())
                if epoch_num == 0:
                    # BSS_EVAL (truth_signal, truth_noise, pred_signal, mix)
                    bss_eval_resuts = config.MAT_ENG.BSS_EVAL(target_wav, noise_wav, mix_wav, mix_wav)
                    batch_sdr.append(bss_eval_resuts['SDR'])
                    batch_sir.append(bss_eval_resuts['SIR'])
                    batch_sar.append(bss_eval_resuts['SAR'])
                    batch_nsdr.append(bss_eval_resuts['NSDR'])
                    print '[Epoch: %d] - SDR:%f, SIR:%f, SAR:%f, NSDR:%f' % \
                          (epoch_num, batch_sdr[-1], batch_sir[-1], batch_sar[-1], batch_nsdr[-1])
                    log_file.write('[Epoch: %d] - SDR:%f, SIR:%f, SAR:%f, NSDR:%f\n' %
                                   (epoch_num, batch_sdr[-1], batch_sir[-1], batch_sar[-1], batch_nsdr[-1]))
                    log_file.flush()
                # 把预测写到文件中
                sf.write(config.TMP_FOLDER+'/test_pred'+str(epoch_num+1)+'.wav', _pred_wav, config.FRAME_RATE)

                # BSS_EVAL (truth_signal, truth_noise, pred_signal, mix)
                bss_eval_resuts = config.MAT_ENG.BSS_EVAL(target_wav, noise_wav, pred_wav, mix_wav)
                batch_sdr.append(bss_eval_resuts['SDR'])
                batch_sir.append(bss_eval_resuts['SIR'])
                batch_sar.append(bss_eval_resuts['SAR'])
                batch_nsdr.append(bss_eval_resuts['NSDR'])
                print '[Epoch: %d] - SDR:%f, SIR:%f, SAR:%f, NSDR:%f' % \
                      (epoch_num+1, batch_sdr[-1], batch_sir[-1], batch_sar[-1], batch_nsdr[-1])
                log_file.write('[Epoch: %d] - SDR:%f, SIR:%f, SAR:%f, NSDR:%f\n' %
                               (epoch_num+1, batch_sdr[-1], batch_sir[-1], batch_sar[-1], batch_nsdr[-1]))
                log_file.flush()
                batch_idx += 1
                return
