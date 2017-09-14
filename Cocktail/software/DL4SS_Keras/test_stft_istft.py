# <-*- encoding: utf-8 -*->
import soundfile as sf
import numpy as np
from mir_eval.separation import bss_eval_sources
import librosa
import matlab.engine


def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=1024, overlap=2, window='hanning'):
    """
    Short-time fourier transform.
        x:
        input waveform (1D array of samples)

        fftsize:
        in samples, size of the fft window

        overlap:
        should be a divisor of fftsize, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: linear domain spectrum (2D complex array)
    """
    hop = int(np.round(fftsize / overlap))
    if window == 'hanning':
        w = sqrt_hann(fftsize)  # 默认采用的汉宁窗
    else:
        w = window
    out = np.array([np.fft.rfft(w*x[i:i+fftsize])  # 傅里叶变换直接调用numpy的函数
                    for i in range(0, len(x)-fftsize, hop)])
    return out


def istft(X, overlap=2, window='hanning'):
    """
    Inverse short-time fourier transform.
        X:
        input spectrum (2D complex array)

        overlap: overlap=config.FRAME_LENGTH//config.FRAME_SHIFT
        should be a divisor of (X.shape[1] - 1) * 2, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    hop = int(np.round(fftsize / overlap))
    if window == 'hanning':
        w = sqrt_hann(fftsize)
    else:
        w = np.array(window)
    x = np.zeros(X.shape[0]*hop)
    wsum = np.zeros(X.shape[0]*hop)
    for n, i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x


# 打开matlab引擎
mat_eng = matlab.engine.start_matlab()
# 定义语音文件目录
ori_male_path = './TIMIT/male_test.wav'
ori_female_path = './TIMIT/female_test.wav'

# 加载语音文件数据至内存()
ori_male, rate = sf.read(ori_male_path)  # signal 是采样值，rate 是采样频率
ori_female, rate = sf.read(ori_female_path)  # signal 是采样值，rate 是采样频率

# 用极小值eps补齐长度不一的音频
if len(ori_male) > len(ori_female):
    ori_female = list(ori_female)
    ori_female.extend(np.ones(len(ori_male)-len(ori_female))*0)  # np.spacing(1))
    ori_female = np.array(ori_female)

elif len(ori_male) < len(ori_female):
    ori_male = list(ori_male)
    ori_male.extend(np.ones(len(ori_female)-len(ori_male))*0)  # np.spacing(1))
    ori_male = np.array(ori_male)

# 计算短时傅里叶变换
nFFT = 1024
windows = 1024
hop = 512
windows = [np.sin(x*np.pi/windows) for x in range(windows)]
# ori_male_spectrum = stft(ori_male, nFFT, int(nFFT/hop), window=windows)
ori_male_spectrum = np.transpose(librosa.core.spectrum.stft(ori_male, nFFT, hop, window=windows))
phase_male = np.angle(ori_male_spectrum)
ori_male_spectrum = np.absolute(ori_male_spectrum)

# 开始逆变换

# 加上相位
ori_male_spectrum_phase = ori_male_spectrum * np.exp(1j * phase_male)

# wavout_male_phase = istft(ori_male_spectrum_phase, 2, window=windows)
wavout_male_phase = librosa.core.spectrum.istft(np.transpose(ori_male_spectrum_phase), hop, window=windows)

# 整理长度一致
minlength = np.min(np.array([len(ori_male), len(ori_female), len(wavout_male_phase)]))

# ori_male = ori_male[:minlength]
# ori_female = ori_female[:minlength]
# wavout_male_phase = wavout_male_phase[:minlength]

# sf.write('pred_wav.wav', wavout_male_phase, rate)
# 计算SDR, SAR, SIR（前面是true，后面是pred）
wav_mix = matlab.double((ori_male+ori_female).tolist())
ori_male = matlab.double(ori_male.tolist())
ori_female = matlab.double(ori_female.tolist())
wavout_male_phase = matlab.double(wavout_male_phase.tolist())
# BSS_EVAL (truth, pred, mix)
bss_eval_resuts = mat_eng.BSS_EVAL(ori_male, ori_female, wav_mix, wav_mix)
sdr = bss_eval_resuts['SDR']
sir = bss_eval_resuts['SIR']
sar = bss_eval_resuts['SAR']
nsdr = bss_eval_resuts['NSDR']
# sdr, sir, sar, popt = bss_eval_sources(ori_male, ori_male)
print 'SDR:', sdr, 'SIR:', sir, 'SAR:', sar, 'NSDR:', nsdr
