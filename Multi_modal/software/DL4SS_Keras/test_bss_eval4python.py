# <-*- encoding: utf-8 -*->
import soundfile as sf
import numpy as np
import matlab.engine


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

# 计算SDR, SAR, SIR（前面是true，后面是pred）
wav_mix = matlab.double((ori_male+ori_female).tolist())
ori_male = matlab.double(ori_male.tolist())
# BSS_EVAL (truth, pred, mix)
bss_eval_resuts = mat_eng.BSS_EVAL(ori_male, wav_mix, wav_mix)
sdr = bss_eval_resuts['SDR']
sir = bss_eval_resuts['SIR']
sar = bss_eval_resuts['SAR']
nsdr = bss_eval_resuts['NSDR']
# sdr, sir, sar, popt = bss_eval_sources(ori_male, ori_male)
print 'SDR:', sdr, 'SIR:', sir, 'SAR:', sar, 'NSDR:', nsdr
