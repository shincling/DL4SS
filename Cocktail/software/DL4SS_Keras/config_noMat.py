# <-*- encoding:utf8 -*->

"""
    Configuration Profile
"""

import time
import ConfigParser
import soundfile as sf
import resampy
# import matlab.engine
import numpy as np
from scipy import signal

__author__ = 'jacoxu'

# 判断是否加载
HAS_INIT_CONFIG = False
MAT_ENG = []
# External configuration file
CONFIG_FILE = './config.cfg'
# 日志记录，Record log into this file, such as dl4ss_output.log_20170303_110305
LOG_FILE_PRE = './dl4ss_output.log'
# 数据集 THCHS-30 或者 WSJ0, TIMIT做为模型调试
DATASET = 'TIMIT'
# 训练语音文件列表
TRAIN_LIST = './train_wavlist_'+DATASET
# 验证语音文件列表
VALID_LIST = './valid_wavlist_'+DATASET
# 测试语音文件列表
TEST_LIST = './test_wavlist_'+DATASET
# 未登录语音列表
UNK_LIST = './unk_wavlist_'+DATASET
# DNN/RNN隐层的维度 hidden units
HIDDEN_UNITS = 16
# DNN/RNN层数
NUM_LAYERS = 1
# Embedding大小
EMBEDDING_SIZE = 20
# 是否丰富数据
AUGMENT_DATA = False
# set the max epoch of training
MAX_EPOCH = 5
# epoch size
EPOCH_SIZE = 20
# batch size
BATCH_SIZE = 2
# 评估的batch size
BATCH_SIZE_EVAL = 10
# feature frame rate
FRAME_RATE = 8000
# 帧时长(ms)
FRAME_LENGTH = int(0.032 * FRAME_RATE)
# 帧移(ms)
FRAME_SHIFT = int(0.016 * FRAME_RATE)
# 是否shuffle_batch
SHUFFLE_BATCH = True
# 设定最小混叠说话人数，Minimum number of mixed speakers for training
MIN_MIX = 2
# 设定最大混叠说话人数，Maximum number of mixed speakers for training
MAX_MIX = 2
# 设置训练/开发/验证模型的最大语音长度(秒)
MAX_LEN = 5
# 帧长
WINDOWS = FRAME_LENGTH
# 临时文件的输出目录
TMP_PRED_WAV_FOLDER = '_tmp_pred_wavs'
# 训练模型权重的目录
TMP_WEIGHT_FOLDER = '_tmp_weights'
# 未登录说话人, False: 由说话人提取记忆，True: 进行语音抽取，# TODO NONE: 有相似度计算确定是否进行语音更新
# 更新，UNK_SPK用spk idx=0替代
UNK_SPK = False
# 未登录说话人语音的最大额外条数
UNK_SPK_SUPP = 10
# 特征Spectral of Log Spectral
IS_LOG_SPECTRAL = False


def update_max_len(file_path_list, max_len):
    tmp_max_len = 0
    # 用于搜集不同语音的长度
    signal_set = set()
    for file_path in file_path_list:
        file_list = open(file_path)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 2:
                print 'Wrong audio list file record in the line:', line
                continue
            file_str = line[0]
            if file_str in signal_set:
                continue
            signal_set.add(file_str)
            signal, rate = sf.read(file_str)  # signal 是采样值，rate 是采样频率
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != FRAME_RATE:
                # 如果频率不是设定的频率则需要进行转换
                signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_fast')
            if len(signal) > tmp_max_len:
                tmp_max_len = len(signal)
        file_list.close()
    if tmp_max_len < max_len:
        max_len = tmp_max_len
    return max_len


def init_config():
    global HAS_INIT_CONFIG
    if HAS_INIT_CONFIG:
        raise Exception("This config has been initialized")
    # 打开matlab引擎
    global MAT_ENG
    # MAT_ENG = matlab.engine.start_matlab()
    print 'has opened the matlab engine'
    _config = ConfigParser.ConfigParser()
    cfg_file = open(CONFIG_FILE, 'r')
    _config.readfp(cfg_file)
    global LOG_FILE_PRE
    LOG_FILE_PRE = _config.get('cfg', 'LOG_FILE_PRE').strip()
    global DATASET
    DATASET = _config.get('cfg', 'DATASET')
    global TRAIN_LIST
    TRAIN_LIST = _config.get('cfg', 'TRAIN_LIST') + DATASET
    global VALID_LIST
    VALID_LIST = _config.get('cfg', 'VALID_LIST') + DATASET
    global TEST_LIST
    TEST_LIST = _config.get('cfg', 'TEST_LIST') + DATASET
    global UNK_LIST
    UNK_LIST = _config.get('cfg', 'UNK_LIST') + DATASET
    global HIDDEN_UNITS
    HIDDEN_UNITS = eval(_config.get('cfg', 'HIDDEN_UNITS'))
    global NUM_LAYERS
    NUM_LAYERS = eval(_config.get('cfg', 'NUM_LAYERS'))
    global EMBEDDING_SIZE
    EMBEDDING_SIZE = eval(_config.get('cfg', 'EMBEDDING_SIZE'))
    if EMBEDDING_SIZE % 2 != 0:
        raise Exception('Embedding size should be even integer.')
    global AUGMENT_DATA
    AUGMENT_DATA = eval(_config.get('cfg', 'AUGMENT_DATA'))
    global MAX_EPOCH
    MAX_EPOCH = eval(_config.get('cfg', 'MAX_EPOCH'))
    global EPOCH_SIZE
    EPOCH_SIZE = eval(_config.get('cfg', 'EPOCH_SIZE'))
    global BATCH_SIZE
    BATCH_SIZE = eval(_config.get('cfg', 'BATCH_SIZE'))
    global BATCH_SIZE_EVAL
    BATCH_SIZE_EVAL = eval(_config.get('cfg', 'BATCH_SIZE_EVAL'))
    global FRAME_RATE
    FRAME_RATE = eval(_config.get('cfg', 'FRAME_RATE'))
    global FRAME_LENGTH
    FRAME_LENGTH = int(eval(_config.get('cfg', 'FRAME_LENGTH')) * FRAME_RATE)
    global FRAME_SHIFT
    FRAME_SHIFT = int(eval(_config.get('cfg', 'FRAME_SHIFT')) * FRAME_RATE)
    global SHUFFLE_BATCH
    SHUFFLE_BATCH = eval(_config.get('cfg', 'SHUFFLE_BATCH'))
    global MIN_MIX
    MIN_MIX = eval(_config.get('cfg', 'MIN_MIX'))
    global MAX_MIX
    MAX_MIX = eval(_config.get('cfg', 'MAX_MIX'))
    global MAX_LEN
    MAX_LEN = int(eval(_config.get('cfg', 'MAX_LEN')) * FRAME_RATE)
    MAX_LEN = update_max_len([TRAIN_LIST, VALID_LIST, TEST_LIST, UNK_LIST], MAX_LEN)
    # 计算混叠语音的语谱图
    global WINDOWS
    win_size = FRAME_LENGTH
    # 正弦窗
    WINDOWS = [np.sin(x_i*np.pi/win_size) for x_i in range(win_size)]
    # square root of hanning window
    # WINDOWS = np.sqrt(signal.get_window('hann', win_size))
    # hanning window
    # WINDOWS = signal.get_window('hann', win_size)
    global TMP_PRED_WAV_FOLDER
    TMP_PRED_WAV_FOLDER = _config.get('cfg', 'TMP_PRED_WAV_FOLDER').strip()
    global TMP_WEIGHT_FOLDER
    TMP_WEIGHT_FOLDER = _config.get('cfg', 'TMP_WEIGHT_FOLDER').strip()
    global UNK_SPK
    UNK_SPK = eval(_config.get('cfg', 'UNK_SPK').strip())
    global UNK_SPK_SUPP
    UNK_SPK_SUPP = eval(_config.get('cfg', 'UNK_SPK_SUPP'))
    cfg_file.close()


def log_config(_log_file):
    _log_file.write('*' * 80 + '\n')
    _log_file.write('Current time:' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    _log_file.write('DATASET:' + str(DATASET) + '\n')
    _log_file.write('TRAIN_LIST:' + str(TRAIN_LIST) + '\n')
    _log_file.write('VALID_LIST:' + str(VALID_LIST) + '\n')
    _log_file.write('TEST_LIST:' + str(TEST_LIST) + '\n')
    _log_file.write('UNK_LIST:' + str(UNK_LIST) + '\n')
    _log_file.write('HIDDEN_UNITS:' + str(HIDDEN_UNITS) + '\n')
    _log_file.write('NUM_LAYERS:' + str(NUM_LAYERS) + '\n')
    _log_file.write('EMBEDDING_SIZE:' + str(EMBEDDING_SIZE) + '\n')
    _log_file.write('AUGMENT_DATA:' + str(AUGMENT_DATA) + '\n')
    _log_file.write('MAX_EPOCH:' + str(MAX_EPOCH) + '\n')
    _log_file.write('EPOCH_SIZE:' + str(EPOCH_SIZE) + '\n')
    _log_file.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
    _log_file.write('BATCH_SIZE_EVAL:' + str(BATCH_SIZE_EVAL) + '\n')
    _log_file.write('FRAME_RATE:' + str(FRAME_RATE) + '\n')
    _log_file.write('FRAME_LENGTH:' + str(FRAME_LENGTH) + '\n')
    _log_file.write('FRAME_SHIFT:' + str(FRAME_SHIFT) + '\n')
    _log_file.write('SHUFFLE_BATCH:' + str(SHUFFLE_BATCH) + '\n')
    _log_file.write('MIN_MIX:' + str(MIN_MIX) + '\n')
    _log_file.write('MAX_MIX:' + str(MAX_MIX) + '\n')
    _log_file.write('MAX_LEN:' + str(MAX_LEN) + '\n')
    _log_file.write('TMP_PRED_WAV_FOLDER:' + str(TMP_PRED_WAV_FOLDER) + '\n')
    _log_file.write('TMP_WEIGHT_FOLDER:' + str(TMP_WEIGHT_FOLDER) + '\n')
    _log_file.write('UNK_SPK:' + str(UNK_SPK) + '\n')
    _log_file.write('UNK_SPK_SUPP:' + str(UNK_SPK_SUPP) + '\n')
    _log_file.write('*' * 80 + '\n')
    _log_file.flush()
