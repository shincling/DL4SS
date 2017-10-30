#coding=utf8
import sys
import numpy as np
import time
import config


def prepare_data():
    if config.MODE==1:
        pass
    elif config.MODE==2:
        pass
    elif config.MODE==3:
        if config.DATASET=='GRID':
            print 'hhh'
        elif config.DATASET=='GRID':
            pass
        else:
            raise ValueError('No such dataset:{} for Video'.format(config.DATASET))

    elif config.MODE==4:
        pass
    else:
        raise ValueError('No such Model:{}'.format(config.MODE))
