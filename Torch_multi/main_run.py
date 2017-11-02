#coding=utf8
import sys
import numpy as np
import time
import config
from predata import prepare_data


# stout=sys.stdout
# log_file=open(config.LOG_FILE_PRE,'w')
# sys.stdout=log_file
# logfile=config.LOG_FILE_PRE



def main():
    print('go to model')
    print '*' * 80

    data_generator=prepare_data('train')




if __name__ == "__main__":
    main()