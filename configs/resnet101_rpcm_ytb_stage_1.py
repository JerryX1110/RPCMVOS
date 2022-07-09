import torch
import argparse
import os
import sys
import cv2
import time
import socket
import random

class Configuration():
    def __init__(self):
        self.EXP_NAME = 'resnet101_rpcm_ytb'

        self.EVAL_AUTO_RESUME = False
        self.UNC_RATIO = 1.0
        self.MEM_EVERY = 5
        self.USE_SF = False
        self.PAST_FRAME_NUM = 4
        self.COMPRESSION_RATE = 4
        self.BLOCK_NUM = 2

        self.DIR_ROOT = '/datasets/'
        self.DIR_DAVIS =  os.path.join(self.DIR_ROOT, 'DAVIS')
        self.DIR_YTB =  os.path.join(self.DIR_ROOT, 'MODELS/STCN-ALL/YouTube/train/')
        self.DIR_YTB_EVAL18 =  os.path.join(self.DIR_ROOT, 'MODELS/STCN-ALL/YouTube2018/valid/')
        self.DIR_YTB_EVAL19 =  os.path.join(self.DIR_ROOT,'MODELS/STCN-ALL/YouTube/valid/')
        self.DIR_RESULT = os.path.join(self.DIR_ROOT, 'result', self.EXP_NAME)



        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_TB_LOG = os.path.join(self.DIR_RESULT, 'log', 'tensorboard')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')

        self.DATASETS = ['youtubevos']
        self.DATA_WORKERS = 4
        self.DATA_RANDOMCROP = (465, 465)
        self.DATA_RANDOMFLIP = 0.5
        self.DATA_MAX_CROP_STEPS = 5
        self.DATA_MIN_SCALE_FACTOR = 1.
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_RANDOM_REVERSE_SEQ = True
        self.DATA_DAVIS_REPEAT = 30
        self.DATA_CURR_SEQ_LEN = 3
        self.DATA_RANDOM_GAP_DAVIS = 3
        self.DATA_RANDOM_GAP_YTB = 3


        self.PRETRAIN = True
        self.PRETRAIN_FULL = False
        self.PRETRAIN_MODEL = '' # path to resnet101-deeplabv3p.pth.tar 

        self.MODEL_BACKBONE = 'resnet'
        self.MODEL_MODULE = 'networks.rpcm.rpcm'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SEMANTIC_EMBEDDING_DIM = 100
        self.MODEL_HEAD_EMBEDDING_DIM = 256
        self.MODEL_PRE_HEAD_EMBEDDING_DIM = 64
        self.MODEL_GN_GROUPS = 32
        self.MODEL_GN_EMB_GROUPS = 25
        self.MODEL_MULTI_LOCAL_DISTANCE = [2, 4, 6, 8, 10, 12]
        self.MODEL_LOCAL_DOWNSAMPLE = True
        self.MODEL_REFINE_CHANNELS = 64  # n * 32
        self.MODEL_LOW_LEVEL_INPLANES = 256 if self.MODEL_BACKBONE == 'resnet' else 24
        self.MODEL_RELATED_CHANNELS = 64
        self.MODEL_EPSILON = 1e-5
        self.MODEL_MATCHING_BACKGROUND = True
        self.MODEL_GCT_BETA_WD = True
        self.MODEL_FLOAT16_MATCHING = False
        self.MODEL_FREEZE_BN = True
        self.MODEL_FREEZE_BACKBONE = False

        self.TRAIN_TOTAL_STEPS = 200000
        self.TRAIN_START_STEP = 0
        self.TRAIN_LR = 0.02
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_COSINE_DECAY = False
        self.TRAIN_WARM_UP_STEPS = 1000
        self.TRAIN_WEIGHT_DECAY = 15e-5
        self.TRAIN_POWER = 0.9
        self.TRAIN_GPUS = 8
        self.TRAIN_BATCH_SIZE = 8
        self.TRAIN_START_SEQ_TRAINING_STEPS = self.TRAIN_TOTAL_STEPS / 2
        self.TRAIN_TBLOG = False
        self.TRAIN_TBLOG_STEP = 60
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_IMG_LOG = False
        self.TRAIN_TOP_K_PERCENT_PIXELS = 0.15
        self.TRAIN_HARD_MINING_STEP = self.TRAIN_TOTAL_STEPS / 2
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 20000
        self.TRAIN_MAX_KEEP_CKPT = 8
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_GLOBAL_ATROUS_RATE = 1
        self.TRAIN_LOCAL_ATROUS_RATE = 1
        self.TRAIN_LOCAL_PARALLEL = True
        self.TRAIN_GLOBAL_CHUNKS = 1
        self.TRAIN_DATASET_FULL_RESOLUTION = True


        self.TEST_GPU_ID = 0
        self.TEST_DATASET = 'youtubevos'
        self.TEST_DATASET_FULL_RESOLUTION = True
        self.TEST_DATASET_SPLIT = ['val']
        self.TEST_CKPT_PATH = None
        self.TEST_CKPT_STEP = None  # if "None", evaluate the latest checkpoint.
        self.TEST_FLIP = False
        self.TEST_MULTISCALE = [1]
        self.TEST_MIN_SIZE = None
        self.TEST_MAX_SIZE = 800 * 1.3 if self.TEST_MULTISCALE == [1.] else 800
        self.TEST_WORKERS = 4
        self.TEST_GLOBAL_CHUNKS = 4
        self.TEST_GLOBAL_ATROUS_RATE = 1
        self.TEST_LOCAL_ATROUS_RATE = 1
        self.TEST_LOCAL_PARALLEL = True

        # dist
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "nccl"

        myname = socket.getfqdn(socket.gethostname(  ))
        myaddr = socket.gethostbyname(myname)

        self.DIST_URL = "tcp://"+myaddr+":"+str(random.randint(30000,50000))
        self.DIST_START_GPU = 0

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
                raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
                raise ValueError('config.py: the number of GPU is 0')
        for path in [self.DIR_RESULT, self.DIR_CKPT, self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG, self.DIR_TB_LOG]:
            if not os.path.isdir(path):
                os.makedirs(path)



cfg = Configuration()
