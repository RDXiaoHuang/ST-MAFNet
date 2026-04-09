import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict

from st_mafnet.arch import STMAFNet
from st_mafnet.runner import STMAFNetRunner
from st_mafnet.data import ForecastingDataset
from basicts.losses import masked_mae
from basicts.utils import load_adj

CFG = EasyDict()

CFG.DESCRIPTION = "STMAFNet(PEMS08) configuration"
CFG.RUNNER = STMAFNetRunner
CFG.DATASET_CLS = ForecastingDataset
CFG.DATASET_NAME = "PEMS08"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12

CFG.GPU_NUM = 1

CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True
CFG.ENV.CUDNN.DETERMINISTIC = True
CFG.ENV.CUDNN.BENCHMARK = False

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STMAFNet"
CFG.MODEL.ARCH = STMAFNet
adj_mx, _ = load_adj("../datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")

CFG.MODEL.PARAM = {
    "num_nodes": 170,
    "supports": [torch.tensor(i) for i in adj_mx],
    "in_channels": 1,
    "hidden_dim": 64,
    "adp_dim": 128,
    "skip_channels": 256,
    "out_dim": 12,
    "num_scales": 4,
    "decoder_layers": 2,
    "fusion_layers": 2,
    "dropout": 0.2,
    "node_dim": 64,
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "adaptive_embedding_dim": 64,
    "if_time_in_day": True,
    "if_day_in_week": True,
    "adaptive_nhead": 4, 
    "adaptive_layers": 2,
    "if_adaptive_graph": True,
    "if_forward_graph": False,
    "if_backward_graph": True,
    "if_multi_scale_anchor": True,
    "if_adp_emb": False
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]

CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 1.0e-5,
    "eps": 1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 18, 36, 54, 72],
    "gamma": 0.5
}

CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 300

CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.DATA.DIR = "../datasets/" + CFG.DATASET_NAME
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True

CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "../datasets/" + CFG.DATASET_NAME
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "../datasets/" + CFG.DATASET_NAME
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True
