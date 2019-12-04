from easydict import EasyDict as edict
import os
import numpy as np

config = edict()
config.anchors = np.array([(6, 24), (9, 35), (14, 52), (21, 80), (37, 128), (81, 272)])
# config.anchors = [(anchor[0]/416, anchor[1]/416) for anchor in anchors]
config.resizeNet = [256, 288, 320, 352, 384, 416, 448, 480]
# config.resizeNet = [320]
config.classes = 1
config.lencell = config.classes + 5
config.lastchannel = config.lencell * config.anchors.shape[0]
config.batchsize = 64
config.imagesize = 416
config.EPS = 1e-14
config.cellsize = 32
# config.centerlamda = 1.0
config.coordlamda = 5.0
config.noobjlamda = 0.5
config.lr = 0.01
config.epochs = [40000, 50000]
config.lr_factor = 0.1
config.num_Process = 5

config.logdir = r'./log'
if not os.path.exists(config.logdir):
    os.mkdir(config.logdir)
config.modelpath = r'./model'
if not os.path.exists(config.modelpath):
    os.mkdir(config.modelpath)

config.threshold = 0.75
config.nms_ths = 0.5
config.targets_per_image = 30
config.train_phase = True
config.test_phase = not config.train_phase


