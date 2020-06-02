import numpy as np

from detectron2.config import get_cfg
from predictor import VisualizationDemo

weights_file = ''
config_file = ''
cfg = None


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.WEIGHTS', weights_file])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def on_set(k, v):
    if k == 'weights':
        global weights_file
        weights_file = v
    elif k == 'config_file':
        global config_file
        config_file = v
    elif k == 'conf_threshold':
        global conf_threshold
        conf_threshold = float(v)


def on_get(k):
    if k == 'weights':
        return weights_file
    elif k == 'config_file':
        return config_file
    elif k == 'conf_threshold':


def on_init():
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


def on_run(image):

    instances = demo.predict_for_instances(img)

