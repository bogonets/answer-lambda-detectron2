import numpy as np

import sys
import os

from detectron2.config import get_cfg
from detectron2.utils import visualizer as vis
from predictor import VisualizationDemo

weights_file = ''
config_file = ''
conf_threshold = 0.5
cfg = None
visualizer = None
enable_bboxes = True
enable_predictions = True


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    if weights_file:
        cfg.merge_from_list(['MODEL.WEIGHTS', weights_file])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf_threshold
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
        return conf_threshold


def on_init():
    global visualizer
    cfg = setup_cfg()

    visualizer = VisualizationDemo(cfg)
    return True


def on_run(image):

    # sys.stdout.write(f"111 shape~~~~ {image.shape}")
    # sys.stdout.flush()

    predictions, draw_image = visualizer.run_on_image(image)
    draw_image = draw_image.get_image()[:, :, ::-1]

    # sys.stdout.write(f"[detectron2-pose] predictions {predictions}")
    # sys.stdout.flush()

    instances = predictions['instances'].to(visualizer.cpu_device)
    keypoints = instances.pred_keypoints.numpy()

    # sys.stdout.write(f"[detectron2-pose] keypoints {keypoints}")
    # sys.stdout.flush()

    return {'draw_image': draw_image, 'keypoints': keypoints}


