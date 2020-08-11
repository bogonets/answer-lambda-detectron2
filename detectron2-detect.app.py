import numpy as np

import sys
import os
import time

from detectron2.config import get_cfg
from predictor import VisualizationDemo
import torch

weights_file = ''
config_file = ''
conf_threshold = 0.5
cfg = None
visualizer = None

gpu = 0


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
    elif k == 'gpu':
        global gpu
        gpu = int(v)


def on_get(k):
    if k == 'weights':
        return weights_file
    elif k == 'config_file':
        return config_file
    elif k == 'conf_threshold':
        return conf_threshold
    elif k == 'gpu':
        return gpu


def print_gpu_info(select, count, current):
    sys.stdout.write("----------------------------\n")
    sys.stdout.write(f"GPU{select} / {count} - {current} / {torch.cuda.get_device_name(current)}\n")
    sys.stdout.write(f"GPU name 0 - {torch.cuda.get_device_name(0)}\n")
    sys.stdout.write(f"GPU name 1 - {torch.cuda.get_device_name(1)}\n")
    sys.stdout.write(f"GPU name 2 - {torch.cuda.get_device_name(2)}\n")
    sys.stdout.write("----------------------------\n")
    sys.stdout.flush()


def set_gpu():
    if not torch.cuda.is_available():
        sys.stderr.write(f"[detectron2-detect.on_init] Pytorch Cuda is not available!.")
        sys.stderr.flush()
        return True

    device_count = torch.cuda.device_count()
    device_index = torch.cuda.current_device()

    if device_count <= gpu:
        sys.stderr.write(f"[detectron2-detect.on_init] Gpu's index is not available. (all gpus: {device_count}, select gpu: {gpu})")
        sys.stderr.flush()
        return True

    # Set GPU.
    torch.cuda.set_device(gpu)

    is_set = False
    for i in range(10):
        device_count = torch.cuda.device_count()
        device_index = torch.cuda.current_device()
        # print_gpu_info(gpu, device_count, device_index)
        if gpu == device_index:
            is_set = True
            time.sleep(2)
            device_index = torch.cuda.current_device()
            sys.stdout.write("----------------------------\n")
            sys.stdout.write(f"GPU SELECTED! - {gpu} == {device_index}\n")
            sys.stdout.write("----------------------------\n")
            sys.stdout.flush()
            break
        torch.cuda.set_device(gpu)

    return is_set


def on_init():
    global visualizer

    if not set_gpu():
        return False

    cfg = setup_cfg()

    visualizer = VisualizationDemo(cfg)
    return True


def on_run(image):

    #sys.stdout.write(f"111 shape~~~~ {image.shape}")
    #sys.stdout.flush()

    predictions = visualizer.predictor(image)

    #sys.stdout.write(f"{instances}")
    #sys.stdout.flush()

    instances = predictions["instances"].to(visualizer.cpu_device)

    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes

    scores = scores.reshape(-1,1)
    classes = classes.reshape(-1,1)
    boxes = np.append(boxes, scores, axis=1)
    boxes = np.append(boxes, classes, axis=1)

    #sys.stdout.write(f"{boxes}")
    #sys.stdout.flush()

    return {'bboxes': boxes}


