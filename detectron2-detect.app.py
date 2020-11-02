import numpy as np

import sys
import os
import time

from detectron2.config import get_cfg
from predictor import VisualizationDemo
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels
import torch

weights_file = ''
config_file = ''
conf_threshold = 0.5
cfg = None
visualizer = None

gpu = 0
enable_draw_image = False
thing_classes = ''
thing_colors = ''


thing_classes_values = []
thing_colors_values = []


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    # sys.stderr.write(f"[detectron2-detect.setup_cfg] config_file {config_file}\n")
    # sys.stderr.write(f"[detectron2-detect.setup_cfg] config_file {weights_file}\n")
    # sys.stderr.flush()
    cfg.merge_from_file(config_file)
    if weights_file:
        cfg.merge_from_list(['MODEL.WEIGHTS', weights_file])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf_threshold
    cfg.freeze()
    return cfg


def read_data(file_path, cb=None):
    result = []
    with open(file_path, 'r') as f:
        while True:
            buf = f.readline()
            if not buf:
                break
            if cb is not None:
                buf = cb(buf)
            result.append(buf)
    return result


def draw_instance_predictions(vis, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, vis.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, vis.output.height, vis.output.width) for x in masks]
        else:
            masks = None

        # print(f"vis instance_mode : {vis._instance_mode}")
        # print(f"vis metadata.get(thing_colors) : {vis.metadata.get('thing_colors')}")
        # if vis._instance_mode == ColorMode.SEGMENTATION and vis.metadata.get("thing_colors"):
        #     colors = [
        #         [x / 255 for x in thing_colors_values[c]] for c in classes
        #     ]
        #     alpha = 0.8
        # else:
        #     colors = None
        #     alpha = 0.5
        # sys.stderr.write(f"[detectron2-detect.draw_instance_predictions] thing_colors_values : {thing_colors_values}\n")
        # sys.stderr.flush()
        if thing_colors_values:
            colors = []
            for x in classes:
                c = []
                for i in thing_colors_values[x]:
                    c.append(float(i) / 255.0)
                colors.append(c)
        else:
            colors = None
        #    [x / 255 for x in thing_colors_values[c]] for c in classes
        #]
        alpha = 0.5

        # sys.stderr.write(f"[detectron2-detect.draw_instance_predictions] labels : {labels}\n")
        # sys.stderr.write(f"[detectron2-detect.draw_instance_predictions] colors : {colors}\n")
        # sys.stderr.write(f"[detectron2-detect.draw_instance_predictions] thing_colors_values : {thing_colors_values}\n")
        # sys.stderr.flush()

        vis.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return vis.output


def on_set(k, v):
    sys.stderr.write(f"[detectron2-detect.on_set] k ({k}) v ({v})\n")
    sys.stderr.flush()
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
    elif k == 'enable_draw_image':
        global enable_draw_image
        # sys.stderr.write(f"[detectron2-detect.on_set] enable_draw_image: {enable_draw_image}\n")
        # sys.stderr.write(f"[detectron2-detect.on_set] v: {v}\n")
        # sys.stderr.flush()
        enable_draw_image = True if v.lower() in ['true'] else False
    elif k == 'thing_classes':
        global thing_classes
        thing_classes = v
    elif k == 'thing_color':
        global thing_colors
        thing_colors = v


def on_get(k):
    if k == 'weights':
        return weights_file
    elif k == 'config_file':
        return config_file
    elif k == 'conf_threshold':
        return conf_threshold
    elif k == 'gpu':
        return gpu
    elif k == 'enable_draw_image':
        return str(enable_draw_image)
    elif k == 'thing_classes':
        return thing_classes
    elif k == 'thing_color':
        return thing_colors


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
    global thing_classes_values
    global thing_colors_values

    if not set_gpu():
        return False

    cfg = setup_cfg()

    visualizer = VisualizationDemo(cfg)

    # Things setting.
    if thing_classes:
        try:
            thing_classes_values = read_data(thing_classes)

            visualizer.metadata.thing_classes = thing_classes_values
        except:
            pass

    if thing_colors:
        thing_colors_values = read_data(thing_colors, lambda x: [int(i * 255) for i in eval(x)])
        visualizer.metadata.thing_colors = thing_colors_values

    return True


def on_run(image):

    # sys.stderr.write(f"[detectron2-detect.on_run] image.shape : {image.shape}\n")
    # sys.stderr.write(f"[detectron2-detect.on_run] enable_draw_image: {enable_draw_image}\n")
    # sys.stderr.flush()

    predictions = visualizer.predictor(image)

    instances = predictions["instances"].to(visualizer.cpu_device)

    # sys.stdout.write(f"[detectron2-detect.on_run] instances : {instances}\n")
    # sys.stdout.flush()

    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes

    # sys.stderr.write(f"[detectron2-detect.on_run] classes : {classes}\n")
    # sys.stderr.flush()
    scores = scores.reshape(-1,1)
    classes = classes.reshape(-1,1)
    boxes = np.append(boxes, scores, axis=1)
    boxes = np.append(boxes, classes, axis=1)

    # sys.stderr.write(f"{classes}\n")
    # sys.stderr.flush()

    if enable_draw_image:
        vis = Visualizer(image, visualizer.metadata, instance_mode=ColorMode.SEGMENTATION)
        draw_image = draw_instance_predictions(vis, predictions=instances)
        draw_image = draw_image.get_image()[:, :, ::-1]
        # sys.stderr.write(f"[detectron2-detect.on_run] draw_image : {draw_image}\n")
        # sys.stderr.flush()
        return {'draw_image': draw_image, 'boxes': boxes}
    else:
        # sys.stderr.write(f"[detectron2-detect.on_run] boxes : {boxes}\n")
        # sys.stderr.flush()
        return {'draw_image': None, 'boxes': boxes}



