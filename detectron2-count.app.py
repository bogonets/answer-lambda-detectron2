import numpy as np

import sys
import os


min_count = 0


def on_set(k, v):
    if k == 'min_count':
        global min_count
        min_count = int(v)


def on_get(k):
    if k == 'min_count':
        return str(min_count)


def on_init():
    return True


def on_run(bboxes):

    # sys.stderr.write(f"[detectron2-count.on_run] start1\n")
    # sys.stderr.flush()

    if not bboxes.shape:
        return {'count': None}

    # sys.stderr.write(f"[detectron2-count.on_run] start2\n")
    # sys.stderr.flush()

    if bboxes.shape[0] < min_count:
        return {'count': None}

    # sys.stderr.write(f"[detectron2-count.on_run] start3\n")
    # sys.stderr.flush()


    return {'count': str(bboxes.shape[0])}

