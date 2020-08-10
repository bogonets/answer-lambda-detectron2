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

    if not bboxes.shape:
        return {}

    if bboxes.shape[0] <= min_count:
        return {}

    return {'count': np.array([bboxes.shape[0]])}

