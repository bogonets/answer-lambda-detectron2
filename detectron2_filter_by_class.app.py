import numpy as np
# import sys

classes = []
names_file = 'coco.names'
names = {}


def read_class_names(class_file_names):
    names_dict = {}
    with open(class_file_names, 'r') as data:
        for ID, name in enumerate(data):
            names_dict[ID] = name.strip('\n')
    return names_dict


def on_set(key, val):
    if key == 'classes':
        global classes
        classes = val.split(',')
    elif key == 'names_file':
        global names_file
        global names
        names_file = val
        names = read_class_names(names_file)


def on_get(key):
    if key == 'classes':
        return ','.join(classes)
    elif key == 'names_file':
        return names_file


def on_run(bboxes):
    # sys.stdout.write(f"[detectron2_filter] bboxes {bboxes} {type(bboxes)}\n")
    # sys.stdout.write(f"[detectron2_filter] bboxes.shape {bboxes.shape} {type(bboxes.shape)}\n")
    # sys.stdout.write(f"[detectron2_filter] bboxes.size {bboxes.size} {type(bboxes.size)}\n")
    # sys.stdout.flush()
    if not bboxes.shape or bboxes.size == 0:
        return {
            'filtered_bboxes': None,
            'remain': None
        }
    filtered_bboxes = []
    remain = []
    for b in bboxes:
        n = names[b[5]]
        if n in classes:
            filtered_bboxes.append(b)
        else:
            remain.append(b)

    # sys.stdout.write(f"[detectron2_filter] filterd_bboxes {filtered_bboxes} {type(filtered_bboxes)}\n")
    # sys.stdout.write(f"[detectron2_filter] remain {remain} {type(remain)}\n")
    # sys.stdout.flush()

    if filtered_bboxes:
        return {
            'filtered_bboxes': np.array(filtered_bboxes),
            'remain': np.array(remain)
        }
    else:
        return {
            'filtered_bboxes': None,
            'remain': np.array(remain)
        }
