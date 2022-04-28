import cv2 
import ray
import os
import numpy as np
import json
from src.utils.dirs import create_dirs
from time import time

def get_video_writer(output_file, width = 1920, height = 1080, fps = 25):
    
    """ 
    Return video writer 

    Parameters
    ----------
    output_path : str
        A string of the entire path with filename to write the video to. 
    width : int
        An integer which specifies the width of the output video
    height : int
        An integer which specifies the heigth of the output video
    Returns
    -------
    Video writer

    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    return out


@ray.remote
def save_image(save_dir, name, im):

    cv2.imwrite(os.path.join(save_dir, str(name) + '.png'), im)    
    

def save_images(base_dir, names, images, iterno = None):

    if iterno is None:
        save_dir = os.path.join(base_dir, "images")
    else:
        save_dir = os.path.join(base_dir, str(iterno), "images")

    create_dirs(save_dir)
    obj_ids = []

    for ii, idx in enumerate(names):

        obj_id = save_image.remote(save_dir, idx, images[ii])
        obj_ids.append(obj_id)

    ray.get(obj_ids)

    return


@ray.remote
def save_json(save_dir, name, dets, im_info, num_objs):

    bboxes = []
    for kk in range(len(dets)):

        bbox = dets[kk][0:4]
        classs = int(dets[kk][4])

        bbox = [int(np.round(x.item())) for x in bbox]
        bbox.append(classs)
        bboxes.append(bbox)

    im_info = [int(im_info[0]), int(im_info[1]), int(im_info[2])]  

    data = {}
    data["imInfo"] = im_info
    data["gt_boxes"] = bboxes

    if num_objs is not None:
        data["num_boxes"] = int(num_objs)
    else:
        data["num_boxes"] = len(bboxes)

    with open(os.path.join(save_dir, name + '.json'), 'w') as outfile:
        json.dump(data, outfile, indent=4)

def save_jsons(base_dir, dets, names, im_info, num_objs=None, iterno=None):

    if iterno is None:
        save_dir = os.path.join(base_dir, "json")
    else:
        save_dir = os.path.join(base_dir, str(iterno), "json")

    create_dirs(save_dir)

    obj_ids = []

    for ii, idx in enumerate(names):

        if num_objs is not None:
            obj_id = save_json.remote(save_dir, idx, dets[ii], im_info[ii], num_objs[ii])
        else:
            obj_id = save_json.remote(save_dir, idx, dets[ii], im_info[ii], num_objs)

        obj_ids.append(obj_id)

    ray.get(obj_ids)

    return

@ray.remote
def save_box_image(classes, box_dir, save_dir, idx, im, gt_boxes, dets, rois):

    cv2.imwrite(os.path.join(save_dir, str(idx) + '.png'), im)

    for jj in range(gt_boxes.shape[0]):
        bbox = gt_boxes[jj]
        bbox = [int(np.round(x.item())) for x in bbox]

        if int(bbox[4]) == 0:
            continue

        cv2.putText(
            im, '%s' % (classes[int(bbox[4])]),
            (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0,
            (0, 255, 255), thickness=1)
        cv2.rectangle(
            im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

    for kk in range(len(dets)):
        bbox = dets[kk][0:4]
        classs = dets[kk][4]
        score = dets[kk][5]

        if score > 0.20:
            bbox = [int(np.round(x.item())) for x in bbox]
            cv2.putText(im, '%s' % (str(classes[int(classs)]) + "_" + str(score)),
                        (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 255), thickness=1)

            cv2.rectangle(im, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (255, 0, 0), 1)

    for jj in range(rois.shape[0]):
        bbox = rois[jj]
        bbox = [int(np.round(x.item())) for x in bbox]

        cv2.rectangle(im, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(box_dir, str(idx) + '.png'), im)

def save_box_images(classes, base_dir, save, images, ids, rois, dets, gt_boxes, iterno=None):

    if iterno is None:
        box_dir = os.path.join(base_dir, "boxes")
        save_dir = os.path.join(base_dir, "images")
    else:
        box_dir = os.path.join(base_dir, str(iterno), "boxes")
        save_dir = os.path.join(base_dir, str(iterno), "images")

    create_dirs(save_dir)
    create_dirs(box_dir)

    obj_ids = []

    for ii, idx in enumerate(ids):

        if not save[ii]:
            continue

        obj_id = save_box_image.remote(classes, box_dir, save_dir, idx, images[ii], gt_boxes[ii], dets[ii], rois[ii])
        obj_ids.append(obj_id)

    ray.get(obj_ids)

    return
