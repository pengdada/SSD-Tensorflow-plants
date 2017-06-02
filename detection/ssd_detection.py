import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

import sys
sys.path.append('../')

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from detection import visualization

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()

ssd_net.params = ssd_net.params._replace(num_classes = 22, no_annotation_label = 22)

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
#ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = '../checkpoints/ckp-22/SSD_300x300_iter_120000.ckpt'
#ckpt_filename = '../checkpoints/model.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=ssd_net.params.num_classes, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

# Test on some demo image and visualize output.
#path = '../demo/'
#path="/home/peng/data/VOCdevkit/VOCcrest/JPEGImages/"
path="/media/peng/DATADRIVE2/LPixel/000001.jpg/Alaska/"
image_names = sorted(os.listdir(path))

def get_multi_img(img, start_scale=1, end_scale=0.0625, scale=0.8, step=150, net_shape=(300, 300)):
    shape = img.shape
    sub_pos = []

    k = 0
    while True:
        _scale = start_scale*math.pow(scale, k)
        if (_scale < end_scale): break
        k = k + 1
        _step   = int(step/_scale)
        _shape  = [int(shape[0]/_scale), int(shape[1]/_scale), shape[2]]
        _net_shape = [int(net_shape[0]/_scale), int(net_shape[1]/_scale)]

        ry = range(0, shape[0]-_net_shape[0], _step)
        rx = range(0, shape[1]-_net_shape[1], _step)
        rh = []
        rw = []
        for i in range(len(ry)):
            if (i == len(ry)-1): rh.append(shape[0] - ry[i])
            else:                rh.append(_net_shape[0])

        for j in range(len(rx)):
            if (j == len(rx)-1): rw.append(shape[1] - rx[j])
            else:                rw.append(_net_shape[1])

        for i in range(len(ry)):
            for j in range(len(rx)):
                y, x, h, w = ry[i], rx[j], rh[i], rw[j]
                sub_pos.append((y, x, h, w))
    return sub_pos



def process_multi_img(img, start_scale=1, end_scale=0.0625, scale=0.8, step=150, net_shape=(300, 300)):
    sub_pos = get_multi_img(img, start_scale, end_scale, scale, step, net_shape)
    result = []
    for (y, x, h, w) in sub_pos:
        _img = img[y:y + h, x:x + w, ...]
        #cv2.imshow("img", _img)
        #cv2.waitKey(300)
        rclasses, rscores, rbboxes = process_image(_img, select_threshold=0.5, nms_threshold=.45, net_shape=net_shape)
        for i in range(len(rclasses)):
            result.append((rclasses[i], rscores[i], rbboxes[i]))
        visualization.plt_bboxes(_img, rclasses, rscores, rbboxes)


for i, _ in enumerate(image_names):
    #img = mpimg.imread(path + image_names[-1])
    #img = mpimg.imread(path + image_names[i])
    img = cv2.imread(path + image_names[i])
    img = img[...,::-1]
    #process_multi_img(img, start_scale=0.5, end_scale=0.5, scale=0.5)
    rclasses, rscores, rbboxes =  process_image(img)
    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    img = img[...,::-1]
    visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes)
