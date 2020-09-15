## @package ginop
#  Test trained network
#
#  This module can be used to test the trained networks
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python test.py -h for help

import tensorflow as tf
import numpy as np
import os
import math
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception

import cv2 as cv
import sys



## Data augmentation
#
#  This function was used for data augmentation.
#  @param image a tf.Tensor, the input images.
#  @param mask a tf.Tensor, the masks, corresponding to the images.
def augment(image, mask):
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    mask = tf.reshape(mask,[mask.shape[0],mask.shape[1],1])
    conc = tf.image.random_flip_left_right(tf.concat([image,mask], 2))
    image, mask = tf.split(conc, [3,1], axis=2)
    return image, mask

## Video capture
#
#  This function captures frames with the webcam
#  @param video_filename Path to video file, or 0. If 0, the webcam is used as a source.
def get_cap(video_filename):
    cap = cv.VideoCapture(video_filename)
    global show_hsv
    show_hsv = False
    return cap

## Visualize flow
#
# This function can be used for visualizing optical flow
#  @param gray np.array, grayscale image
#  @param flow np.array, optical flow
#  @param name string, name for the pop-up window
def visualize_flow(gray, flow=None, name='mask'):
    cv.imshow(name, gray)
    global show_hsv
    if show_hsv:
        cv.imshow('flow HSV', draw_hsv(flow))
    ch = cv.waitKey(5)
    if ch == 27:
        return False
    if ch == ord('1'):
        show_hsv = not show_hsv
        print('HSV flow visualization is', ['off', 'on'][show_hsv])
    return True

## Convert image to grayscale
#
#  This function converts a caption to grayscale image
#  @param cap opencv caption
def get_grayImage(cap):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

## Draw flow
#
#  Helper function to visualize optical flow
def draw_flow(img, flow, step=6):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Test the neural network model')
    parser.add_argument(
        '-sdir', dest='sdir',
        help='Directory relative to cwd containing saved model files. Default is: stage3_final',
        default='stage3_final', type=str)
    parser.add_argument(
        '-tdir', dest='tdir',
        help='Name of the model metadata file. Default is: model_final.meta',
        default='model_final.meta', type=str)
    parser.add_argument(
        '-s', dest='s',
        help='Size of input images (s x s). Default is: 299',
        default=299, type=int)

    args = parser.parse_args()
    return args


## Run test
#
#  This function loads the model and displays the raw and tresholded predictions
#  as well as the raw camera image in three separate windows.
#  @param args Object for passing the options of the model. Use the DS class of ginop.utils module.
def run_test(args):

    import ginop

    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(os.path.abspath('__file__')),args.sdir,args.tdir))
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.path.dirname(os.path.abspath('__file__')),args.sdir)))
        print('Model restored')    
        graph = tf.get_default_graph() 

        iterator_init = graph.get_operation_by_name('iterator_init_op')
        valid_init = graph.get_operation_by_name('valid_init_op')
        train_step = graph.get_operation_by_name('Fine_tuning/train_op')
        mask = graph.get_tensor_by_name('Fine_tuning/Th_prediction:0')
        pred = graph.get_tensor_by_name('Fine_tuning/Prediction:0')
        loss = graph.get_tensor_by_name('Fine_tuning/Loss:0')
        labels = graph.get_tensor_by_name('Labels:0')    
        loss_weight = graph.get_tensor_by_name('Fine_tuning/Loss_weight:0')
        inputs = graph.get_tensor_by_name('Inputs:0')
        avg_valid = graph.get_tensor_by_name('avg_valid:0')

        cap = get_cap(0)
        prevgray = get_grayImage(cap)
        ih,iw = prevgray.shape
        prevgray=prevgray[:,int((iw-ih)/2):int((iw+ih)/2)]
        prevgray = cv.resize(prevgray, (args.s,args.s))
        count = 0
        while True:
            gray = get_grayImage(cap)
            gray=gray[:,int((iw-ih)/2):int((iw+ih)/2)]
            gray = cv.resize(gray, (args.s,args.s))
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
            prevgray = gray
            flow_normalized = ginop.utils.normalize_flow(flow)
            gray = (gray-gray.min())/(gray.max()-gray.min())
            gray = np.reshape(gray,(args.s,args.s,1))
            conc_img = np.concatenate((flow_normalized,gray), axis=2)
            p,m = sess.run([pred,mask], feed_dict={inputs: [conc_img]})
            viz = cv.resize(np.reshape(m,(30,30)), (args.s,args.s))
            viz_2 = cv.resize(np.reshape(p,(30,30)), (args.s,args.s))
            visualize_flow(viz)
            visualize_flow(viz_2, name='pred')
            visualize_flow(gray, name='imgs')

if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()
    run_test(args)