## @package ginop
#  Create dataset from images
#
#  This module was used to create the modified dataset from the DAVIS dataset
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python create_dataset_i.py -h for help

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Create Dataset From Image Sequences')
    parser.add_argument(
        '-sdir', dest='sdir',
        help='Directory relative to cwd containing image sequences. Default is: DAVIS',
        default='DAVIS', type=str)
    parser.add_argument(
        '-tdir', dest='tdir',
        help='Directory relative to cwd for containing the dataset. Default is: data',
        default='data', type=str)
    parser.add_argument(
        '-s', dest='s',
        help='Target size for images in the dataset (s x s). Default is: 299',
        default=299, type=int)

    args = parser.parse_args()
    return args

## Create dataset
#
#  This function is used for crafting the optical flow inputs nad the masks from image sequences.
#  The inputs and masks are stored in separate .npy files in each subdirectory.
#  The name of directories for the masks and the images have to match.
#  @param args Object for passing the options of the dataset. Use the DS class of utils module.
def formulate_inputs_masks(args):

    import utils

    d = args.sdir
    data_dirs = os.listdir(d)
    for directory in data_dirs:
        onlyfiles = utils.onlyfiles(os.path.join(d,directory))
        onlyfiles = [f for f in onlyfiles if f.split(os.sep)[-1] != 'inputs.npy' and f.split(os.sep)[-1] != 'masks.npy']
        c = 0
        for file in onlyfiles:
            if c == 0:
                gray = cv2.imread(file, cv2.IMREAD_COLOR)
                ih,iw,ic = gray.shape
                gray=gray[:,int((iw-ih)/2):int((iw+ih)/2),:]
                gray = cv2.resize(gray, (args.s,args.s))
                imseq = np.array([gray])
                inputs = np.array([])
            else:
                prevgray = cv2.cvtColor(imseq[-1], cv2.COLOR_BGR2GRAY)
                gray=gray[:,int((iw-ih)/2):int((iw+ih)/2),:]
                gray = cv2.resize(gray, (args.s,args.s))
                imseq = np.append(imseq, [gray], axis=0)
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
                flow_normalized = utils.normalize_flow(flow)
                gray = (gray-gray.min())/(gray.max()-gray.min())
                gray = np.reshape(gray,(args.s,args.s,1))
                conc_img = np.concatenate((flow_normalized,gray), axis=2)
                if c == 1:
                    inputs = np.array([conc_img])
                else:
                    inputs = np.append(inputs, [conc_img], axis=0)
            gray = cv2.imread(file, cv2.IMREAD_COLOR)
            c += 1
        np.save(d+'/%s/inputs.npy' % directory, inputs)
        print('saved inputs to '+ d + '/%s/inputs.npy'  % directory)

    for directory in data_dirs:
        onlyfiles = utils.onlyfiles(os.path.join(args.tdir,directory))
        onlyfiles = [f for f in onlyfiles if f.split(os.sep)[-1] != 'inputs.npy' and f.split(os.sep)[-1] != 'masks.npy']
        c = 0
        for file in onlyfiles:
            if c == 0:
                mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                ih,iw = mask.shape
                mask=mask[:,int((iw-ih)/2):int((iw+ih)/2)]
                mask = cv2.resize(mask, (args.s,args.s))
                masks = np.array([mask])
            else:
                mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                mask=mask[:,int((iw-ih)/2):int((iw+ih)/2)]
                mask = cv2.resize(mask, (args.s,args.s))
                masks = np.append(masks, [mask], axis=0)
            c += 1
        masks = (masks/255).astype(int)
        np.save(d+'/%s/masks.npy' % directory, masks)
        print('saved masks to' + d+ '/%s/masks.npy' % directory)



if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()
    formulate_inputs_masks(args)