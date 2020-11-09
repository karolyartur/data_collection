## @package data_collection
#  Create dataset from videos
#
#  This module was used to create the dataset from the recorded video sequences
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python create_dataset.py -h for help

import os
import cv2
import sys
import numpy as np
from importlib import reload

## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Create Dataset From Video Sequences')
    parser.add_argument(
        '-sdir', dest='sdir',
        help='Directory relative to cwd containing video sequences. Default is: vids',
        default='vids', type=str)
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
#  This function is used for creating a dataset from video sequences
#  @param args Object for passing the options of the dataset. Use the DS class of utils module.
def create_dataset(args):

    import utils
    import videoseg.src.utils as nlc_utils
    import videoseg.src.run_full_mod as nlc

    if nlc:
        reload(nlc)

    vids = []
    dirs = []

    onlyfiles = utils.onlyfiles(args.sdir)

    for i in range(len(onlyfiles)):
        directory = args.tdir + '/' + (onlyfiles[i].split(os.sep)[-1].split('.')[0])
        if not os.path.exists(directory):
            os.makedirs(os.path.join(directory,'imgs'))
            os.makedirs(os.path.join(directory,'video'))
            os.makedirs(os.path.join(directory,'np_arrays'))
            print('Splitting ', onlyfiles[i])
            vidcap = cv2.VideoCapture(onlyfiles[i])
            success,image = vidcap.read()
            ih,iw,ic = image.shape
            count = 0
            while success:
                image=image[:,int((iw-ih)/2):int((iw+ih)/2),:]
                image = cv2.resize(image, (args.s,args.s))
                if count == 0:
                    imseq = np.array([image])
                    inputs = np.array([])
                else:
                    prevgray = cv2.cvtColor(imseq[-1], cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    imseq = np.append(imseq, [image], axis=0)
                    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
                    flow_normalized = utils.normalize_flow(flow)
                    gray = (gray-gray.min())/(gray.max()-gray.min())
                    gray = np.reshape(gray,(args.s,args.s,1))
                    conc_img = np.concatenate((flow_normalized,gray), axis=2)
                    if count == 1:
                        inputs = np.array([conc_img])
                    else:
                        inputs = np.append(inputs, [conc_img], axis=0)
                    cv2.imwrite('%s/imgs/input%s.jpg' % (directory,str(count).zfill(5)), cv2.cvtColor((conc_img*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
                cv2.imwrite('%s/imgs/frame%s.jpg' % (directory,str(count).zfill(5)), image)
                success,image = vidcap.read()
                count += 1
            print('Ready')
            np.save('%s/np_arrays/imgs.npy' % directory, imseq)
            np.save('%s/np_arrays/inputs.npy' % directory, inputs)
            vids.append(imseq)
            dirs.append(directory)
        
    masks = nlc.demo_images(vids=vids)
    for i in range(len(vids)):
        nlc_utils.im2vid('%s/video/vid.avi' % dirs[i], vids[i], masks[i])
        np.save('%s/np_arrays/masks.npy' % dirs[i], masks[i])

if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()
    create_dataset(args)