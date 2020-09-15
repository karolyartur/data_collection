## @package ginop
#  Compute metrics for a set of segmented images
#
#  This module was used to compute the comparison metrics of the predictions
#  of our model and the uNLC method on hand-labeled data.
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python metrics.py -h for help

import sys
import os
import cv2
import numpy as np
from scipy import ndimage


## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Compute metrics for a set of segmented images')
    parser.add_argument(
        '-sdir', dest='sdir',
        help='Directory relative to cwd containing images. Default is: hand_label',
        default='hand_label', type=str)
    parser.add_argument(
        '-lex', dest='lex',
        help='Unique expression in the name of hand-labeled images. Default is: label',
        default='label', type=str)
    parser.add_argument(
        '-m1ex', dest='m1ex',
        help='Unique expression in the name of segmented images created by method 1. Default is: unlc',
        default='unlc', type=str)
    parser.add_argument(
        '-m2ex', dest='m2ex',
        help='Unique expression in the name of segmented images created by method 2. Default is: p0',
        default='p0', type=str)
    parser.add_argument(
        '-t', dest='t',
        help='''Boolean value to set if temporal stability should be calculated. Any value means setting this flag to true.
        The files for temporal stability calculation can be of any name, but should be placed in a subdirectory of sdir, named t,
        and inside this directory separated into two directories, m1 and m2 for method1 and method2 respectively.
        Default vale: false''',
        type=bool)

    args = parser.parse_args()
    return args


## Create dataset
#
#  This function is used for compute the comparison metrics for a set of images.
#  @param args Object for passing the options for the calculations
def compute(args):

    import ginop


    def func(a,b):
        if a >= b:
            return(b/a)
        else:
            return(a/b)

    onlyfiles = ginop.utils.onlyfiles(args.sdir)

    labels = []
    unlc = []
    preds = []

    for file in onlyfiles:
        if args.lex in (file.split(os.sep)[-1].split('.')[0]):
            im = cv2.imread(file,0)
            im = cv2.resize(im,(30,30))
            im = cv2.threshold(im,127,1,cv2.THRESH_BINARY)[1]
            labels.append(im)
        if args.m1ex in (file.split(os.sep)[-1].split('.')[0]):
            im = cv2.imread(file,0)
            im = cv2.threshold(im,127,1,cv2.THRESH_BINARY)[1]
            unlc.append(im)
        if args.m2ex in (file.split(os.sep)[-1].split('.')[0]):
            im = cv2.imread(file,0)
            im = cv2.threshold(im,127,1,cv2.THRESH_BINARY)[1]
            preds.append(im)

    acc_p = 0
    acc_u = 0
    acc_r = 0
    size_p = 0
    size_u = 0
    size_r = 0
    center_p = 0
    center_u = 0
    center_r = 0
    dice_p = 0
    dice_u = 0
    dice_r = 0
    prec_p = 0
    prec_u = 0
    prec_r = 0
    recall_p = 0
    recall_u = 0
    recall_r = 0
    temp_p = 0
    temp_u = 0
    count = 0
    count_2 = 0
    temp = []

    for i in range(len(preds)):
        rand = np.random.randint(0,2,(30,30))
        prev = dice_r
        acc_p += (np.equal(preds[i],labels[i])).astype(int).sum()/900
        acc_u += (np.equal(unlc[i],labels[i])).astype(int).sum()/900
        acc_r += (np.equal(rand,labels[i])).astype(int).sum()/900
        if labels[i].sum() != 0:
            count += 1
            size_p += func(preds[i].sum(),labels[i].sum())
            size_u += func(unlc[i].sum(),labels[i].sum())
            size_r += func(rand.sum(),labels[i].sum())
            if preds[i].sum() != 0:
                count_2 += 1
                center_p += ((np.asarray(ndimage.measurements.center_of_mass(labels[i]))-np.asarray(ndimage.measurements.center_of_mass(preds[i])))**2).sum()**(0.5)
                center_u += ((np.asarray(ndimage.measurements.center_of_mass(labels[i]))-np.asarray(ndimage.measurements.center_of_mass(unlc[i])))**2).sum()**(0.5)
                center_r += ((np.asarray(ndimage.measurements.center_of_mass(labels[i]))-np.asarray(ndimage.measurements.center_of_mass(rand[i])))**2).sum()**(0.5)
                dice_p += 2*(preds[i]*labels[i]).sum()/(preds[i].sum()+labels[i].sum())
                dice_u += 2*(unlc[i]*labels[i]).sum()/(unlc[i].sum()+labels[i].sum())
                dice_r += 2*(rand*labels[i]).sum()/(rand.sum()+labels[i].sum())
        prec_p += (((1-preds[i])*(1-labels[i])).sum())/(1-preds[i]).sum()
        prec_u += (((1-unlc[i])*(1-labels[i])).sum())/(1-unlc[i]).sum()
        prec_r += (((1-rand)*(1-labels[i])).sum())/(1-rand).sum()
        recall_p += (((1-preds[i])*(1-labels[i])).sum())/(1-labels[i]).sum()
        recall_u += (((1-unlc[i])*(1-labels[i])).sum())/(1-labels[i]).sum()
        recall_r += (((1-rand)*(1-labels[i])).sum())/(1-labels[i]).sum()
        temp.append(dice_r-prev)

    if args.t:
        onlyfiles = ginop.utils.onlyfiles(os.path.join(args.sdir,'t','m2'))

        co = 0
        for file in onlyfiles:
            im = cv2.imread(file,0)
            im = cv2.threshold(im,127,1,cv2.THRESH_BINARY)[1]
            if co != 0:
                temp_p += func(im.sum(),prev_im.sum())
            prev_im = np.copy(im)
            co += 1
    
        onlyfiles_2 = ginop.utils.onlyfiles(os.path.join(args.sdir,'t','m1'))

        co_2 = 0
        for file in onlyfiles_2:
            im_2 = cv2.imread(file,0)
            im_2 = cv2.threshold(im_2,127,1,cv2.THRESH_BINARY)[1]
            if co_2 != 0:
                temp_u += func(im_2.sum(),prev_im_2.sum())
            prev_im_2 = np.copy(im_2)
            co_2 += 1

    acc_p /= (len(preds))
    acc_u /= (len(preds))
    acc_r /= (len(preds))
    size_p /= count
    size_u /= count
    size_r /= count
    center_p /= count_2
    center_u /= count_2
    center_r /= count_2
    dice_p /= count_2
    dice_u /= count_2
    dice_r /= count_2
    prec_p /= (len(preds))
    prec_u /= (len(preds))
    prec_r /= (len(preds))
    recall_p /= (len(preds))
    recall_u /= (len(preds))
    recall_r /= (len(preds))

    if args.t:
        temp_p /= (co-1)
        temp_u /= (co_2-1)

    print('\nPIXEL-WISE ACCURACY:')
    print('method1: ', acc_u)
    print('method2: ', acc_p)
    print('random: ', acc_r)

    print('\nSIZE ACCURACY:')
    print('method1: ', size_u)
    print('method2: ', size_p)
    print('random: ', size_r)

    print('\nDISTANCE OF CENTERS:')
    print('method2: ', center_u)
    print('method2: ', center_p)
    print('random: ', center_r)

    print('\nDICE SCORE:')
    print('method1: ', dice_u)
    print('method2: ', dice_p)
    print('random: ', dice_r)

    print('\nPRECISION AND RECALL:')
    print('method1: ', prec_u, recall_u)
    print('method2: ', prec_p, recall_p)
    print('random: ', prec_r, recall_r)

    if args.t:
        print('\nTEMPORAL STABILITY:')
        print('method1: ', temp_u)
        print('method2: ', temp_p)


if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()
    compute(args)