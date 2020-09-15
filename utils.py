## @package ginop
#  Util module
#
#  This module contains utility functions for the package.
#  It gets automatically imported when a module of the project or the project itself is imported.

## Normalize optical flow
#
#  This function normalizes optical flow. The x and y channels are normalized separately.
#
#  Return value is np.float32, shape(h,w,c), the normalized optical flow.
#  @param flow Optical flow of np.float32, shape:(w,h,c), where w is width, h is height and c is channels
def normalize_flow(flow):

    import numpy as np

    flow_x_channel = flow[:,:,0]
    flow_y_channel = flow[:,:,1]

    flow_min_x = flow_x_channel.min()
    flow_max_x = flow_x_channel.max()
    
    flow_min_y = flow_y_channel.min()
    flow_max_y = flow_y_channel.max()
    
    flow_normalized_x = (flow_x_channel - flow_min_x)/(flow_max_x - flow_min_x) 

    flow_normalized_y = (flow_y_channel - flow_min_y)/(flow_max_y - flow_min_y) 

    flow_normalized_both_channel = np.array([flow_normalized_x, flow_normalized_y])
    flow_normalized_both_channel_transposed = np.transpose(flow_normalized_both_channel, (1,2,0))

    return flow_normalized_both_channel_transposed


## Return a list of files in a directory
#
#  This function returns the list of files, and the files only in a selected directory.
#
#  Returns a list containing the name of files in the directory.
#  @param dir A string containing the directory path
def onlyfiles(dir):

    import os

    return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


## Dataset options class
#
#  This class is for using the create_dataset and create_dataset_i scripts in python modules.
#  The objects of this class can be passed to the create_dataset and the formulate_inputs_masks functions.
#  Change the attributes of the object to set options for the dataset.
class DS():
    def __init__(self):
        self.sdir = 'vids'
        self.tdir = 'data'
        self.s = 299


## Metrics options class
#
#  This class is for using the metrics script in python code.
#  The objects of this class can be passed to the compute function.
#  Change the attributes of the object to set options for the metrics.
class MET():
    def __init__(self):
        self.sdir = 'hand_label'
        self.lex = 'label'
        self.m1ex = 'unlc'
        self.m2ex = 'p0'
        self.t = False

## Record video options class
#
#  This class is for using the record_videos script in python code.
#  The objects of this class can be passed to the record_videos function.
#  Change the attributes of the object to set options for the metrics.
class VID():
    def __init__(self):
        self.tdir = 'vids'