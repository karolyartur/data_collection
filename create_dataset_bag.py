## @package ginop
#  Create dataset from bag_file
#
#  This module can be used to create the dataset from the recorded bag files
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python create_dataset.py -h for help

import os
import cv2
import sys
import json
import yaml
import numpy as np
import pyrealsense2 as rs
import realsense_input_module as ri
import ur_tests_module as tests
from importlib import reload
from datetime import timedelta
import egomotion_filter_module as emf

## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Create Dataset From Bag Files')
    parser.add_argument(
        '-filename', dest='filename',
        help='Name of the bag file. Default is: data',
        default='data', type=str)
    parser.add_argument(
        '-tdir', dest='tdir',
        help='Directory relative to cwd for containing the dataset. Default is: data',
        default='data', type=str)
    parser.add_argument(
        '-s', dest='s',
        help='Target size for images in the dataset (s x s). Default is: 299',
        default=299, type=int)
    parser.add_argument(
        '-correct', dest='correct',
        help='Boolean swith to switch emf correction mode on and off. Default is: false (off)',
        default=False, action='store_true')

    args = parser.parse_args()
    return args


## Create dataset
#
#  This function is used for creating a dataset from bag files
#  @param args Object for passing the options of the dataset.
def create_dataset(args, pipeline, playback):

    import utils
    import videoseg.src.utils as nlc_utils
    import videoseg.src.run_full_mod as nlc
    if nlc:
        reload(nlc)

    # Emf settings
    step = 16
    dt = 1.0 / 30 #s

    with open('cam_robot_transform_config.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        reg_params = yaml.load(file, Loader=yaml.FullLoader)
        R_cam_robot = np.matrix(reg_params.get('R_cam_robot'))
        t_cam_robot = np.matrix(reg_params.get('t_cam_robot'))
    print("Camera registration read succesfully!\nR_cam_robot:\n{}\nt_cam_robot:\n{}".format(R_cam_robot,t_cam_robot))


    T_cam_robot = np.append(R_cam_robot, t_cam_robot, axis = 1)
    T_cam_robot = np.append(T_cam_robot, np.matrix('0 0 0 1'), axis = 0)

    R_robot_cam = R_cam_robot.transpose()
    t_robot_cam = -1.0 * R_robot_cam.dot(t_cam_robot)
    T_robot_cam = np.append(R_robot_cam, t_robot_cam, axis = 1)
    T_robot_cam = np.append(T_robot_cam, np.matrix('0 0 0 1'), axis = 0)

    counter = 1
    frame_num = 0

    playback_duration_in_ms = playback.get_duration()/timedelta(microseconds=1)

    vids = []
    dirs = []

    with open('data.txt', 'r') as f:
        robot_states = f.readlines()
        while abs(playback.get_position()*0.001 - playback_duration_in_ms) > 10:

            robot_velocity_at_frame = None
            robot_ang_velocity_at_frame = None

            robot_states_after = next((line for line in robot_states if float(line.split(',')[1])*1000000 > playback.get_position()*0.001))
            if robot_states.index(robot_states_after) != 0:
                robot_states_before = robot_states[robot_states.index(robot_states_after)-1]
                timestamp_before = float(robot_states_before.split(',')[1])*1000000
                timestamp_of_frame = playback.get_position()*0.001
                timestamp_after = float(robot_states_after.split(',')[1])*1000000
                robot_states_before = json.loads(robot_states_before.split(',')[0].replace(' ', ','))
                robot_states_after = json.loads(robot_states_after.split(',')[0].replace(' ', ','))
                ratio = (timestamp_of_frame - timestamp_before) / (timestamp_after - timestamp_before)
                robot_velocity_at_frame = ratio * (robot_states_after[3] - robot_states_before[3]) + robot_states_before[3]
                robot_ang_velocity_at_frame = ratio * (robot_states_after[4] - robot_states_before[4]) + robot_states_before[4]
                # print('Before: ' + str(timestamp_before) + ', Frame: ' + str(timestamp_of_frame) + ', After: ' + str(timestamp_after))
                # print('Vel. before: ' + str(robot_states_before[3]) + ', Vel. after: ' + str(robot_states_after[3]) + ', Vel: ' + str(robot_velocity_at_frame))

            directory = os.path.join(args.tdir,str(counter).zfill(5))
            if not os.path.exists(directory):
                os.makedirs(os.path.join(directory,'imgs'))
                os.makedirs(os.path.join(directory,'video'))
                os.makedirs(os.path.join(directory,'np_arrays'))

            # Wait for a coherent pair of frames: depth and color
            frames_sr300 = ri.get_frames(pipeline)
            depth_frame = ri.get_depth_frames(frames_sr300)
            color_frame = ri.get_color_frames(frames_sr300)
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = ri.convert_img_to_nparray(depth_frame)
            color_image = ri.convert_img_to_nparray(color_frame)
            image = color_image

            ih,iw,ic = color_image.shape
            if args.correct:
                color_image_original = color_image
            color_image=color_image[:,int((iw-ih)/2):int((iw+ih)/2),:]
            color_image = cv2.resize(color_image, (args.s,args.s))


            if frame_num == 0:
                imseq = np.array([color_image])
                inputs = np.array([])
                if args.correct:
                    imseq_original = np.array([color_image_original])

                    # Align depth to color
                    depth_frame_prev_aligned = emf.align_depth_to_color(frames_sr300)

                    # Deprojection
                    deproject_flow_prev = emf.deproject(depth_frame_prev_aligned, step=step)
            else:
                if args.correct:
                    prevgray_original = cv2.cvtColor(imseq_original[-1], cv2.COLOR_BGR2GRAY)
                    gray_original = cv2.cvtColor(color_image_original, cv2.COLOR_BGR2GRAY)
                    imseq_original = np.append(imseq_original, [color_image_original], axis=0)
                    flow_original = cv2.calcOpticalFlowFarneback(prevgray_original, gray_original, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
                prevgray = cv2.cvtColor(imseq[-1], cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                imseq = np.append(imseq, [color_image], axis=0)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)
                ri.show_imgs('Optical flow', emf.draw_flow(gray, flow.astype(int), step=step), 1)
                if args.correct:
                    # Get the previous and the current pixel locations
                    lines = emf.get_lines(gray_original, flow_original, step=step)

                    # Align depth to color
                    depth_frame_aligned = emf.align_depth_to_color(frames_sr300)

                    # Deproject the current pixels for 3D optical flow
                    deproject_flow = emf.deproject_flow_new(depth_frame_aligned, lines, step=step)

                    # Calculate 3D optical flow
                    diff_flow = emf.flow_3d(deproject_flow, deproject_flow_prev, dt)
                    diff_flow_robot = emf.transform_velocites(diff_flow, T_robot_cam)

                    if robot_velocity_at_frame is not None and robot_ang_velocity_at_frame is not None:
                        v_robot = np.matrix([[robot_velocity_at_frame],  [0] , [0]])
                        omega_robot = np.matrix([[0],  [0] , [robot_ang_velocity_at_frame]])

                        deprojected_coordinates_robot = emf.transform_points(deproject_flow, T_robot_cam)

                        velocities_from_egomotion_robot = \
                            emf.velocity_from_point_clouds_robot_frame(deprojected_coordinates_robot, \
                                            v_robot, omega_robot)
                        
                        threshold_lower = abs(v_robot[0][0])*0.001
                        threshold_upper = 0.5

                        egomotion_filtered_flow = \
                            emf.velocity_comparison(depth_frame_aligned, \
                                            diff_flow_robot, \
                                            velocities_from_egomotion_robot, \
                                            threshold_lower, threshold_upper, step=step)

                        flow = emf.filtered_flow_2d(egomotion_filtered_flow, \
                                        flow_original, step=step)

                        flow = flow[:,int((iw-ih)/2):int((iw+ih)/2)]
                        flow = cv2.resize(flow, (299,299))

                        ri.show_imgs('Optical flow filtered', \
                            emf.draw_flow(gray, flow, step=step), 1)
                
                flow_normalized = utils.normalize_flow(flow)
                gray = (gray-gray.min())/(gray.max()-gray.min())
                gray = np.reshape(gray,(args.s,args.s,1))
                conc_img = np.concatenate((flow_normalized,gray), axis=2)

                if frame_num == 1:
                    inputs = np.array([conc_img])
                else:
                    inputs = np.append(inputs, [conc_img], axis=0)

                if args.correct:
                    deproject_flow_prev = deproject_flow
                    depth_frame_prev_aligned = depth_frame_aligned
            
                cv2.imwrite('%s/imgs/input%s.jpg' % (directory,str(frame_num).zfill(5)), cv2.cvtColor((conc_img*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite('%s/imgs/frame%s.jpg' % (directory,str(frame_num).zfill(5)), color_image)


            frame_num += 1

            if frame_num == 100 or not abs(playback.get_position()*0.001 - playback_duration_in_ms) > 10:
                frame_num = 0
                counter += 1
                np.save('%s/np_arrays/imgs.npy' % directory, imseq)
                np.save('%s/np_arrays/inputs.npy' % directory, inputs)
                vids.append(imseq)
                dirs.append(directory)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = ri.get_depth_colormap(depth_image, 0.03)

            # Stack both images horizontally
            images = np.hstack((image, depth_colormap))

            # Show images
            ri.show_imgs('SR300', images, 1)
    
    cv2.destroyAllWindows()
    masks = nlc.demo_images(vids=vids)
    for i in range(len(vids)):
        nlc_utils.im2vid('%s/video/vid.avi' % dirs[i], vids[i], masks[i])
        np.save('%s/np_arrays/masks.npy' % dirs[i], masks[i])

if __name__ == "__main__":
    sys.path.insert(0,'..')
    args = parse_args()

    if not os.path.exists(args.tdir):
        os.makedirs(args.tdir)

    # Get configs for the two streams
    config_1 = ri.get_config()

    # Get pipelines for the two streams
    pipeline_1 = ri.get_pipeline()

    profile_1, playback = tests.start_sr300_2(config_1, pipeline_1, 640, 480, 30, args.filename +'.bag')

    try:
        create_dataset(args, pipeline_1, playback)
    
    except KeyboardInterrupt:
        pass

    finally:
        # Stop streaming
        pipeline_1.stop()