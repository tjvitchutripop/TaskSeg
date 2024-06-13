import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torch
import sys
sys.path.append('core')
import logging
import glob
import os
import argparse
from datetime import datetime
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import *
from rlbench.observation_config import ObservationConfig
from rlbench.demo import Demo
from typing import List

import torchvision as tv
from torchvision import transforms as T
import matplotlib.pyplot as plt

import importlib

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def get_pred_flow(obs_1, obs_2):

    with torch.no_grad():
        image1 = load_image(obs_1)
        image2 = load_image(obs_2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = raft_model(image1, image2, iters=40, test_mode=True)
        return flow_up

def get_flow_magnitudes(flow):
    flow = flow[0].permute(1,2,0).cpu().numpy()
    u = flow[:,:,0]
    v = flow[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    return rad

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped

def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        if i != 0 and ((obs.gripper_open != prev_gripper_open or stopped) and not obs.gripper_open):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints

def connected_components(dilated_gt_seg, gripper_pos, pcd, max_distance = 0.1):
    # Use connected components to get closest segments to gripper
    num_labels, labeled_image = cv2.connectedComponents(dilated_gt_seg)
    close_segments = []
    for label in range(1, num_labels + 1):
        connected_component = np.column_stack(np.where(labeled_image == label))
        for point in connected_component:
            # Calculate the distance between the gripper and point cloud of pixel
            distance = np.linalg.norm(gripper_pos - pcd[point[0]][point[1]])
            if distance <= max_distance:
                close_segments.append(label)
                break
        # Check if the distance is within the specified maximum distanc
    # Create an empty binary mask with the same shape as the labeled image
    final_gt_seg_1 = np.zeros_like(labeled_image, dtype=np.uint8)
    # Iterate through each selected segment and include its pixels in the combined mask
    for label in close_segments:
        final_gt_seg_1[labeled_image == label] = 1
    return final_gt_seg_1

def generate_pseudo_gt(demo, keypoint_1, keypoint_2, camera_view):
    gripper_pos = demo[keypoint_1].gripper_pose[:3]
    if camera_view == 'front':
        obs_1 = demo[keypoint_1].front_rgb
        pcd_1 = demo[keypoint_1].front_point_cloud
        depth_1 = demo[keypoint_1].front_depth
        gt_mask = demo[keypoint_1].front_mask
    elif camera_view == 'left_shoulder':
        obs_1 = demo[keypoint_1].left_shoulder_rgb
        pcd_1 = demo[keypoint_1].left_shoulder_point_cloud
        depth_1 = demo[keypoint_1].left_shoulder_depth
        gt_mask = demo[keypoint_1].left_shoulder_mask
    elif camera_view == 'right_shoulder':
        obs_1 = demo[keypoint_1].right_shoulder_rgb
        pcd_1 = demo[keypoint_1].right_shoulder_point_cloud
        depth_1 = demo[keypoint_1].right_shoulder_depth
        gt_mask = demo[keypoint_1].right_shoulder_mask
    elif camera_view == 'overhead':
        obs_1 = demo[keypoint_1].overhead_rgb
        pcd_1 = demo[keypoint_1].overhead_point_cloud
        depth_1 = demo[keypoint_1].overhead_depth
        gt_mask = demo[keypoint_1].overhead_mask

    # Get ground truth robot segmentation for initial grasp
    robot_seg = gt_mask
    robot_seg[robot_seg>=48] = 1
    robot_seg[robot_seg<30] = 1
    robot_seg[robot_seg!=1] = 0
    mag_sum = np.zeros((obs_1.shape[0], obs_1.shape[1]))

    # Get pseudo-ground truth segmentation
    if keypoint_1 > keypoint_2:
        frame_range = range(keypoint_2, keypoint_1) # Unused in current implementation (if we ever wanted to aggreegate flow in reverse direction)
    else:
        frame_range = range(keypoint_1+1, keypoint_2+1) # Start prediction from subsequent frame to next keyframe
    for idx in frame_range:
        if camera_view == 'front':
            obs_n = demo[idx].front_rgb
        elif camera_view == 'left_shoulder':
            obs_n = demo[idx].left_shoulder_rgb
        elif camera_view == 'right_shoulder':
            obs_n = demo[idx].right_shoulder_rgb
        elif camera_view == 'overhead':
            obs_n = demo[idx].overhead_rgb
        # Use RAFT to get optical flow
        optical_flow = get_pred_flow(obs_1, obs_n)
        magnitudes = get_flow_magnitudes(optical_flow)
        # Zero out magnitudes of pixels that are part of the robot
        mag_no_robot = np.multiply(magnitudes, robot_seg)
        if mag_no_robot.max() > 0:
            mag_sum += mag_no_robot/np.max(mag_no_robot)

    norm_mag_sum = mag_sum/(len(frame_range))
    kernel = np.ones((5, 5), np.uint8)
    ground_truth_seg_loose = (norm_mag_sum >= 0.001).astype(np.uint8)
    ground_truth_seg_tight = (norm_mag_sum >= 0.05).astype(np.uint8)
    eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
    dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)
    eroded_gt_seg_tight = cv2.erode(ground_truth_seg_tight, kernel, iterations=1)
    dilated_gt_seg_tight = cv2.dilate(eroded_gt_seg_tight, kernel, iterations=1)

    # Ensure all parts of segmentation in workspace with point cloud
    for i in range(ground_truth_seg_loose.shape[0]):
        for j in range(ground_truth_seg_loose.shape[1]):
            if pcd_1[i][j][0] < -0.6 or pcd_1[i][j][0] > 0.6 or pcd_1[i][j][1] < -0.6 or pcd_1[i][j][1] > 0.6 or pcd_1[i][j][2] < 0.7525: #-0.5, 0.52, -0.55, 0.55
                dilated_gt_seg_loose[i][j] = 0  
                dilated_gt_seg_tight[i][j] = 0
    
    # Find connected component closest to gripper position
    final_gt_seg_l = connected_components(dilated_gt_seg_loose, gripper_pos, pcd_1)
    final_gt_seg_t = connected_components(dilated_gt_seg_tight, gripper_pos, pcd_1)

    return final_gt_seg_l, final_gt_seg_t, obs_1, depth_1, pcd_1, gt_mask

def get_keypoint_pseudo_gt(demo, keypoint_1, keypoint_2):
    pseudo_gt_seg_1_l, pseudo_gt_seg_1_t, rgb_1, depth_1, pcd_1, gt_mask_1 = generate_pseudo_gt(demo, keypoint_1, keypoint_2, 'front')
    pseudo_gt_seg_2_l, pseudo_gt_seg_2_t, rgb_2, depth_2, pcd_2, gt_mask_2 = generate_pseudo_gt(demo, keypoint_1, keypoint_2, 'left_shoulder')
    pseudo_gt_seg_3_l, pseudo_gt_seg_3_t, rgb_3, depth_3, pcd_3, gt_mask_3 = generate_pseudo_gt(demo, keypoint_1, keypoint_2, 'right_shoulder')
    pseudo_gt_seg_4_l, pseudo_gt_seg_4_t, rgb_4, depth_4, pcd_4, gt_mask_4 = generate_pseudo_gt(demo, keypoint_1, keypoint_2, 'overhead')
    pseudo_gt_mask_l = {'front': pseudo_gt_seg_1_l, 'left_shoulder': pseudo_gt_seg_2_l, 'right_shoulder': pseudo_gt_seg_3_l, 'overhead': pseudo_gt_seg_4_l}
    pseudo_gt_mask_t = {'front': pseudo_gt_seg_1_t, 'left_shoulder': pseudo_gt_seg_2_t, 'right_shoulder': pseudo_gt_seg_3_t, 'overhead': pseudo_gt_seg_4_t}
    rgb = {'front': rgb_1, 'left_shoulder': rgb_2, 'right_shoulder': rgb_3, 'overhead': rgb_4}
    depth = {'front': depth_1, 'left_shoulder': depth_2, 'right_shoulder': depth_3, 'overhead': depth_4}
    pcd = {'front': pcd_1, 'left_shoulder': pcd_2, 'right_shoulder': pcd_3, 'overhead': pcd_4}
    gt_mask = {'front': gt_mask_1, 'left_shoulder': gt_mask_2, 'right_shoulder': gt_mask_3, 'overhead': gt_mask_4}
    # Display RGB and pseudo ground truth
    # plt.imshow(rgb_1)
    # plt.imshow(pseudo_gt_seg_1_l, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_front_l'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_2)
    # plt.imshow(pseudo_gt_seg_2_l, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_left_shoulder_l'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_3)
    # plt.imshow(pseudo_gt_seg_3_l, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_right_shoulder_l'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_4)
    # plt.imshow(pseudo_gt_seg_4_l, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_overhead_l'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_1)
    # plt.imshow(pseudo_gt_seg_1_t, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_front_t'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_2)
    # plt.imshow(pseudo_gt_seg_2_t, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_left_shoulder_t'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_3)
    # plt.imshow(pseudo_gt_seg_3_t, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_right_shoulder_t'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
    # plt.imshow(rgb_4)
    # plt.imshow(pseudo_gt_seg_4_t, alpha=0.5)
    # plt.savefig('./test-frames/pseudo_gt_overhead_t'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)

    return pseudo_gt_mask_l, pseudo_gt_mask_t, rgb, depth, pcd, gt_mask
    
if __name__ == '__main__':
    # Specify the package name
    package_name = 'rlbench.tasks'
    # Initialize an empty list to store the imported classes
    TASKS = []
    # Use importlib to import all classes from the specified package
    package = importlib.import_module(package_name)
    # Iterate through the items in the package and filter for classes
    for name, obj in package.__dict__.items():
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'type':
            TASKS.append(obj)
    DEVICE = 'cuda'
    START_RANGE = 0
    END_RANGE = 27
    # Intervals for all RLBench Tasks (0,27) (27,54) (54,80) (80,106)s
    TASKS = TASKS[START_RANGE:END_RANGE]
    # TASKS = [PhoneOnBase, StackWine, InsertOntoSquarePeg, PlaceShapeInShapeSorter, CloseBox, PlaceCups, SweepToDustpan, OpenDoor, HitBallWithQueue, ScoopWithSpatula, InsertUsbInComputer]
    print(TASKS)
    SAMPLES = 15

    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--threshold', default=0.001, help='threshold') # Use 0.05 if want a tighter threhold for flow aggregation (currently collecting both loose and tight pseudo ground truth)
    args = parser.parse_args()

    # Set up RAFT model
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module
    raft_model.to(DEVICE)
    raft_model.eval()

    # Set up logging
    logging.basicConfig(filename='collect_data_'+str(start_time)+'.log', level=logging.DEBUG)

    live_demos = True
    DATASET = '' if live_demos else 'PATH/TO/YOUR/DATASET'
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    # Loop through each task in the range
    for idx, t in enumerate(TASKS):
        print("Starting",idx+START_RANGE)
        logging.info("Starting "+str(idx+START_RANGE))
        task = env.get_task(t)
        task_name = task.get_name()
        samples_done = 0
        demo_errors = 0
        keyframe_errors = 0
        if not os.path.exists('./training-dataset/'+"rlbench-alltasks-15"):
            os.makedirs('./training-dataset/'+"rlbench-alltasks-15")
            
        # For each task sample a number of demos
        while samples_done < SAMPLES:
            task.sample_variation()
            descriptions, obs = task.reset()
            # Get demonstration
            try:
                demo = task.get_demos(1, live_demos=live_demos)  # -> List[List[Observation]]
            except:
                print("Error getting demo for", task_name, "sample", samples_done+1)
                logging.error("Error getting demo for "+task_name+" sample "+str(samples_done+1))
                demo_errors += 1
                if demo_errors > 3:
                    break
                else:
                    continue
            demo = np.array(demo, dtype=object).flatten()

            # Get keypoints
            keypoints = keypoint_discovery(demo)
            if len(keypoints) < 1:
                print("Error with number of keyframes for", task_name, "sample", samples_done+1)
                logging.error("Error with number of keyframes for "+task_name+" sample "+str(samples_done+1))
                keyframe_errors += 1
                if keyframe_errors > 3:
                    break
                else:
                    continue  

            # Initialize lists to store data          
            pseudo_gt_masks_loose = []
            pseudo_gt_masks_tight = []
            rgb_obs = []
            depth_obs = []
            pcd_obs = []
            gt_masks = []

            # For each keypoint, get pseudo ground truth
            for keypoint_idx, keypoint in enumerate(keypoints):
                if keypoint_idx == len(keypoints)-1:
                    pseudo_gt_mask_l, pseudo_gt_mask_t, rgb, depth, pcd, gt_mask = get_keypoint_pseudo_gt(demo, keypoint, len(demo)-1)
                else:
                    pseudo_gt_mask_l, pseudo_gt_mask_t, rgb, depth, pcd, gt_mask = get_keypoint_pseudo_gt(demo, keypoint, keypoints[keypoint_idx+1])
                pseudo_gt_masks_loose.append(pseudo_gt_mask_l)
                pseudo_gt_masks_tight.append(pseudo_gt_mask_t)
                rgb_obs.append(rgb)
                depth_obs.append(depth)
                pcd_obs.append(pcd)
                gt_masks.append(gt_mask)
                
            # Save the data
            np.savez('./training-dataset/'+"rlbench-alltasks-15/"+task_name+"-"+str(samples_done)+"-action_object.npz", 
                    first_rgb = demo[0].front_rgb, 
                    final_rgb = demo[-1].front_rgb, 
                    rgb = rgb_obs,
                    desc= descriptions, 
                    depth = depth_obs, 
                    pcd = pcd_obs,
                    gt_mask = gt_masks,
                    pseudo_gt_loose = pseudo_gt_masks_loose,
                    pseudo_gt_tight = pseudo_gt_masks_tight
                    ) 
            samples_done += 1    
            print("Samples left to do:",SAMPLES-samples_done, "for", task_name)
            logging.info("Samples left to do: "+str(SAMPLES-samples_done)+ " for "+task_name)
        if samples_done < SAMPLES:
            print("Problem with", task_name)
            logging.warning("Problem with "+task_name)
    print("Done")
    logging.info("Done")
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    logging.info('Duration: {}'.format(end_time - start_time))
    env.shutdown()