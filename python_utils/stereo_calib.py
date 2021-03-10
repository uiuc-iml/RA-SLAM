import cv2
import numpy as np
import yaml

def parse_calibration_yaml(calibration_file):
    with open(calibration_file, 'r') as f:
        yaml_node = yaml.safe_load(f)

    left_cam_fx = yaml_node['Calibration.left.fx']
    left_cam_fy = yaml_node['Calibration.left.fy']
    left_cam_cx = yaml_node['Calibration.left.cx']
    left_cam_cy = yaml_node['Calibration.left.cy']
    left_cam_distortion = np.array(yaml_node['Calibration.left.distortion'])
    left_cam_matrix = np.array([
        [left_cam_fx, 0, left_cam_cx],
        [0, left_cam_fy, left_cam_cy],
        [0, 0, 1]
    ])

    right_cam_fx = yaml_node['Calibration.right.fx']
    right_cam_fy = yaml_node['Calibration.right.fy']
    right_cam_cx = yaml_node['Calibration.right.cx']
    right_cam_cy = yaml_node['Calibration.right.cy']
    right_cam_distortion = np.array(yaml_node['Calibration.right.distortion'])
    right_cam_matrix = np.array([
        [right_cam_fx, 0, right_cam_cx],
        [0, right_cam_fy, right_cam_cy],
        [0, 0, 1]
    ])

    R, _ = cv2.Rodrigues(np.array(yaml_node['Calibration.rotation']))
    T = np.array(yaml_node['Calibration.translation'])

    return left_cam_matrix, right_cam_matrix, left_cam_distortion, right_cam_distortion, R, T

