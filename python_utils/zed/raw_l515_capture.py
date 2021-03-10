"""
capture raw images
"""

import argparse
import cv2
import numpy as np
import os
import pyrealsense2 as rs

from utils.zed.calib import RESOLUTIONS

def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-r', '--resolution', type=str, required=True,
                        choices=RESOLUTIONS.keys(), help='image resolution')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='directory to store output images')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='camera device id (default to 0)')
    parser.add_argument('-g', '--gain', type=float, default=None,
                        help='fixed gain value (default to auto gain)')
    parser.add_argument('-b', '--brightness', type=float, default=None,
                        help='fixed brightness (default to auto brightness)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    img_size_wh = RESOLUTIONS[args.resolution]
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size_wh[0] * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size_wh[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
    zed_dir = os.path.join(args.output, 'zed')
    l515_dir = os.path.join(args.output, 'l515')
    os.makedirs(zed_dir, exist_ok=True)
    os.makedirs(l515_dir, exist_ok=True)
    # setup l515
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    try:
        cnt = 0
        while True:
            # zed frame
            ret, img = cap.read()
            assert ret, 'camera not returning image'
            img_zed = img[:, :img_size_wh[0], :]
            # l515 framee
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            img_l515 = np.asanyarray(color_frame.get_data())
            # show image
            cv2.imshow('zed', img_zed)
            cv2.imshow('l515', img_l515)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('c'):
                cv2.imwrite(os.path.join(zed_dir, '{}.png'.format(cnt)), img_zed)
                cv2.imwrite(os.path.join(l515_dir, '{}.png'.format(cnt)), img_l515)
                cnt += 1
    finally:
        cap.release()

