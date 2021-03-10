"""
capture raw images
"""

import argparse
import cv2
import numpy as np
import os

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
    cap.set(cv2.CAP_PROP_FPS, 60)
    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)
    left_dir = os.path.join(args.output, 'left')
    right_dir = os.path.join(args.output, 'right')
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    try:
        cnt = 0
        while True:
            ret, img = cap.read()
            assert ret, 'camera not returning image'
            img_left = img[:, :img_size_wh[0], :]
            img_right = img[:, img_size_wh[0]:, :]
            cv2.imshow('left', img_left)
            cv2.imshow('right', img_right)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('c'):
                cv2.imwrite(os.path.join(left_dir, '{}.png'.format(cnt)), img_left)
                cv2.imwrite(os.path.join(right_dir, '{}.png'.format(cnt)), img_right)
                cnt += 1
    finally:
        cap.release()

