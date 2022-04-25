import os
from pathlib import Path
import sys

import cv2
import pyzed.sl as sl

def svo_to_png(input_path, output_dir):
    # Make Paths
    l_dir = f"{output_dir}/l"
    r_dir = f"{output_dir}/r"
    make_path(l_dir)
    make_path(r_dir)

    # Create a ZED camera object
    zed = sl.Camera()

    # Set SVO path for playback
    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(input_path)

    # Open the ZED
    zed = sl.Camera()
    err = zed.open(init_parameters)

    l_img = sl.Mat()
    r_img = sl.Mat()
    i = 0
    step = 10

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Read side by side frames stored in the SVO
        zed.set_svo_position(i * step)
        zed.retrieve_image(l_img, sl.VIEW.LEFT)
        zed.retrieve_image(r_img, sl.VIEW.RIGHT)

        cv2.imshow("LEFT", l_img.get_data())
        cv2.imshow("RIGHT", r_img.get_data())
        cv2.waitKey(1)

        # Save Images
        save_img(l_img, f"{l_dir}/{i}.png")
        save_img(r_img, f"{r_dir}/{i}.png")
        i += 1

        # svo_position = zed.get_svo_position() # Get frame count
        # zed.set_svo_position(0) # Use this to set position

def make_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_img(mat, file_path):
    img = mat.write(file_path)
    if img != sl.ERROR_CODE.SUCCESS:
        print(f"FAILED: {repr(img)}")

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    svo_to_png(input_path, output_dir)