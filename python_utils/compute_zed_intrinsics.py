import numpy as np
import cv2

def estimate_intrinsics(checkerboard_img_list, stereo_flag):
    assert len(checkerboard_img_list) > 1
    for i in range(len(checkerboard_img_list) - 1):
        assert checkerboard_img_list[i].shape == checkerboard_img_list[i + 1].shape
    # If using stereo camera (e.g., ZED), OpenCV capture concatenates left and right
    # frame into a single frame horizontally
    if stereo_flag:
        left_image_seq = []
        right_image_seq = []
        for img in checkerboard_img_list:
            H, W, C = img.shape
            assert W % 2 == 0
            left_img = img[:,:(W // 2),:]
            right_img = img[:,(W // 2):,:]
            left_image_seq.append(left_img)
            right_image_seq.append(right_img)
        calibration_tasks = [left_image_seq, right_image_seq]
    else:
        calibration_tasks = [checkerboard_img_list] # nested list
    # Defining the dimensions of checkerboard
    # https://raw.githubusercontent.com/opencv/opencv/4.x/doc/pattern.png
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Extracting path of individual image stored in a given directory
    for image_seq in calibration_tasks:
        for img in image_seq:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
                
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            
            cv2.imshow('img',img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

# Print ot use a computer screen to display the following chessboard pattern
# https://raw.githubusercontent.com/opencv/opencv/4.x/doc/pattern.png

def main(camera_idx, stereo_flag=True):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    checkerboard_img_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream end. Exiting ...")
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            checkerboard_img_list.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    intrinsics = estimate_intrinsics(checkerboard_img_list, stereo_flag)
    print(intrinsics)

if __name__ == '__main__':
    main(0)
