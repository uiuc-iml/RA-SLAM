'''
Module for evaluating one stream in ScanNet v2
'''
import sys
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import JVisualizer
from pykdtree.kdtree import KDTree
import pymesh
from IPython import embed
from sklearn.metrics import confusion_matrix

# from dl_codebase
TRAIN_SPLIT_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt"
VAL_SPLIT_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt"
CLASS_NAMES_LIST = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
NYU_SCANNET_MAPPING = np.zeros(41) # NYU40 has 40 labels
for i in range(len(VALID_CLASS_IDS)):
    NYU_SCANNET_MAPPING[VALID_CLASS_IDS[i]] = i + 1

NUM_CLASSES = len(VALID_CLASS_IDS) + 1

class ScannetEval:
    """
    Given a semantic TSDF reconstruction of the scene and the groundtruth
    annotated triangle mesh file, compute various evaluation statistics
    """
    def __init__(self, tsdf_path, gt_poly_path, nn_method='point'):
        """
        Initialize evaluation process

        params
            - tsdf_path: path to TSDF file produced by examples/scannet_evaluation/eval_one.cc
            - gt_poly_path: path to ScanNet _vh_clean_2.labels.ply
            - nn_method: can be either 'point' or 'mesh'
                - if 'point' is specified, ground truth label of every voxel is assigned as
                    the label of the nearest *vertex* in the provided triangle mesh.
                - if 'mesh' is specified, ground truth label of every voxel is assigned as
                    the label of the nearest *triangle mesh* from the provided triangle mesh.
        """
        assert nn_method in ("point", "mesh")
        self.xyz_pc, self.predicted_label_arr = self.read_tsdf_from_file(tsdf_path)
        assert self.xyz_pc.shape[0] == self.predicted_label_arr.shape[0]
        self.gt_mesh_pymesh = pymesh.load_mesh(gt_poly_path)

        # Label Assignment
        # 1. query from KD tree to find nearest vertex in the mesh
        gt_label_arr = self.gt_mesh_pymesh.get_attribute('vertex_label').astype(np.int)
        if nn_method == "point":
            gt_label_arr = self.get_nearest_point_label(gt_label_arr)
        else:
            gt_label_arr = self.get_nearest_mesh_label(gt_label_arr)
        # 2. remove points whose ground truth id is 0 (unannotated)
        assert gt_label_arr.shape[0] == self.xyz_pc.shape[0]
        keep_idx = (gt_label_arr != 0)
        gt_label_arr = gt_label_arr[keep_idx]
        self.xyz_pc = self.xyz_pc[keep_idx,:]
        self.predicted_label_arr = self.predicted_label_arr[keep_idx,:].flatten()
        # 3. map ground truth label from nyu40 class to high touch/low touch
        self.mapped_gt_label_arr = NYU_SCANNET_MAPPING[gt_label_arr]
        self.conf_mat = confusion_matrix(self.mapped_gt_label_arr, self.predicted_label_arr, labels=range(NUM_CLASSES))
         
    def get_nearest_point_label(self, gt_label_arr):
        """
        Use KDTree to assign labels from ground truth mesh to reconstructed voxels
        """
        # pykdtree is order of magnitude faster than scipy implementation
        gt_kdtree = KDTree(self.gt_mesh_pymesh.vertices)
        _, nn_idx = gt_kdtree.query(self.xyz_pc, k = 1)
        assert self.xyz_pc.shape[0] not in nn_idx, "NN not found!"
        ret = gt_label_arr[nn_idx]
        return ret.astype(np.int)

    def get_nearest_mesh_label(self, gt_label_arr):
        """
        Use pymesh rtree implementation to find closest triangle mesh to every
        given voxel and assign label
        """
        _, nn_face_idx, _ = pymesh.distance_to_mesh(self.gt_mesh_pymesh, self.xyz_pc)
        ret = []
        faces_arr = self.gt_mesh_pymesh.faces[nn_face_idx]
        for i in range(faces_arr.shape[0]):
            vertices = faces_arr[i]
            # This is more efficient than np.all(arr == arr[0])
            if gt_label_arr[vertices[0]] != gt_label_arr[vertices[1]]\
              or gt_label_arr[vertices[0]] != gt_label_arr[vertices[2]]\
              or gt_label_arr[vertices[1]] != gt_label_arr[vertices[2]]:
                # Triangle whose vertices have inconsistent labels
                # TODO: modify this to select the nearest vertex
                label = gt_label_arr[vertices[0]]
            else:
                label = gt_label_arr[vertices[0]]
            ret.append(label)
        return np.array(ret, dtype = np.int)
    
    def get_classwise_iou(self):
        """
        in the context of binary prediction, IoU = TP / (TP + FP + FN)

        return: IoU (Intersection-over-Union) of voxels
        """
        confusion_mat = self.get_confusion_matrix()
        assert confusion_mat.shape[0] == confusion_mat.shape[1]
        assert len(confusion_mat.shape) == 2
        classwise_iou_list = []
        for i in range(confusion_mat.shape[0]):
            row_sum = np.sum(confusion_mat[i])
            col_sum = np.sum(confusion_mat[:,i])
            tp = confusion_mat[i, i]
            classwise_iou_list.append(tp / (row_sum + col_sum - tp + 1e-15))
        return classwise_iou_list
    
    def get_mean_iou(self):
        return np.mean(self.get_classwise_iou())

    def get_voxel_acc(self):
        """
        return: voxel accuracy = (TP + TN) / num_voxels
        """
        confusion_mat = self.get_confusion_matrix()
        return np.trace(confusion_mat) / np.sum(confusion_mat)

    def get_confusion_matrix(self):
        """
        Get confusiom matrix
        """
        return self.conf_mat

    @staticmethod
    def read_tsdf_from_file(tsdf_path, TSDF_THRESHOLD=0.1):
        """
        Parse binary array file dumped by examples/scannet_evaluation/eval_one.cc
        (the array is produced by download_semantic_kernel) into np.array

        return: semantic TSDF of shape (n, 5) where
            - n is the number of valid TSDF points
            - 5 represents (x, y, z, tsdf, prob)
        """
        data = np.fromfile(tsdf_path, dtype = np.float32)
        
        # raw data format: (x, y, z, tsdf, max_cls)
        # Note: max_cls is integer. Others are float32. So separate handling is needed
        data = np.reshape(data, (-1, 5))
        data = data[:,[0, 1, 2, 3]] # (N, 4)

        class_data = np.fromfile(tsdf_path, dtype=np.int32)
        class_data = class_data.reshape((-1, 5))
        class_data = class_data[:,[4]] # (N, 1)

        valid_idx = np.abs(data[:, 3]) < TSDF_THRESHOLD

        # (N, x, y, z), (N, class)
        xyz_data = data[:,[0, 1, 2]]
        return xyz_data[valid_idx,:], class_data[valid_idx,:]

if __name__ == '__main__':
    from pprint import pprint

    if len(sys.argv) > 1:
        assert len(sys.argv) == 3
        tsdf_path = sys.argv[1]
        gt_poly_path = sys.argv[2]
    else:
        tsdf_path = "/home/roger/RA-SLAM/build/raw_tsdf.bin"
        gt_poly_path = "/media/roger/My Book/data/scannet_v2/scans/scene0249_00/scene0249_00_vh_clean_2.labels.ply"

    for method in ["point"]:
        # print("======== Evaluating using nearest {} ========".format(method))
        start_cp = time.time()
        my_eval = ScannetEval(tsdf_path, gt_poly_path, nn_method=method)
        # print("Voxel Acc: {:.4f}".format(my_eval.get_voxel_acc()))
        # print("Classwise IoU: {}".format(my_eval.get_classwise_iou()))
        # print("Mean IoU: {:.4f}".format(my_eval.get_mean_iou()))
        # print("Confusion Matrix:")
        # pprint(my_eval.get_confusion_matrix())
        # print("Elapsed time: {}".format(time.time() - start_cp))
        np.save("conf_mat.npy", my_eval.get_confusion_matrix())
