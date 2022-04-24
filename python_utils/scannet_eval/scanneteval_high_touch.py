'''
Module for evaluating one stream in ScanNet v2
'''
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import JVisualizer
from pykdtree.kdtree import KDTree
from utils import LabelParser
import pymesh

my_labelparser = LabelParser()
nyuid_to_ht_map = my_labelparser.get_nyuid_to_ht_map()

class ScannetEval:
    """
    Given a semantic TSDF reconstruction of the scene and the groundtruth
    annotated triangle mesh file, compute various evaluation statistics
    """
    def __init__(self, tsdf_path, gt_poly_path, p_cutoff = 0.5, nn_method = "point"):
        """
        Initialize evaluation process

        params
            - tsdf_path: path to TSDF file produced by examples/scannet_evaluation/eval_one.cc
            - gt_poly_path: path to ScanNet _vh_clean_2.labels.ply
            - p_cutoff: high touch probability cut off
            - nn_method: can be either 'point' or 'mesh'
                - if 'point' is specified, ground truth label of every voxel is assigned as
                    the label of the nearest *vertex* in the provided triangle mesh.
                - if 'mesh' is specified, ground truth label of every voxel is assigned as
                    the label of the nearest *triangle mesh* from the provided triangle mesh.
        """
        assert nn_method in ("point", "mesh")
        self.tsdf_np = self.read_tsdf_from_file(tsdf_path)
        self.semantic_pc = self.tsdf_to_semantic_pc(self.tsdf_np)
        self.xyz_pc = self.semantic_pc[:, :3]
        self.gt_mesh_pymesh = pymesh.load_mesh(gt_poly_path)

        # Label Assignment
        # 1. query from KD tree to find nearest vertex in the mesh
        gt_label_arr = self.gt_mesh_pymesh.get_attribute('vertex_label').astype(np.int)
        if nn_method == "point":
            gt_label_arr = self.get_nearest_point_label(gt_label_arr)
        else:
            gt_label_arr = self.get_nearest_mesh_label(gt_label_arr)
        # 2. remove points whose ground truth id is 0 (unannotated)
        keep_idx = (gt_label_arr != 0)
        gt_label_arr = gt_label_arr[keep_idx]
        # 3. map ground truth label from nyu40 class to high touch/low touch
        self.gt_high_touch_arr = np.zeros_like(gt_label_arr)
        for i in range(gt_label_arr.shape[0]):
            self.gt_high_touch_arr[i] = nyuid_to_ht_map[gt_label_arr[i]]
        # 4. Assign labels
        # TODO: make a PR curve by varying p_cutoff (e.g., 0.05)
        self.predicted_label_arr = (self.semantic_pc[:, 3] > p_cutoff).astype(np.int)
        self.predicted_label_arr = self.predicted_label_arr[keep_idx]
    
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
    
    def get_iou(self):
        """
        in the context of binary prediction, IoU = TP / (TP + FP + FN)

        return: IoU (Intersection-over-Union) of voxels
        """
        confusion_mat = self.get_confusion_matrix()
        return confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1] + confusion_mat[1, 0] + 1e-15)

    def get_voxel_acc(self):
        """
        return: voxel accuracy = (TP + TN) / num_voxels
        """
        confusion_mat = self.get_confusion_matrix()
        return (confusion_mat[0, 0] + confusion_mat[1, 1]) / np.sum(confusion_mat)

    def get_per_class_accuracy(self):
        raise NotImplementedError

    def get_precision(self):
        """
        return: precision = TP / (TP + FP)
        """
        confusion_mat = self.get_confusion_matrix()
        return confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1] + 1e-15)

    def get_recall(self):
        """
        return: recall = TP / (TP + FN)
        """
        confusion_mat = self.get_confusion_matrix()
        return confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[1, 0] + 1e-15)

    def get_confusion_matrix(self):
        """
        Get TP/TN/FP/FN confusion matrix for high touch class

        return: np.array([
            [TP, FP],
            [FN, TN]
        ])
        
        where every element is an integer counter
        """
        tp_count = np.logical_and(self.predicted_label_arr == 1, self.gt_high_touch_arr == 1)
        tp_count = np.sum(tp_count)
        tn_count = np.logical_and(self.predicted_label_arr == 0, self.gt_high_touch_arr == 0)
        tn_count = np.sum(tn_count)
        fp_count = np.logical_and(self.predicted_label_arr == 1, self.gt_high_touch_arr == 0)
        fp_count = np.sum(fp_count)
        fn_count = np.logical_and(self.predicted_label_arr == 0, self.gt_high_touch_arr == 1)
        fn_count = np.sum(fn_count)
        return np.array([[tp_count, fp_count], [fn_count, tn_count]])

    @staticmethod
    def read_tsdf_from_file(tsdf_path):
        """
        Parse binary array file dumped by examples/scannet_evaluation/eval_one.cc
        (the array is produced by download_semantic_kernel) into np.array

        return: semantic TSDF of shape (n, 5) where
            - n is the number of valid TSDF points
            - 5 represents (x, y, z, tsdf, prob)
        """
        data = np.fromfile(tsdf_path, dtype = np.float32)
        
        # data format: (x, y, z, tsdf, prob)
        data = np.reshape(data, (-1, 5))
        return data

    @staticmethod
    def tsdf_to_semantic_pc(tsdf_np, TSDF_THRESHOLD = 0.1):
        """
        Parse TSDF file into semantic point cloud

        params:
            - tsdf_np: (n, 5), where each row is (x, y, z, tsdf, prob)
        
        return: semantic_pc: (m, 4), where each row is (x, y, z, prob),
            and $m < n$ where some voxels with large distance to nearest
            surface are eliminated.
        """
        semantic_pc = tsdf_np[np.abs(tsdf_np[:, 3]) < TSDF_THRESHOLD, :]
        return semantic_pc[:,[0, 1, 2, 4]]

if __name__ == '__main__':
    from pprint import pprint

    tsdf_path = "/tmp/data.bin"
    gt_poly_path = "/media/roger/My Book/data/scannet_v2/scans/scene0327_00/scene0327_00_vh_clean_2.labels.ply"

    for method in ["point", "mesh"]:
        print("======== Evaluating using nearest {} ========".format(method))
        start_cp = time.time()
        my_eval = ScannetEval(tsdf_path, gt_poly_path, nn_method=method)
        print("Voxel Acc: {:.4f}".format(my_eval.get_voxel_acc()))
        print("Precision: {:.4f}".format(my_eval.get_precision()))
        print("Recall: {:.4f}".format(my_eval.get_recall()))
        print("High touch IoU: {:.4f}".format(my_eval.get_iou()))
        print("Confusion Matrix:")
        pprint(my_eval.get_confusion_matrix())
        print("Elapsed time: {}".format(time.time() - start_cp))
